//! Pression faiblement compressible IMPLICITE (Helmholtz).
//!
//! Fermeture volumique : à la fin du pas, Σ_k m_k/ρ_k(p, T) = 1 dans chaque cellule, avec les masses m_k déjà advectées.
//! Linéarisée (Newton, 2–4 itérations) autour de p_k :  V(p) ≈ V(p_k, T_k) − κ_eff (p − p_k),
//! κ_eff = Σ V_i κ_T,i − T (Σ V_i β_i)² / Σ m_i cp_i   (compressibilité à entropie constante du mélange : gaz → 1/(γp),
//! bulle dans l'eau → quasi isotherme).
//! Équation : −∇·(dt/ρ_f ∇δp) + (κ_eff/dt) δp = (V−1)/dt − ∇·(Ũ − uⁿ),   Ũ = u* − dt/ρ_f ∇p_k,
//! uⁿ⁺¹ = Ũ − dt/ρ_f ∇δp. Le terme acoustique κ/dt est implicite : le pas de temps n'est PAS limité par c ≈ 1480 m/s.
//! Matrice SPD résolue par gradients conjugués préconditionnés par Cholesky incomplet IC(0).

use super::multiphase::{Bc, MultiphaseSim};

/// pression absolue minimale d'une cellule fluide (Pa)
pub(crate) const P_FLOOR: f64 = 10.0;

struct Stencil {
    diag: Vec<f64>,
    ae: Vec<f64>,
    an: Vec<f64>,
}

impl MultiphaseSim {
    /// Volume du mélange V, κ_eff et γ_T = dT/dp (adiabatique) d'une cellule à (p, T).
    pub(crate) fn cell_closure(&self, c: usize, p: f64, t: f64) -> (f64, f64, f64) {
        let (mut v, mut kap, mut vb, mut mcp) = (0.0, 0.0, 0.0, 0.0);
        for (k, ph) in self.reg.phases.iter().enumerate() {
            let m = self.mass[k][c];
            if m <= 0.0 {
                continue;
            }
            let vk = m / ph.rho(p, t);
            v += vk;
            kap += vk * ph.kappa_t(p, t);
            vb += vk * ph.beta(p, t);
            mcp += m * ph.cp;
        }
        let gamma_t = t * vb / mcp.max(1e-30);
        (v, kap - t * vb * vb / mcp.max(1e-30), gamma_t)
    }

    /// vitesse de face corrigée Ũ = u* − dt/ρ_f ∇p (0 sur parois/solides)
    fn face_velocities(&self, us: &[f64], vs: &[f64], p: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let mut un = vec![0.0; (nx + 1) * ny];
        let mut vn = vec![0.0; nx * (ny + 1)];
        for j in 0..ny {
            for i in 0..=nx {
                let f = j * (nx + 1) + i;
                let (l, r) = (if i > 0 { Some(j * nx + i - 1) } else { None }, if i < nx { Some(j * nx + i) } else { None });
                un[f] = match (l, r) {
                    (Some(a), Some(b)) => {
                        if self.is_solid(a) || self.is_solid(b) { 0.0 } else { us[f] - dt / self.face_rho_x(i, j) * (p[b] - p[a]) / dx }
                    }
                    (None, Some(b)) => match (self.bc[0], self.is_solid(b)) {
                        (Bc::Open { p: po }, false) => us[f] - dt / self.rho_mix[b] * (p[b] - po) / (0.5 * dx),
                        _ => 0.0,
                    },
                    (Some(a), None) => match (self.bc[1], self.is_solid(a)) {
                        (Bc::Open { p: po }, false) => us[f] - dt / self.rho_mix[a] * (po - p[a]) / (0.5 * dx),
                        _ => 0.0,
                    },
                    _ => 0.0,
                };
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let f = j * nx + i;
                let (l, r) = (if j > 0 { Some((j - 1) * nx + i) } else { None }, if j < ny { Some(j * nx + i) } else { None });
                vn[f] = match (l, r) {
                    (Some(a), Some(b)) => {
                        if self.is_solid(a) || self.is_solid(b) { 0.0 } else { vs[f] - dt / self.face_rho_y(i, j) * (p[b] - p[a]) / dx }
                    }
                    (None, Some(b)) => match (self.bc[2], self.is_solid(b)) {
                        (Bc::Open { p: po }, false) => vs[f] - dt / self.rho_mix[b] * (p[b] - po) / (0.5 * dx),
                        _ => 0.0,
                    },
                    (Some(a), None) => match (self.bc[3], self.is_solid(a)) {
                        (Bc::Open { p: po }, false) => vs[f] - dt / self.rho_mix[a] * (po - p[a]) / (0.5 * dx),
                        _ => 0.0,
                    },
                    _ => 0.0,
                };
            }
        }
        for (f, p) in self.pf_u.iter().enumerate() {
            if !p.is_nan() {
                un[f] = *p;
            }
        }
        for (f, p) in self.pf_v.iter().enumerate() {
            if !p.is_nan() {
                vn[f] = *p;
            }
        }
        (un, vn)
    }

    pub(crate) fn divergence_of(&self, u: &[f64], v: &[f64], i: usize, j: usize) -> f64 {
        let (nx, dx) = (self.nx, self.dx);
        (u[j * (nx + 1) + i + 1] - u[j * (nx + 1) + i] + v[(j + 1) * nx + i] - v[j * nx + i]) / dx
    }

    /// Couplage de pression entre deux cellules voisines a, b à travers une face : 0 = aucun (solide, face à vitesse imposée),
    /// 1 = couplage complet, 2 = Dirichlet (une des deux est épinglée : δp = 0 de son côté, terme de diagonale seulement)
    fn link(&self, a: usize, b: usize, fixed: bool) -> u8 {
        if fixed || self.is_solid(a) || self.is_solid(b) {
            0
        } else if self.pinned(a) || self.pinned(b) {
            2
        } else {
            1
        }
    }

    fn build_stencil(&self, kappa: &[f64], dt: f64) -> Stencil {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut s = Stencil { diag: vec![1.0; n], ae: vec![0.0; n], an: vec![0.0; n] };
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                if self.is_solid(c) || self.pinned(c) {
                    continue; // ligne identité (δp = 0)
                }
                let mut d = kappa[c] / dt;
                // est
                if i + 1 < nx {
                    let k = self.link(c, c + 1, self.face_u_fixed(self.uidx(i + 1, j)));
                    if k > 0 {
                        let t = dt / (self.face_rho_x(i + 1, j) * dx * dx);
                        if k == 1 {
                            s.ae[c] = t;
                        }
                        d += t;
                    }
                } else if let Bc::Open { .. } = self.bc[1] {
                    d += 2.0 * dt / (self.rho_mix[c] * dx * dx);
                }
                // ouest
                if i > 0 {
                    if self.link(c - 1, c, self.face_u_fixed(self.uidx(i, j))) > 0 {
                        d += dt / (self.face_rho_x(i, j) * dx * dx);
                    }
                } else if let Bc::Open { .. } = self.bc[0] {
                    d += 2.0 * dt / (self.rho_mix[c] * dx * dx);
                }
                // nord
                if j + 1 < ny {
                    let k = self.link(c, c + nx, self.face_v_fixed(self.vidx(i, j + 1)));
                    if k > 0 {
                        let t = dt / (self.face_rho_y(i, j + 1) * dx * dx);
                        if k == 1 {
                            s.an[c] = t;
                        }
                        d += t;
                    }
                } else if let Bc::Open { .. } = self.bc[3] {
                    d += 2.0 * dt / (self.rho_mix[c] * dx * dx);
                }
                // sud
                if j > 0 {
                    if self.link(c - nx, c, self.face_v_fixed(self.vidx(i, j))) > 0 {
                        d += dt / (self.face_rho_y(i, j) * dx * dx);
                    }
                } else if let Bc::Open { .. } = self.bc[2] {
                    d += 2.0 * dt / (self.rho_mix[c] * dx * dx);
                }
                s.diag[c] = d;
            }
        }
        s
    }

    /// Résout A x = b (PCG + IC(0)). Retourne le nombre d'itérations.
    fn pcg(&self, st: &Stencil, b: &[f64], x: &mut [f64]) -> usize {
        let nx = self.nx;
        let n = b.len();
        let mut l = vec![1.0; n];
        let (mut lw, mut ls) = (vec![0.0; n], vec![0.0; n]);
        for c in 0..n {
            let mut d = st.diag[c];
            if c % nx > 0 {
                lw[c] = -st.ae[c - 1] / l[c - 1];
                d -= lw[c] * lw[c];
            }
            if c >= nx {
                ls[c] = -st.an[c - nx] / l[c - nx];
                d -= ls[c] * ls[c];
            }
            l[c] = d.max(1e-300).sqrt();
        }
        let precond = |r: &[f64], z: &mut [f64], y: &mut [f64]| {
            for c in 0..n {
                let mut s = r[c];
                if lw[c] != 0.0 {
                    s -= lw[c] * y[c - 1];
                }
                if ls[c] != 0.0 {
                    s -= ls[c] * y[c - nx];
                }
                y[c] = s / l[c];
            }
            for c in (0..n).rev() {
                let mut s = y[c];
                if c + 1 < n && lw[c + 1] != 0.0 {
                    s -= lw[c + 1] * z[c + 1];
                }
                if c + nx < n && ls[c + nx] != 0.0 {
                    s -= ls[c + nx] * z[c + nx];
                }
                z[c] = s / l[c];
            }
        };
        let apply = |x: &[f64], out: &mut [f64]| {
            for c in 0..n {
                let mut s = st.diag[c] * x[c];
                if st.ae[c] != 0.0 {
                    s -= st.ae[c] * x[c + 1];
                }
                if c % nx > 0 && st.ae[c - 1] != 0.0 {
                    s -= st.ae[c - 1] * x[c - 1];
                }
                if st.an[c] != 0.0 {
                    s -= st.an[c] * x[c + nx];
                }
                if c >= nx && st.an[c - nx] != 0.0 {
                    s -= st.an[c - nx] * x[c - nx];
                }
                out[c] = s;
            }
        };
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        apply(x, &mut ax);
        for c in 0..n {
            r[c] = b[c] - ax[c];
        }
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let (mut z, mut y) = (vec![0.0; n], vec![0.0; n]);
        precond(&r, &mut z, &mut y);
        let mut pdir = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut it = 0;
        while it < self.pressure_max_iter {
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn <= self.pressure_tol * bn {
                break;
            }
            apply(&pdir, &mut ax);
            let pap: f64 = pdir.iter().zip(&ax).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-300 {
                break;
            }
            let al = rz / pap;
            for c in 0..n {
                x[c] += al * pdir[c];
                r[c] -= al * ax[c];
            }
            precond(&r, &mut z, &mut y);
            let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let be = rz2 / rz.max(1e-300);
            rz = rz2;
            for c in 0..n {
                pdir[c] = z[c] + be * pdir[c];
            }
            it += 1;
        }
        it
    }

    /// Projette (u*, v*) : met à jour self.p et self.u/self.v (u^{n+1}). `un`, `vn` = vitesses du début de pas (celles qui ont
    /// servi à advecter les masses). Retourne le nombre total d'itérations PCG.
    pub(crate) fn solve_pressure(&mut self, dt: f64, us: &[f64], vs: &[f64], un: &[f64], vn: &[f64]) -> usize {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let p_start = self.p.clone();
        let t_star: Vec<f64> = (0..n).map(|c| self.temp[c]).collect();
        let mut pk = self.p.clone();
        let mut total_it = 0;
        let mut kappa = vec![0.0; n];
        for _newton in 0..4 {
            let mut vol = vec![1.0; n];
            for c in 0..n {
                if self.is_solid(c) {
                    continue;
                }
                if self.pinned(c) {
                    kappa[c] = 1.0;
                    continue; // cellule épinglée : δp = 0
                }
                let (_, _, g0) = self.cell_closure(c, p_start[c], t_star[c]);
                let tk = t_star[c] + g0 * (pk[c] - p_start[c]);
                let (v, kap, _) = self.cell_closure(c, pk[c], tk);
                vol[c] = v;
                kappa[c] = kap.max(1e-30);
            }
            let st = self.build_stencil(&kappa, dt);
            let (ut, vt) = self.face_velocities(us, vs, &pk, dt);
            let mut b = vec![0.0; n];
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    if self.is_solid(c) || self.pinned(c) {
                        continue;
                    }
                    let d_new = self.divergence_of(&ut, &vt, i, j);
                    let d_old = self.divergence_of(un, vn, i, j);
                    b[c] = (vol[c] - 1.0) / dt - (d_new - d_old);
                }
            }
            let mut dp = vec![0.0; n];
            total_it += self.pcg(&st, &b, &mut dp);
            let mut dmax = 0.0f64;
            for c in 0..n {
                if !self.is_solid(c) && !self.pinned(c) {
                    pk[c] += dp[c];
                    // Plancher de pression : aucun fluide ne descend sous P_FLOOR (vide / cavitation). Avant, seules les cellules contenant
                    // du gaz étaient protégées : une cellule quasi vide d'eau seule pouvait atteindre −8e7 Pa, puis le travail de pression
                    // −dt·p·∇·u sur sa capacité calorifique minuscule donnait T = −1.6e5 K → NaN.
                    if !pk[c].is_finite() {
                        pk[c] = p_start[c].max(P_FLOOR);
                    } else if pk[c] < P_FLOOR {
                        pk[c] = P_FLOOR;
                    }
                    dmax = dmax.max(dp[c].abs());
                }
            }
            if dmax < 1e-3 {
                break;
            }
        }
        let (uf, vf) = self.face_velocities(us, vs, &pk, dt);
        self.u = uf;
        self.v = vf;
        self.p = pk;
        total_it
    }
}
