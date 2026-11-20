//! Advection : (1) masses partielles + énergie + fractions par balayages PLIC conservatifs (flux géométriques exacts),
//! (2) vitesse par semi-lagrangien MacCormack limité.

use super::multiphase::{clampf, Bc, MultiphaseSim};
use super::vof::Plic;

/// Interpolation bilinéaire d'un champ w×h aux indices (gx, gy) (bornés), renvoie (valeur, min, max) des 4 voisins.
fn bilerp(f: &[f64], w: usize, h: usize, gx: f64, gy: f64) -> (f64, f64, f64) {
    let gx = gx.clamp(0.0, (w - 1) as f64);
    let gy = gy.clamp(0.0, (h - 1) as f64);
    let (i0, j0) = ((gx.floor() as usize).min(w.saturating_sub(2)), (gy.floor() as usize).min(h.saturating_sub(2)));
    let (i1, j1) = ((i0 + 1).min(w - 1), (j0 + 1).min(h - 1));
    let (fx, fy) = (gx - i0 as f64, gy - j0 as f64);
    let (a, b, c, d) = (f[j0 * w + i0], f[j0 * w + i1], f[j1 * w + i0], f[j1 * w + i1]);
    let v = (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy;
    (v, a.min(b).min(c).min(d), a.max(b).max(c).max(d))
}

/// Interpolation bicubique de Catmull–Rom, BORNÉE par min/max des 4 voisins (ne crée aucun extremum : pas de création d'énergie).
fn bicubic_limited(f: &[f64], w: usize, h: usize, gx: f64, gy: f64) -> f64 {
    let (lin, lo, hi) = bilerp(f, w, h, gx, gy);
    let gx = gx.clamp(0.0, (w - 1) as f64);
    let gy = gy.clamp(0.0, (h - 1) as f64);
    let (i0, j0) = (gx.floor() as isize, gy.floor() as isize);
    let (fx, fy) = (gx - i0 as f64, gy - j0 as f64);
    let at = |i: isize, j: isize| f[(j.clamp(0, h as isize - 1) as usize) * w + i.clamp(0, w as isize - 1) as usize];
    let cr = |p0: f64, p1: f64, p2: f64, p3: f64, t: f64| {
        p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
    };
    let mut col = [0.0; 4];
    for (q, dj) in (-1isize..=2).enumerate() {
        col[q] = cr(at(i0 - 1, j0 + dj), at(i0, j0 + dj), at(i0 + 1, j0 + dj), at(i0 + 2, j0 + dj), fx);
    }
    let v = cr(col[0], col[1], col[2], col[3], fy);
    if v.is_finite() { clampf(v, lo, hi) } else { lin }
}

impl MultiphaseSim {
    fn sample_u(&self, u: &[f64], x: f64, y: f64) -> (f64, f64, f64) {
        bilerp(u, self.nx + 1, self.ny, x / self.dx, y / self.dx - 0.5)
    }
    fn sample_v(&self, v: &[f64], x: f64, y: f64) -> (f64, f64, f64) {
        bilerp(v, self.nx, self.ny + 1, x / self.dx - 0.5, y / self.dx)
    }
    fn vel_at(&self, u: &[f64], v: &[f64], x: f64, y: f64) -> (f64, f64) {
        (self.sample_u(u, x, y).0, self.sample_v(v, x, y).0)
    }
    /// Pied de la caractéristique (RK2) ; `s` = ±1 (arrière / avant)
    fn trace(&self, u: &[f64], v: &[f64], x: f64, y: f64, dt: f64, s: f64) -> (f64, f64) {
        let (a, b) = self.vel_at(u, v, x, y);
        let (xm, ym) = (x - 0.5 * s * dt * a, y - 0.5 * s * dt * b);
        let (a2, b2) = self.vel_at(u, v, xm, ym);
        (x - s * dt * a2, y - s * dt * b2)
    }

    /// Advection de la vitesse (MacCormack limité) avec le champ (u, v) lui-même. Renvoie les nouveaux champs.
    pub(crate) fn advect_velocity(&self, dt: f64) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (u0, v0) = (&self.u, &self.v);
        let mut un = u0.clone();
        let mut vn = v0.clone();
        // composante u
        let mut ustar = u0.clone();
        for j in 0..ny {
            for i in 0..=nx {
                let (x, y) = (i as f64 * dx, (j as f64 + 0.5) * dx);
                let (xb, yb) = self.trace(u0, v0, x, y, dt, 1.0);
                ustar[j * (nx + 1) + i] = if self.maccormack { self.sample_u(u0, xb, yb).0 } else { bicubic_limited(u0, nx + 1, ny, xb / dx, yb / dx - 0.5) };
            }
        }
        for j in 0..ny {
            for i in 0..=nx {
                let (x, y) = (i as f64 * dx, (j as f64 + 0.5) * dx);
                let (xb, yb) = self.trace(u0, v0, x, y, dt, 1.0);
                let (_, lo, hi) = self.sample_u(u0, xb, yb);
                let (xf, yf) = self.trace(u0, v0, x, y, dt, -1.0);
                let back = self.sample_u(&ustar, xf, yf).0;
                let c = ustar[j * (nx + 1) + i] + 0.5 * (u0[j * (nx + 1) + i] - back);
                un[j * (nx + 1) + i] = if self.maccormack { clampf(c, lo, hi) } else { ustar[j * (nx + 1) + i] };
            }
        }
        let mut vstar = v0.clone();
        for j in 0..=ny {
            for i in 0..nx {
                let (x, y) = ((i as f64 + 0.5) * dx, j as f64 * dx);
                let (xb, yb) = self.trace(u0, v0, x, y, dt, 1.0);
                vstar[j * nx + i] = if self.maccormack { self.sample_v(v0, xb, yb).0 } else { bicubic_limited(v0, nx, ny + 1, xb / dx - 0.5, yb / dx) };
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let (x, y) = ((i as f64 + 0.5) * dx, j as f64 * dx);
                let (xb, yb) = self.trace(u0, v0, x, y, dt, 1.0);
                let (_, lo, hi) = self.sample_v(v0, xb, yb);
                let (xf, yf) = self.trace(u0, v0, x, y, dt, -1.0);
                let back = self.sample_v(&vstar, xf, yf).0;
                let c = vstar[j * nx + i] + 0.5 * (v0[j * nx + i] - back);
                vn[j * nx + i] = if self.maccormack { clampf(c, lo, hi) } else { vstar[j * nx + i] };
            }
        }
        (un, vn)
    }

    /// α_k voisin (indices bornés ; voisin solide → miroir de la cellule centrale = gradient nul à la paroi)
    fn alpha_nb(&self, a: &[f64], i: isize, j: isize, ci: usize, cj: usize) -> f64 {
        let (ii, jj) = (i.clamp(0, self.nx as isize - 1) as usize, j.clamp(0, self.ny as isize - 1) as usize);
        let c = self.idx(ii, jj);
        if self.is_solid(c) {
            a[self.idx(ci, cj)]
        } else {
            a[c]
        }
    }
    /// Gradient de Youngs (en unités « cellule »)
    pub(crate) fn youngs_grad(&self, a: &[f64], i: usize, j: usize) -> (f64, f64) {
        let g = |di: isize, dj: isize| self.alpha_nb(a, i as isize + di, j as isize + dj, i, j);
        let gx = ((g(1, 1) + 2.0 * g(1, 0) + g(1, -1)) - (g(-1, 1) + 2.0 * g(-1, 0) + g(-1, -1))) / 8.0;
        let gy = ((g(1, 1) + 2.0 * g(0, 1) + g(-1, 1)) - (g(1, -1) + 2.0 * g(0, -1) + g(-1, -1))) / 8.0;
        (gx, gy)
    }

    /// Balayage PLIC dans la direction `dir` (0 = x, 1 = y) avec les vitesses de face courantes. `aw` : α de travail.
    fn plic_sweep(&mut self, dir: usize, dt: f64, aw: &mut Vec<Vec<f64>>) {
        let (nx, ny) = (self.nx, self.ny);
        let np = self.reg.phases.len();
        // groupes de phases miscibles : un groupe se comporte comme UNE phase pour l'interface (pas d'interface entre membres miscibles)
        let groups = self.reg.miscible_groups();
        let ng = groups.len();
        let gorder = self.group_order(&groups);
        // 1. PLIC des groupes mixtes (sur α normalisé)
        let n = nx * ny;
        let mut an = aw.clone();
        for c in 0..n {
            let s: f64 = (0..np).map(|k| an[k][c].clamp(0.0, 1.0)).sum::<f64>().max(1e-300);
            for k in 0..np {
                an[k][c] = an[k][c].clamp(0.0, 1.0) / s;
            }
        }
        let ag: Vec<Vec<f64>> = groups.iter().map(|g| (0..n).map(|c| g.iter().map(|&k| an[k][c]).sum::<f64>().min(1.0)).collect()).collect();
        let mut plic: Vec<Vec<Option<Plic>>> = vec![vec![None; n]; ng];
        for g in 0..ng {
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    let a = ag[g][c];
                    if a > 1e-10 && a < 1.0 - 1e-10 && !self.is_solid(c) {
                        let (gx, gy) = self.youngs_grad(&ag[g], i, j);
                        plic[g][c] = Some(Plic::new(a, gx, gy));
                    }
                }
            }
        }
        // 2. flux géométriques par face
        let (fi, fj) = if dir == 0 { (nx + 1, ny) } else { (nx, ny + 1) };
        let nf = fi * fj;
        let mut ff = vec![0.0f64; nf * np];
        let mut donor = vec![usize::MAX; nf];
        let mut recv = vec![usize::MAX; nf];
        let mut inflow_ghost = vec![false; nf];
        let mut ghost_t = vec![self.t_ambient; nf];
        // température reconstruite à la face (MUSCL, limiteur de van Leer) ; NaN = celle de la cellule donneuse
        let mut face_t = vec![f64::NAN; nf];
        let t_old = self.temp.clone();
        let van_leer = |a: f64, b: f64| if a * b > 0.0 { 2.0 * a * b / (a + b) } else { 0.0 };
        // cellule fluide sans interface (un seul groupe présent à 99 %) : la reconstruction d'ordre 2 y est sûre
        let single = |c: usize| -> Option<usize> { (0..ng).find(|&g| ag[g][c] > 0.99) };
        for fj_ in 0..fj {
            for fi_ in 0..fi {
                let f = fj_ * fi + fi_;
                let (uf, lo, hi) = if dir == 0 {
                    (self.u[self.uidx(fi_, fj_)], if fi_ > 0 { Some(self.idx(fi_ - 1, fj_)) } else { None }, if fi_ < nx { Some(self.idx(fi_, fj_)) } else { None })
                } else {
                    (self.v[self.vidx(fi_, fj_)], if fj_ > 0 { Some(self.idx(fi_, fj_ - 1)) } else { None }, if fj_ < ny { Some(self.idx(fi_, fj_)) } else { None })
                };
                if uf == 0.0 {
                    continue;
                }
                if lo.map_or(false, |c| self.is_solid(c)) || hi.map_or(false, |c| self.is_solid(c)) {
                    continue;
                }
                let xi = (uf.abs() * dt / self.dx).min(1.0);
                let (d, r) = if uf > 0.0 { (lo, hi) } else { (hi, lo) };
                donor[f] = d.unwrap_or(usize::MAX);
                recv[f] = r.unwrap_or(usize::MAX);
                match d {
                    None => {
                        inflow_ghost[f] = true; // fluide entrant : ambiant, ou celui de l'entrée de bord si le bord en a une
                        let (ga, gt) = self.ghost_source(dir, fi_, fj_);
                        for k in 0..np {
                            ff[f * np + k] = xi * ga[k];
                        }
                        ghost_t[f] = gt;
                    }
                    Some(dc) => {
                        // bande balayée dans la cellule donneuse (coordonnées unité)
                        let (s0, s1) = if uf > 0.0 { (1.0 - xi, 1.0) } else { (0.0, xi) };
                        let mut rem = xi;
                        let mut spare = vec![0.0; ng];
                        let mut fg = vec![0.0; ng];
                        let pure = (0..ng).find(|&g| ag[g][dc] >= 1.0 - 1e-10);
                        if let Some(gp) = pure {
                            fg[gp] = xi;
                        } else {
                            for (pos, &g) in gorder.iter().enumerate() {
                                let a = ag[g][dc];
                                let want = if pos == ng - 1 {
                                    rem
                                } else if let Some(pl) = plic[g][dc] {
                                    if dir == 0 { pl.area(s0, s1, 0.0, 1.0) } else { pl.area(0.0, 1.0, s0, s1) }
                                } else {
                                    0.0
                                };
                                let fk = want.min(rem).min(a).max(0.0);
                                fg[g] = fk;
                                rem -= fk;
                                spare[g] = a - fk;
                            }
                            if rem > 1e-14 {
                                let tot: f64 = spare.iter().map(|s| s.max(0.0)).sum();
                                if tot > 0.0 {
                                    for g in 0..ng {
                                        fg[g] += rem * spare[g].max(0.0) / tot;
                                    }
                                }
                            }
                        }
                        // cellule en amont de la donneuse (pour la pente) et receveuse, si fluides
                        let up = if dir == 0 {
                            if uf > 0.0 { (fi_ >= 2).then(|| self.idx(fi_ - 2, fj_)) } else { (fi_ + 1 < nx).then(|| self.idx(fi_ + 1, fj_)) }
                        } else if uf > 0.0 {
                            (fj_ >= 2).then(|| self.idx(fi_, fj_ - 2))
                        } else {
                            (fj_ + 1 < ny).then(|| self.idx(fi_, fj_ + 1))
                        };
                        let up = up.filter(|&c| !self.is_solid(c));
                        let rc = r.filter(|&c| !self.is_solid(c));
                        // reconstruction MUSCL (ordre 2, TVD) de la composition d'un groupe miscible et de la température, hors interfaces :
                        // sans elle la diffusion numérique du schéma décentré (≈ u·dx/2) dépasse la diffusion moléculaire des mélanges
                        let smooth = match (up, rc) {
                            (Some(uc), Some(rcv)) => {
                                let gd = single(dc);
                                if gd.is_some() && single(uc) == gd && single(rcv) == gd { Some((uc, rcv, gd.unwrap())) } else { None }
                            }
                            _ => None,
                        };
                        if let Some((uc, rcv, _)) = smooth {
                            let (tu, td, tr) = (t_old[uc], t_old[dc], t_old[rcv]);
                            face_t[f] = td + 0.5 * (1.0 - xi) * van_leer(td - tu, tr - td);
                        }
                        // flux du groupe → membres, au prorata de leur composition (reconstruite à la face si possible)
                        for g in 0..ng {
                            if fg[g] > 0.0 {
                                let tot = ag[g][dc].max(1e-300);
                                match smooth {
                                    Some((uc, rcv, gs)) if gs == g && groups[g].len() > 1 => {
                                        let (tu, tr) = (ag[g][uc].max(1e-300), ag[g][rcv].max(1e-300));
                                        let mut phi: Vec<f64> = groups[g]
                                            .iter()
                                            .map(|&k| {
                                                let (pu, pd, pr) = (an[k][uc] / tu, an[k][dc] / tot, an[k][rcv] / tr);
                                                (pd + 0.5 * (1.0 - xi) * van_leer(pd - pu, pr - pd)).max(0.0)
                                            })
                                            .collect();
                                        let sphi: f64 = phi.iter().sum::<f64>().max(1e-300);
                                        phi.iter_mut().for_each(|x| *x /= sphi);
                                        for (q, &k) in groups[g].iter().enumerate() {
                                            ff[f * np + k] = fg[g] * phi[q];
                                        }
                                    }
                                    _ => {
                                        for &k in &groups[g] {
                                            ff[f * np + k] = fg[g] * an[k][dc] / tot;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // 3. limiteur de sortie par cellule et par phase
        let mut outsum = vec![0.0f64; n * np];
        for f in 0..nf {
            if donor[f] != usize::MAX {
                for k in 0..np {
                    outsum[donor[f] * np + k] += ff[f * np + k];
                }
            }
        }
        for f in 0..nf {
            if donor[f] != usize::MAX {
                for k in 0..np {
                    let o = outsum[donor[f] * np + k];
                    let cap = aw[k][donor[f]].max(0.0) * 0.999;
                    if o > cap && o > 0.0 {
                        ff[f * np + k] *= cap / o;
                    }
                }
            }
        }
        // 4. mise à jour conservative : α, m_k, e, face par face DANS L'ORDRE des indices (la masse volumique m/α de la donneuse est
        //    lue sur l'état courant, flux entrants des faces précédentes compris). NB : une évaluation simultanée sur l'état d'avant le
        //    balayage (forme flux « pure ») a été essayée : elle transporte sans les amortir les écarts de remplissage m/α ≠ ρ des
        //    cellules d'eau et diverge sur un jet d'eau entrant dans l'air ; le GPU reproduit exactement cet ordre (une ligne, resp.
        //    une colonne, par fil)
        let a2 = self.dx * self.dx;
        for f in 0..nf {
            if donor[f] == usize::MAX && !inflow_ghost[f] {
                continue;
            }
            let (d, r) = (donor[f], recv[f]);
            let mut ef = 0.0;
            let mut mtot = 0.0;
            for k in 0..np {
                let fk = ff[f * np + k];
                if fk == 0.0 {
                    continue;
                }
                let (rho_d, t_d, p_d) = if d == usize::MAX {
                    let pr = if r != usize::MAX { self.p[r] } else { 101325.0 };
                    (self.reg.phases[k].rho(pr, ghost_t[f]), ghost_t[f], pr)
                } else {
                    // masse volumique de la phase dans la cellule donneuse (m/α), repli EOS si α ≈ 0
                    let rho_eos = self.reg.phases[k].rho(self.p[d], t_old[d]);
                    let rho = if aw[k][d] > 1e-8 { self.mass[k][d] / aw[k][d] } else { rho_eos };
                    let tf = if face_t[f].is_nan() { t_old[d] } else { face_t[f] };
                    (if rho.is_finite() && rho > 0.0 { rho } else { rho_eos }, tf, self.p[d])
                };
                let _ = p_d;
                // jamais plus que la masse présente dans la donneuse (traces : α ≈ 0 et masse volumique de repli EOS)
                let mf = if d != usize::MAX { (fk * rho_d).min(self.mass[k][d].max(0.0)) } else { fk * rho_d };
                mtot += mf;
                ef += mf * self.reg.phases[k].e_int(t_d);
                if d != usize::MAX {
                    aw[k][d] -= fk;
                    self.mass[k][d] -= mf;
                } else {
                    self.boundary_flux[k] += mf * a2;
                }
                if r != usize::MAX {
                    aw[k][r] += fk;
                    self.mass[k][r] += mf;
                } else {
                    self.boundary_flux[k] -= mf * a2;
                }
            }
            {
                let (fi_, fj_) = (f % fi, f / fi);
                let sgn = if dir == 0 { self.u[self.uidx(fi_, fj_)] } else { self.v[self.vidx(fi_, fj_)] }.signum();
                if dir == 0 { let q = self.uidx(fi_, fj_); self.mfx[q] += sgn * mtot; } else { let q = self.vidx(fi_, fj_); self.mfy[q] += sgn * mtot; }
            }
            if d != usize::MAX {
                self.e[d] -= ef;
            }
            if r != usize::MAX {
                self.e[r] += ef;
            }
        }
        for c in 0..n {
            if !self.is_solid(c) {
                self.temp[c] = self.temp_from_energy(c);
            }
        }
        let _ = Bc::Wall;
    }

    /// Ordre de priorité des GROUPES miscibles pour l'allocation des flux : groupes liquides d'abord, gazeux ensuite (le dernier prend le reste).
    /// Sans miscibilité chaque groupe est une phase : ordre identique à l'ancien `phase_order`.
    pub(crate) fn group_order(&self, groups: &[Vec<usize>]) -> Vec<usize> {
        let mut o: Vec<usize> = (0..groups.len()).collect();
        o.sort_by_key(|&g| if self.reg.phases[groups[g][0]].is_gas() { 1 } else { 0 });
        o
    }

    /// Advection conservative de (m_k, e) avec les vitesses de face courantes. Balayages alternés x/y.
    pub(crate) fn advect_mass_energy(&mut self, dt: f64) {
        let np = self.reg.phases.len();
        let n = self.nx * self.ny;
        let mut aw: Vec<Vec<f64>> = (0..np).map(|k| self.alpha[k].clone()).collect();
        let _ = n;
        self.mfx.iter_mut().for_each(|v| *v = 0.0);
        self.mfy.iter_mut().for_each(|v| *v = 0.0);
        self.m_tot_old = (0..n).map(|c| (0..np).map(|k| self.mass[k][c]).sum()).collect();
        let first = self.step_count % 2;
        for s in 0..2 {
            self.plic_sweep((first + s) % 2, dt, &mut aw);
        }
    }
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 { 0.0 } else if a.abs() < b.abs() { a } else { b }
}

impl MultiphaseSim {
    /// Advection CONSERVATIVE de la quantité de mouvement ρ_f·u sur les volumes de contrôle des faces, avec les flux de masse
    /// du VOF (mfx, mfy) : ρ_f,nouveau = ρ_f − Σ flux de masse exactement, donc la vitesse d'une face d'air n'est jamais
    /// « contaminée » par celle de l'eau voisine (Rudman 1998). Valeur transportée : reconstruction minmod d'ordre 2.
    /// À appeler APRÈS advect_mass_energy (masses déjà mises à jour). Renvoie (u, v) advectés, sans forces.
    pub(crate) fn advect_momentum(&self) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny) = (self.nx, self.ny);
        let np = self.reg.phases.len();
        let mnew = |c: usize| -> f64 { (0..np).map(|k| self.mass[k][c]).sum::<f64>() };
        let mold = &self.m_tot_old;
        let (u0, v0) = (&self.u, &self.v);
        let mut un = u0.clone();
        let mut vn = v0.clone();
        let ux = |i: isize, j: isize| u0[(j.clamp(0, ny as isize - 1) as usize) * (nx + 1) + i.clamp(0, nx as isize) as usize];
        let vy = |i: isize, j: isize| v0[(j.clamp(0, ny as isize) as usize) * nx + i.clamp(0, nx as isize - 1) as usize];
        // composante u : faces intérieures 1..nx-1
        for j in 0..ny {
            for i in 1..nx {
                let (a, b) = (j * nx + i - 1, j * nx + i);
                if self.is_solid(a) || self.is_solid(b) { continue; }
                let (ii, jj) = (i as isize, j as isize);
                let rho_old = 0.5 * (mold[a] + mold[b]);
                let rho_new = 0.5 * (mnew(a) + mnew(b));
                // frontières en x : centres des cellules i-1 (gauche) et i (droite)
                let fr = 0.5 * (self.mfx[j * (nx + 1) + i] + self.mfx[j * (nx + 1) + i + 1]);
                let fl = 0.5 * (self.mfx[j * (nx + 1) + i - 1] + self.mfx[j * (nx + 1) + i]);
                let edge = |lo: isize| -> f64 { // valeur transportée entre u(lo) et u(lo+1) ; flux défini par l'appelant
                    let _ = lo; 0.0
                };
                let _ = edge;
                let carry_x = |f: f64, lo: isize| -> f64 {
                    if f >= 0.0 { ux(lo, jj) + 0.5 * minmod(ux(lo, jj) - ux(lo - 1, jj), ux(lo + 1, jj) - ux(lo, jj)) }
                    else { ux(lo + 1, jj) - 0.5 * minmod(ux(lo + 1, jj) - ux(lo, jj), ux(lo + 2, jj) - ux(lo + 1, jj)) }
                };
                // frontières en y : coins (i, j) bas et (i, j+1) haut
                let fb = 0.5 * (self.mfy[j * nx + i - 1] + self.mfy[j * nx + i]);
                let ft = 0.5 * (self.mfy[(j + 1) * nx + i - 1] + self.mfy[(j + 1) * nx + i]);
                let carry_y = |f: f64, lo: isize| -> f64 { // entre u(i,lo) et u(i,lo+1)
                    if f >= 0.0 { ux(ii, lo) + 0.5 * minmod(ux(ii, lo) - ux(ii, lo - 1), ux(ii, lo + 1) - ux(ii, lo)) }
                    else { ux(ii, lo + 1) - 0.5 * minmod(ux(ii, lo + 1) - ux(ii, lo), ux(ii, lo + 2) - ux(ii, lo + 1)) }
                };
                let (cr, cl) = (carry_x(fr, ii), carry_x(fl, ii - 1));
                let (ct, cb) = (if j + 1 < ny { carry_y(ft, jj) } else { ux(ii, jj) }, if j > 0 { carry_y(fb, jj - 1) } else { ux(ii, jj) });
                let (ft_, fb_) = (if j + 1 < ny { ft } else { 0.0 }, if j > 0 { fb } else { 0.0 });
                let q = rho_old * u0[j * (nx + 1) + i] - (fr * cr - fl * cl) - (ft_ * ct - fb_ * cb);
                let (lo, hi) = (ux(ii, jj).min(ux(ii - 1, jj)).min(ux(ii + 1, jj)).min(ux(ii, jj - 1)).min(ux(ii, jj + 1)),
                                ux(ii, jj).max(ux(ii - 1, jj)).max(ux(ii + 1, jj)).max(ux(ii, jj - 1)).max(ux(ii, jj + 1)));
                if rho_new > 1e-8 { un[j * (nx + 1) + i] = clampf(q / rho_new, lo, hi); }
            }
        }
        // composante v : faces intérieures 1..ny-1
        for j in 1..ny {
            for i in 0..nx {
                let (a, b) = ((j - 1) * nx + i, j * nx + i);
                if self.is_solid(a) || self.is_solid(b) { continue; }
                let (ii, jj) = (i as isize, j as isize);
                let rho_old = 0.5 * (mold[a] + mold[b]);
                let rho_new = 0.5 * (mnew(a) + mnew(b));
                let ft = 0.5 * (self.mfy[j * nx + i] + self.mfy[(j + 1) * nx + i]);
                let fb = 0.5 * (self.mfy[(j - 1) * nx + i] + self.mfy[j * nx + i]);
                let carry_y = |f: f64, lo: isize| -> f64 {
                    if f >= 0.0 { vy(ii, lo) + 0.5 * minmod(vy(ii, lo) - vy(ii, lo - 1), vy(ii, lo + 1) - vy(ii, lo)) }
                    else { vy(ii, lo + 1) - 0.5 * minmod(vy(ii, lo + 1) - vy(ii, lo), vy(ii, lo + 2) - vy(ii, lo + 1)) }
                };
                let fr = 0.5 * (self.mfx[j * (nx + 1) + i + 1] + self.mfx[(j - 1) * (nx + 1) + i + 1]);
                let fl = 0.5 * (self.mfx[j * (nx + 1) + i] + self.mfx[(j - 1) * (nx + 1) + i]);
                let carry_x = |f: f64, lo: isize| -> f64 { // entre v(lo,j) et v(lo+1,j)
                    if f >= 0.0 { vy(lo, jj) + 0.5 * minmod(vy(lo, jj) - vy(lo - 1, jj), vy(lo + 1, jj) - vy(lo, jj)) }
                    else { vy(lo + 1, jj) - 0.5 * minmod(vy(lo + 1, jj) - vy(lo, jj), vy(lo + 2, jj) - vy(lo + 1, jj)) }
                };
                let (ct, cb) = (carry_y(ft, jj), carry_y(fb, jj - 1));
                let (cr, cl) = (if i + 1 < nx { carry_x(fr, ii) } else { vy(ii, jj) }, if i > 0 { carry_x(fl, ii - 1) } else { vy(ii, jj) });
                let (fr_, fl_) = (if i + 1 < nx { fr } else { 0.0 }, if i > 0 { fl } else { 0.0 });
                let q = rho_old * v0[j * nx + i] - (ft * ct - fb * cb) - (fr_ * cr - fl_ * cl);
                let (lo, hi) = (vy(ii, jj).min(vy(ii, jj - 1)).min(vy(ii, jj + 1)).min(vy(ii - 1, jj)).min(vy(ii + 1, jj)),
                                vy(ii, jj).max(vy(ii, jj - 1)).max(vy(ii, jj + 1)).max(vy(ii - 1, jj)).max(vy(ii + 1, jj)));
                if rho_new > 1e-8 { vn[j * nx + i] = clampf(q / rho_new, lo, hi); }
            }
        }
        (un, vn)
    }
}
