//! Advection : (1) masses partielles + énergie + fractions par balayages PLIC conservatifs (flux géométriques exacts),
//! (2) vitesse par semi-lagrangien MacCormack limité.

use super::multiphase::{Bc, MultiphaseSim};
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
    if v.is_finite() { v.clamp(lo, hi) } else { lin }
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
                un[j * (nx + 1) + i] = if self.maccormack { c.clamp(lo, hi) } else { ustar[j * (nx + 1) + i] };
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
                vn[j * nx + i] = if self.maccormack { c.clamp(lo, hi) } else { vstar[j * nx + i] };
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
        let order = self.phase_order();
        // 1. PLIC des phases mixtes (sur α normalisé)
        let n = nx * ny;
        let mut an = aw.clone();
        for c in 0..n {
            let s: f64 = (0..np).map(|k| an[k][c].clamp(0.0, 1.0)).sum::<f64>().max(1e-300);
            for k in 0..np {
                an[k][c] = an[k][c].clamp(0.0, 1.0) / s;
            }
        }
        let mut plic: Vec<Vec<Option<Plic>>> = vec![vec![None; n]; np];
        for k in 0..np {
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    let a = an[k][c];
                    if a > 1e-10 && a < 1.0 - 1e-10 && !self.is_solid(c) {
                        let (gx, gy) = self.youngs_grad(&an[k], i, j);
                        plic[k][c] = Some(Plic::new(a, gx, gy));
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
                        inflow_ghost[f] = true; // air ambiant entrant
                        ff[f * np + self.ambient_phase] = xi;
                    }
                    Some(dc) => {
                        // bande balayée dans la cellule donneuse (coordonnées unité)
                        let (s0, s1) = if uf > 0.0 { (1.0 - xi, 1.0) } else { (0.0, xi) };
                        let mut rem = xi;
                        let mut spare = vec![0.0; np];
                        let pure = (0..np).find(|&k| an[k][dc] >= 1.0 - 1e-10);
                        if let Some(kp) = pure {
                            ff[f * np + kp] = xi;
                        } else {
                            for (pos, &k) in order.iter().enumerate() {
                                let a = an[k][dc];
                                let want = if pos == np - 1 {
                                    rem
                                } else if let Some(pl) = plic[k][dc] {
                                    if dir == 0 { pl.area(s0, s1, 0.0, 1.0) } else { pl.area(0.0, 1.0, s0, s1) }
                                } else {
                                    0.0
                                };
                                let fk = want.min(rem).min(a).max(0.0);
                                ff[f * np + k] = fk;
                                rem -= fk;
                                spare[k] = a - fk;
                            }
                            if rem > 1e-14 {
                                let tot: f64 = spare.iter().map(|s| s.max(0.0)).sum();
                                if tot > 0.0 {
                                    for k in 0..np {
                                        ff[f * np + k] += rem * spare[k].max(0.0) / tot;
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
        // 4. mise à jour conservative : α, m_k, e
        let t_old = self.temp.clone();
        let a2 = self.dx * self.dx;
        for f in 0..nf {
            if donor[f] == usize::MAX && !inflow_ghost[f] {
                continue;
            }
            let (d, r) = (donor[f], recv[f]);
            let mut ef = 0.0;
            for k in 0..np {
                let fk = ff[f * np + k];
                if fk == 0.0 {
                    continue;
                }
                let (rho_d, t_d, p_d) = if d == usize::MAX {
                    let pr = if r != usize::MAX { self.p[r] } else { 101325.0 };
                    (self.reg.phases[k].rho(pr, self.t_ambient), self.t_ambient, pr)
                } else {
                    // masse volumique de la phase dans la cellule donneuse (m/α), repli EOS si α ≈ 0
                    let rho_eos = self.reg.phases[k].rho(self.p[d], t_old[d]);
                    let rho = if aw[k][d] > 1e-8 { self.mass[k][d] / aw[k][d] } else { rho_eos };
                    (if rho.is_finite() && rho > 0.0 { rho } else { rho_eos }, t_old[d], self.p[d])
                };
                let _ = p_d;
                let mf = fk * rho_d;
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

    /// Ordre de priorité des phases pour l'allocation des flux : liquides d'abord, gaz ensuite (le dernier prend le reste).
    pub(crate) fn phase_order(&self) -> Vec<usize> {
        let mut o: Vec<usize> = (0..self.reg.phases.len()).collect();
        o.sort_by_key(|&k| if self.reg.phases[k].is_gas() { 1 } else { 0 });
        o
    }

    /// Advection conservative de (m_k, e) avec les vitesses de face courantes. Balayages alternés x/y.
    pub(crate) fn advect_mass_energy(&mut self, dt: f64) {
        let np = self.reg.phases.len();
        let n = self.nx * self.ny;
        let mut aw: Vec<Vec<f64>> = (0..np).map(|k| self.alpha[k].clone()).collect();
        let _ = n;
        let first = self.step_count % 2;
        for s in 0..2 {
            self.plic_sweep((first + s) % 2, dt, &mut aw);
        }
    }
}
