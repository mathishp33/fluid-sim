//! Forces sur les faces : gravité, viscosité (tenseur symétrique complet, μ variable), tension de surface (CSF équilibrée,
//! courbure par fonctions de hauteur). Toutes sont des ACCÉLÉRATIONS de face (m/s²) ; la pression utilise le même ρ_f
//! que ces forces, donc l'hydrostatique est équilibrée exactement au niveau discret.

use super::multiphase::{Bc, MultiphaseSim};

impl MultiphaseSim {
    /// Fraction volumique totale de liquide ℓ (marqueur de l'interface liquide/gaz)
    fn liquid_marker(&self) -> Vec<f64> {
        let n = self.nx * self.ny;
        let liq = self.liquid_phases();
        (0..n).map(|c| if self.is_solid(c) { 0.0 } else { liq.iter().map(|&k| self.alpha[k][c]).sum::<f64>().clamp(0.0, 1.0) }).collect()
    }

    fn marker_at(&self, l: &[f64], i: isize, j: isize, ci: usize, cj: usize) -> f64 {
        let (ii, jj) = (i.clamp(0, self.nx as isize - 1) as usize, j.clamp(0, self.ny as isize - 1) as usize);
        let c = self.idx(ii, jj);
        if self.is_solid(c) { l[self.idx(ci, cj)] } else { l[c] }
    }

    /// Courbure κ = ∇·n_sortant (unités 1/m) par fonction de hauteur 3×7 ; repli : normale lissée. NaN si pas d'interface.
    pub(crate) fn curvature_field(&self, l: &[f64]) -> Vec<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let mut kap = vec![f64::NAN; nx * ny];
        let m = |i: isize, j: isize, ci: usize, cj: usize| self.marker_at(l, i, j, ci, cj);
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                let a = l[c];
                if !(a > 1e-6 && a < 1.0 - 1e-6) {
                    continue;
                }
                let (gx, gy) = self.youngs_grad(l, i, j);
                let vertical = gy.abs() >= gx.abs();
                // somme des 7 cellules le long de l'orientation dominante, pour les 3 colonnes (lignes) voisines
                let mut hs = [0.0f64; 3];
                let mut valid = true;
                for (q, off) in (-1isize..=1).enumerate() {
                    let mut s = 0.0;
                    let (mut first, mut last) = (0.0, 0.0);
                    for t in -3isize..=3 {
                        let (ii, jj) = if vertical { (i as isize + off, j as isize + t) } else { (i as isize + t, j as isize + off) };
                        let v = m(ii, jj, i, j);
                        if t == -3 { first = v }
                        if t == 3 { last = v }
                        s += v;
                    }
                    // la colonne doit contenir toute l'interface : extrémités pures et opposées
                    if (first - last).abs() < 0.5 {
                        valid = false;
                    }
                    hs[q] = s;
                }
                if valid {
                    let (hm, h0, hp) = (hs[0], hs[1], hs[2]);
                    let h1 = 0.5 * (hp - hm);
                    let h2 = hp - 2.0 * h0 + hm;
                    kap[c] = -h2 / (1.0 + h1 * h1).powf(1.5) / dx;
                }
            }
        }
        // repli : divergence de la normale (α lissé 1-2-1)
        let mut sm = l.to_vec();
        for j in 0..ny {
            for i in 0..nx {
                let mut s = 0.0;
                for dj in -1isize..=1 {
                    for di in -1isize..=1 {
                        let w = (2 - di.abs()) as f64 * (2 - dj.abs()) as f64;
                        s += w * m(i as isize + di, j as isize + dj, i, j);
                    }
                }
                sm[self.idx(i, j)] = s / 16.0;
            }
        }
        let nrm = |i: isize, j: isize, ci: usize, cj: usize| -> (f64, f64) {
            let g = |di: isize, dj: isize| self.marker_at(&sm, i + di, j + dj, ci, cj);
            let gx = (g(1, 0) - g(-1, 0)) * 0.5;
            let gy = (g(0, 1) - g(0, -1)) * 0.5;
            let s = (gx * gx + gy * gy).sqrt().max(1e-12);
            (-gx / s, -gy / s)
        };
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                let a = l[c];
                if a > 1e-6 && a < 1.0 - 1e-6 && kap[c].is_nan() {
                    let (ii, jj) = (i as isize, j as isize);
                    let div = (nrm(ii + 1, jj, i, j).0 - nrm(ii - 1, jj, i, j).0 + nrm(ii, jj + 1, i, j).1 - nrm(ii, jj - 1, i, j).1) * 0.5 / dx;
                    kap[c] = div;
                }
            }
        }
        kap
    }

    /// Vitesse tangentielle de la paroi adhérente du côté `side` (0 gauche, 1 droite, 2 bas, 3 haut) : un mur (`Bc::Wall`) sans
    /// entrée, déclaré adhérent (`no_slip`) ; None : glissement libre (ou bord ouvert, entrée)
    pub(crate) fn wall_slide(&self, side: usize) -> Option<f64> {
        match (self.bc[side], &self.inlet[side], self.no_slip[side]) {
            (Bc::Wall, None, Some(v)) => Some(v),
            _ => None,
        }
    }

    /// Vitesse de la paroi adhérente au nœud (i, j) d'un bord du domaine (le long du bord, sans les coins) ; None ailleurs
    pub(crate) fn wall_node_vel(&self, i: usize, j: usize) -> Option<f64> {
        let (nx, ny) = (self.nx, self.ny);
        if (j == 0 || j == ny) && i >= 1 && i < nx {
            self.wall_slide(if j == 0 { 2 } else { 3 })
        } else if (i == 0 || i == nx) && j >= 1 && j < ny {
            self.wall_slide(if i == 0 { 0 } else { 1 })
        } else {
            None
        }
    }

    /// Contrainte de cisaillement τxy au nœud (i, j) d'un bord du domaine contre une paroi adhérente : non-glissement à la vitesse U
    /// de la paroi, la face voisine étant à une demi-maille — τxy = 2 μ (u − U) / dx en bas (u de la première rangée), 2 μ (U − u) / dx
    /// en haut, 2 μ (v − U) / dx à gauche, 2 μ (U − v) / dx à droite (la vitesse normale d'une paroi est nulle tout du long : ∂v/∂x
    /// = 0 en bas). μ moyenne des deux mailles du nœud, plafonnée comme à un nœud intérieur si `cap_dt`. None : pas une paroi
    /// adhérente, ou une des deux mailles est solide (glissement libre, comme avant).
    pub(crate) fn wall_node_txy(&self, i: usize, j: usize, mu: &[f64], u: &[f64], v: &[f64], cap_dt: Option<f64>) -> Option<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let uw = self.wall_node_vel(i, j)?;
        let (a, b, d) = if j == 0 || j == ny {
            let jc = if j == 0 { 0 } else { ny - 1 };
            let du = u[jc * (nx + 1) + i] - uw;
            (self.idx(i - 1, jc), self.idx(i, jc), if j == 0 { du } else { -du })
        } else {
            let ic = if i == 0 { 0 } else { nx - 1 };
            let dv = v[j * nx + ic] - uw;
            (self.idx(ic, j - 1), self.idx(ic, j), if i == 0 { dv } else { -dv })
        };
        if self.is_solid(a) || self.is_solid(b) {
            return None;
        }
        let m = 0.5 * (mu[a] + mu[b]);
        let m = match cap_dt {
            Some(dt) => m.min(self.explicit_node_mu_cap([a, b, a, b], dt)),
            None => m,
        };
        Some(2.0 * m * d / dx)
    }

    /// Ajoute dt × (gravité + viscosité + tension de surface) aux champs de face us, vs. Les champs ρ, μ, α sont ceux du début de pas.
    pub(crate) fn add_forces(&self, dt: f64, us: &mut [f64], vs: &mut [f64]) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        // gravité
        for f in us.iter_mut() {
            *f += dt * self.gravity[0];
        }
        for f in vs.iter_mut() {
            *f += dt * self.gravity[1];
        }
        // accélérateurs et couches éponges (cellules spéciales)
        self.add_accelerators(dt, us, vs);
        self.add_sponges(dt, us, vs);
        // tension de surface (CSF équilibré) : a = σ κ ∇ℓ / ρ_f
        if self.surface_tension {
            let l = self.liquid_marker();
            let kap = self.curvature_field(&l);
            let liq = self.liquid_phases();
            let sigma_c: Vec<f64> = (0..nx * ny)
                .map(|c| {
                    let ell: f64 = liq.iter().map(|&k| self.alpha[k][c]).sum();
                    if ell < 1e-12 { 0.0 } else { liq.iter().map(|&k| self.alpha[k][c] * self.reg.phases[k].sigma).sum::<f64>() / ell }
                })
                .collect();
            let face = |a: usize, b: usize| -> f64 {
                let (wa, wb) = (l[a] * (1.0 - l[a]), l[b] * (1.0 - l[b]));
                let (ka, kb) = (if kap[a].is_nan() { 0.0 } else { kap[a] }, if kap[b].is_nan() { 0.0 } else { kap[b] });
                let w = wa + wb;
                if w < 1e-12 { return 0.0; }
                let kf = (ka * wa + kb * wb) / w;
                let sg = (sigma_c[a] * wa + sigma_c[b] * wb) / w;
                sg * kf * (l[b] - l[a]) / dx
            };
            for j in 0..ny {
                for i in 1..nx {
                    let (a, b) = (self.idx(i - 1, j), self.idx(i, j));
                    if !self.is_solid(a) && !self.is_solid(b) {
                        us[j * (nx + 1) + i] += dt * face(a, b) / self.face_rho_x(i, j);
                    }
                }
            }
            for j in 1..ny {
                for i in 0..nx {
                    let (a, b) = (self.idx(i, j - 1), self.idx(i, j));
                    if !self.is_solid(a) && !self.is_solid(b) {
                        vs[j * nx + i] += dt * face(a, b) / self.face_rho_y(i, j);
                    }
                }
            }
        }
        // viscosité : ∇·[μ(∇u + ∇uᵀ)] / ρ_f  (parois : glissement libre, sauf les parois adhérentes des bords) ; au-delà de la borne
        // explicite 0.2 ρ dx²/dt, le reste de μ est implicite (`apply_implicit_viscosity`)
        if self.viscous {
            let mue: Vec<f64> = (0..nx * ny).map(|c| self.mu_cell[c].min(self.explicit_mu_cap(c, dt))).collect();
            // corps en mouvement (mp_bodies.rs) : contraintes calculées sur la vitesse du MÉLANGE fluide + corps de chaque face,
            // u + (1 − A) V_corps — l'intérieur d'un corps qui bouge n'est pas un fluide au repos qui freinerait la couche entraînée
            let mixture = if self.has_moving_bodies() { Some(self.mixture_velocity()) } else { None };
            let (u, v) = match &mixture {
                Some((a, b)) => (a, b),
                None => (&self.u, &self.v),
            };
            let ui = |i: usize, j: usize| u[j * (nx + 1) + i];
            let vi = |i: usize, j: usize| v[j * nx + i];
            // τxx, τyy aux centres
            let mut txx = vec![0.0; nx * ny];
            let mut tyy = vec![0.0; nx * ny];
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    txx[c] = 2.0 * mue[c] * (ui(i + 1, j) - ui(i, j)) / dx;
                    tyy[c] = 2.0 * mue[c] * (vi(i, j + 1) - vi(i, j)) / dx;
                }
            }
            // τxy aux nœuds (i, j) : 0 ≤ i ≤ nx, 0 ≤ j ≤ ny ; nul sur les bords et contre les solides (glissement libre)
            let mut txy = vec![0.0; (nx + 1) * (ny + 1)];
            for j in 1..ny {
                for i in 1..nx {
                    let cs = [self.idx(i - 1, j - 1), self.idx(i, j - 1), self.idx(i - 1, j), self.idx(i, j)];
                    if cs.iter().any(|&c| self.is_solid(c)) {
                        continue;
                    }
                    let mu = (cs.iter().map(|&c| mue[c]).sum::<f64>() / 4.0).min(self.explicit_node_mu_cap(cs, dt));
                    txy[j * (nx + 1) + i] = mu * ((ui(i, j) - ui(i, j - 1)) / dx + (vi(i, j) - vi(i - 1, j)) / dx);
                }
            }
            // parois adhérentes (bords du domaine) : frottement de la paroi sur la rangée de faces qui la longe
            if self.no_slip.iter().any(|s| s.is_some()) {
                let nodes = (0..=nx).flat_map(|i| [(i, 0), (i, ny)]).chain((1..ny).flat_map(|j| [(0, j), (nx, j)]));
                for (i, j) in nodes {
                    if let Some(t) = self.wall_node_txy(i, j, &mue, u, v, Some(dt)) {
                        txy[j * (nx + 1) + i] = t;
                    }
                }
            }
            for j in 0..ny {
                for i in 1..nx {
                    let (a, b) = (self.idx(i - 1, j), self.idx(i, j));
                    if self.is_solid(a) || self.is_solid(b) { continue; }
                    let fx = (txx[b] - txx[a]) / dx + (txy[(j + 1) * (nx + 1) + i] - txy[j * (nx + 1) + i]) / dx;
                    us[j * (nx + 1) + i] += dt * fx / self.face_rho_x(i, j);
                }
            }
            for j in 1..ny {
                for i in 0..nx {
                    let (a, b) = (self.idx(i, j - 1), self.idx(i, j));
                    if self.is_solid(a) || self.is_solid(b) { continue; }
                    let fy = (tyy[b] - tyy[a]) / dx + (txy[j * (nx + 1) + i + 1] - txy[j * (nx + 1) + i]) / dx;
                    vs[j * nx + i] += dt * fy / self.face_rho_y(i, j);
                }
            }
        }
    }
}
