use rayon::prelude::*;
use crate::simulation::fluid::{Boundary, BoundaryType, CellEffect, CellType, FluidSim, GridBoundary};

const CFL: f32 = 0.9;
const MAX_SUBSTEPS: usize = 64;
const MAX_VISC_SUBSTEPS: usize = 16;

#[derive(Clone, Copy)]
struct SharedMut(*mut f32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}
impl SharedMut {
    #[inline]
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

// Flux du TRACEUR (advection + diffusion) aux faces de la grille MAC. Unités de grille.
struct TracerKernel<'a> {
    w: usize,
    h: usize,
    u: &'a [f32],
    v: &'a [f32],
    open_u: &'a [u8],
    open_v: &'a [u8],
    d: f32, // diffusion du tracer en cellules²/s
}

impl TracerKernel<'_> {
    #[inline]
    fn fx(&self, tr: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let uf = self.u[i];
        let adv = uf * if uf > 0.0 { tr[i] } else { tr[i + 1] };
        let dif = if x >= 1 && x + 2 < self.w && self.open_u[i] != 0 {
            -self.d * (tr[i + 1] - tr[i])
        } else {
            0.0
        };
        adv + dif
    }

    #[inline]
    fn fy(&self, tr: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let vf = self.v[i];
        let adv = vf * if vf > 0.0 { tr[i] } else { tr[i + self.w] };
        let dif = if y >= 1 && y + 2 < self.h && self.open_v[i] != 0 {
            -self.d * (tr[i + self.w] - tr[i])
        } else {
            0.0
        };
        adv + dif
    }
}

impl FluidSim {
    // ------------------------------------------------------------------ viscosité

    pub(crate) fn viscosity_substeps(&self, dt: f32) -> usize {
        ((4.0 * self.nu_grid() * dt / CFL).ceil() as usize).clamp(1, MAX_VISC_SUBSTEPS)
    }

    // Laplacien de la vitesse sur les faces. Contribution d'un voisin (face de même orientation) :
    //  - face ouverte : v_voisin − c
    //  - fermée, glissement libre (no_slip = false) : 0 (gradient nul)
    //  - fermée, no-slip : direction normale ou voisin = cellule fantôme du bord -> valeur stockée − c
    //    (0 pour une face fermée par un solide, valeur de CL au bord) ; direction tangentielle contre
    //    un solide -> −2c (paroi à mi-cellule, adhérence)
    pub fn diffuse_velocity(&mut self, dt: f32) {
        if self.nu_grid() <= 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let n = self.viscosity_substeps(dt);
        let k = self.nu_grid() * dt / n as f32;
        let ns = self.no_slip;
        let term = move |open: bool, val: f32, c: f32, normal: bool, ghost: bool| -> f32 {
            if open { val - c } else if !ns { 0.0 } else if normal || ghost { val - c } else { -2.0 * c }
        };

        for _ in 0..n {
            self.velocity_x_temp.copy_from_slice(&self.velocity_x);
            self.velocity_y_temp.copy_from_slice(&self.velocity_y);

            {
                let (u, ou) = (&self.velocity_x, &self.open_u);
                self.velocity_x_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    if y == 0 || y >= h - 1 {
                        return;
                    }
                    for x in 1..w - 2 {
                        let i = x + y * w;
                        if ou[i] == 0 {
                            continue;
                        }
                        let c = u[i];
                        let lap = term(ou[i + 1] != 0, u[i + 1], c, true, false)
                            + term(ou[i - 1] != 0, u[i - 1], c, true, false)
                            + term(ou[i + w] != 0, u[i + w], c, false, y + 1 == h - 1)
                            + term(ou[i - w] != 0, u[i - w], c, false, y == 1);
                        row[x] = c + k * lap;
                    }
                });
            }
            {
                let (v, ov) = (&self.velocity_y, &self.open_v);
                self.velocity_y_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    if y == 0 || y >= h - 2 {
                        return;
                    }
                    for x in 1..w - 1 {
                        let i = x + y * w;
                        if ov[i] == 0 {
                            continue;
                        }
                        let c = v[i];
                        let lap = term(ov[i + 1] != 0, v[i + 1], c, false, x + 1 == w - 1)
                            + term(ov[i - 1] != 0, v[i - 1], c, false, x == 1)
                            + term(ov[i + w] != 0, v[i + w], c, true, false)
                            + term(ov[i - w] != 0, v[i - w], c, true, false);
                        row[x] = c + k * lap;
                    }
                });
            }

            std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
            std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
        }
    }

    // ------------------------------------------------------------------ flottabilité

    // y pointe vers le bas : dv/dt = -(accélération vers le haut).
    //  - low-Mach (gaz)         : a_haut = g·(ρ_ref/ρ − 1), avec ρ(T, P_th)
    //  - Boussinesq (autres)    : a_haut = g·(ρ_ref − ρ)/ρ_ref via buoyancy_factor(T)
    fn apply_buoyancy(&mut self, dt: f32) {
        let g = self.gravity_grid();
        if g == 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let low = self.low_mach_active();
        let a = g * dt;
        let (t, rho, ov) = (&self.temperature, &self.rho, &self.open_v);
        let factor = |i: usize| if low { m.buoyancy_from_rho(rho[i]) } else { m.buoyancy_factor(t[i]) };
        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 2 { return; }
            for x in 1..w - 1 {
                let i = x + y * w;
                if ov[i] != 0 {
                    row[x] -= a * 0.5 * (factor(i) + factor(i + w));
                }
            }
        });
    }

    // ------------------------------------------------------------------ température

    // fantômes : Inlet = valeur, Wall = adiabatique,
    // Outlet = ambiante (t_ref) avant l'advection, copie de l'intérieur après
    fn update_ghost_temperature(&mut self, before_advection: bool) {
        let (w, h) = (self.width, self.height);
        let t_ref = self.material.t_ref;
        let ghost = |b: &Boundary, inner: f32| match b.boundary_type {
            BoundaryType::Inlet => b.temperature,
            BoundaryType::Outlet if before_advection => t_ref,
            _ => inner,
        };
        for y in 1..h - 1 {
            let (l, r) = (y * w, (w - 1) + y * w);
            self.temperature[l] = ghost(&self.left_boundary, self.temperature[l + 1]);
            self.temperature[r] = ghost(&self.right_boundary, self.temperature[r - 1]);
        }
        for x in 1..w - 1 {
            let (t, b) = (x, x + (h - 1) * w);
            self.temperature[t] = ghost(&self.top_boundary, self.temperature[t + w]);
            self.temperature[b] = ghost(&self.bottom_boundary, self.temperature[b - w]);
        }
    }

    // Équation d'énergie (forme non conservative, valable avec div u ≠ 0) :
    //   dT/dt = −u·∇T + (∇·(k∇T) + q + dP_th/dt) / (ρ·cp)
    // - advection : upwind sur les seules faces entrantes (= flux conservatif − T·div u)
    // - conduction : Σ k_face·(T_j − T_i)/dx², k_face = moyenne harmonique ; chaque cellule
    //   divise par SA capacité thermique (fluide ρ·cp, solide ρ_s·cp_s) -> l'énergie est conservée
    //   aux interfaces fluide/solide ; pas de conduction à travers les faces du bord (adiabatique)
    // - q : source de chaleur (W/m³), dP_th/dt : travail de pression (fluide seulement)
    fn temperature_substep(&mut self, dts: f32) {
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let low = self.low_mach_active();
        let dpdt = self.dpdt;
        let (kf, ks, cp) = (m.conductivity, m.solid_conductivity, m.cp);
        let (cap_ref, cap_solid) = (m.heat_capacity_ref(), m.heat_capacity_solid());

        self.temperature_temp.copy_from_slice(&self.temperature); // conserve les fantômes

        let (t, u, v, s, rho, q) = (&self.temperature, &self.velocity_x, &self.velocity_y,
                                    &self.solids, &self.rho, &self.heat_source);
        let k_of = |i: usize| if s[i] == CellType::Wall { ks } else { kf };
        let k_face = |a: usize, b: usize| {
            let (ka, kb) = (k_of(a), k_of(b));
            let sum = ka + kb;
            if sum <= 0.0 { 0.0 } else { 2.0 * ka * kb / sum }
        };

        self.temperature_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                let ti = t[i];

                // advection (faces entrantes)
                let mut adv = 0.0;
                let (fr, fl, fd, fu) = (u[i], -u[i - 1], v[i], -v[i - w]); // flux sortants
                if fr < 0.0 { adv += fr * (t[i + 1] - ti); }
                if fl < 0.0 { adv += fl * (t[i - 1] - ti); }
                if fd < 0.0 { adv += fd * (t[i + w] - ti); }
                if fu < 0.0 { adv += fu * (t[i - w] - ti); }

                // conduction (faces intérieures seulement)
                let mut cond = 0.0;
                if x + 2 < w { cond += k_face(i, i + 1) * (t[i + 1] - ti); }
                if x >= 2 { cond += k_face(i, i - 1) * (t[i - 1] - ti); }
                if y + 2 < h { cond += k_face(i, i + w) * (t[i + w] - ti); }
                if y >= 2 { cond += k_face(i, i - w) * (t[i - w] - ti); }

                let wall = s[i] == CellType::Wall;
                let cap = if wall { cap_solid } else if low { rho[i] * cp } else { cap_ref }.max(1.0);
                let mut dtdt = -adv + (cond * inv_dx2 + q[i]) / cap;
                if !wall {
                    dtdt += dpdt / cap;
                }
                row[x] = ti + dts * dtdt;
            }
        });
        std::mem::swap(&mut self.temperature, &mut self.temperature_temp);
    }

    // diffusivité thermique max en cellules²/s (pour la stabilité des sous-pas)
    fn thermal_alpha_max_grid(&self) -> f32 {
        let m = self.material;
        let rho_f = if self.low_mach_active() { self.rho_min } else { m.rho_ambient() };
        let a_f = m.conductivity / (rho_f.max(1e-6) * m.cp);
        let a_s = m.alpha_solid();
        a_f.max(a_s) / (self.dx * self.dx)
    }

    // ------------------------------------------------------------------ masques / conductances

    #[inline]
    fn offset_factor(x: usize, size: usize, offset: usize) -> f32 {
        if offset == 0 {
            return 1.0;
        }

        if size <= 2 || offset >= size {
            return 0.0;
        }

        let distance_from_edge = x.min(size - 1 - x);

        (distance_from_edge as f32 / offset as f32).clamp(0.0, 1.0)
    }

    fn rebuild_masks(&mut self) {
        let (w, h) = (self.width, self.height);
        let s = &self.solids;
        let lo = self.left_boundary.boundary_type == BoundaryType::Outlet;
        let ro = self.right_boundary.boundary_type == BoundaryType::Outlet;
        let to = self.top_boundary.boundary_type == BoundaryType::Outlet;
        let bo = self.bottom_boundary.boundary_type == BoundaryType::Outlet;

        self.open_u.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            row.fill(0);
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 0..w - 1 {
                let i = x + y * w;
                let open = if x == 0 {
                    lo && s[i + 1] != CellType::Wall
                } else if x == w - 2 {
                    ro && s[i] != CellType::Wall
                } else {
                    s[i] != CellType::Wall && s[i + 1] != CellType::Wall
                };
                row[x] = open as u8;
            }
        });

        self.open_v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            row.fill(0);
            if y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                let open = if y == 0 {
                    to && s[i + w] != CellType::Wall
                } else if y == h - 2 {
                    bo && s[i] != CellType::Wall
                } else {
                    s[i] != CellType::Wall && s[i + w] != CellType::Wall
                };
                row[x] = open as u8;
            }
        });
    }

    // Conductance des faces pour le solveur de pression : open · (ρ_ref/ρ_face).
    // ρ_face = moyenne arithmétique des deux cellules. Incompressible / Boussinesq : 1 si ouverte.
    // C'est le « /ρ » de l'équation de quantité de mouvement (projection à densité variable).
    fn rebuild_conductances(&mut self) {
        let w = self.width;
        let varrho = self.low_mach_active();
        let rho_ref = self.material.rho_ambient();
        let (rho, ou, ov) = (&self.rho, &self.open_u, &self.open_v);
        let beta = |a: usize, b: usize| 2.0 * rho_ref / (rho[a] + rho[b]).max(1e-9);

        self.cond_u.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x + y * w;
                row[x] = if ou[i] == 0 { 0.0 } else if varrho { beta(i, i + 1) } else { 1.0 };
            }
        });
        self.cond_v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x + y * w;
                row[x] = if ov[i] == 0 { 0.0 } else if varrho { beta(i, i + w) } else { 1.0 };
            }
        });
    }

    // ------------------------------------------------------------------ advection de la vitesse

    fn sample_face(field: &[f32], solids: &[CellType], w: usize, h: usize, horiz: bool, px: f32, py: f32, fallback: f32) -> f32 {
        let (gx, gy, xm, ym) = if horiz {
            (px - 0.5, py, w - 2, h - 1)
        } else {
            (px, py - 0.5, w - 1, h - 2)
        };
        let gx = gx.clamp(0.0, xm as f32);
        let gy = gy.clamp(0.0, ym as f32);

        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(xm);
        let y1 = (y0 + 1).min(ym);

        let sx = gx - x0 as f32;
        let sy = gy - y0 as f32;

        let stride = if horiz { 1 } else { w };
        let taps = [
            (x0, y0, (1.0 - sx) * (1.0 - sy)),
            (x1, y0, sx * (1.0 - sy)),
            (x0, y1, (1.0 - sx) * sy),
            (x1, y1, sx * sy),
        ];

        let mut sum = 0.0;
        let mut weight = 0.0;
        for (x, y, wt) in taps {
            let i = x + y * w;
            if solids[i] == CellType::Wall && solids[i + stride] == CellType::Wall {
                continue;
            }
            sum += field[i] * wt;
            weight += wt;
        }

        if weight > 1e-6 { sum / weight } else { fallback }
    }

    pub fn advect_velocity(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.velocity_x_temp.copy_from_slice(&self.velocity_x);
        self.velocity_y_temp.copy_from_slice(&self.velocity_y);

        {
            let (u, v, s, ou) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_u);
            self.velocity_x_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 2 {
                    let i = x + y * w;
                    if ou[i] == 0 {
                        continue; // face fermée : fixée par les conditions aux limites
                    }
                    let (px, py) = (x as f32 + 0.5, y as f32);
                    let vx = u[i];
                    let vy = Self::sample_face(v, s, w, h, false, px, py, 0.0);
                    row[x] = Self::sample_face(u, s, w, h, true, px - vx * dt, py - vy * dt, vx);
                }
            });
        }

        {
            let (u, v, s, ov) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_v);
            self.velocity_y_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 2 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if ov[i] == 0 {
                        continue;
                    }
                    let (px, py) = (x as f32, y as f32 + 0.5);
                    let vy = v[i];
                    let vx = Self::sample_face(u, s, w, h, true, px, py, 0.0);
                    row[x] = Self::sample_face(v, s, w, h, false, px - vx * dt, py - vy * dt, vy);
                }
            });
        }

        std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
        std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
    }

    // ------------------------------------------------------------------ conditions aux limites

    // CL des faces à appliquer avant la projection
    // (composante normale, composante tangentielle fantôme). no-slip : fantôme = 2·U_paroi − intérieur.
    fn edge_values(b: &GridBoundary, no_slip: bool, factor: f32, inlet_n: f32, inlet_t: f32, inner_n: f32, inner_t: f32) -> (f32, f32) {
        match b.boundary_type {
            BoundaryType::Wall => (0.0, if no_slip { -inner_t } else { inner_t }),
            BoundaryType::Outlet => (inner_n, inner_t), // grad = 0
            BoundaryType::Inlet => (inlet_n * factor, if no_slip { 2.0 * inlet_t * factor - inner_t } else { inlet_t * factor }),
        }
    }

    fn apply_velocity_boundaries(&mut self) {
        let (w, h) = (self.width, self.height);
        let ns = self.no_slip;
        let dx = self.dx;

        // faces intérieures fermées par un solide -> 0
        for y in 1..h - 1 {
            for x in 1..w - 2 {
                let i = x + y * w;
                if self.open_u[i] == 0 {
                    self.velocity_x[i] = 0.0;
                }
            }
        }
        for y in 1..h - 2 {
            for x in 1..w - 1 {
                let i = x + y * w;
                if self.open_v[i] == 0 {
                    self.velocity_y[i] = 0.0;
                }
            }
        }

        // gauche / droite : normale = velocity_x (face de bord), tangentielle = velocity_y (fantôme)
        for y in 1..h - 1 {
            let row = y * w;

            let b = self.left_boundary.to_grid(dx);
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vx, b.vy,
                                           self.velocity_x[row + 1], self.velocity_y[row + 1]);
            self.velocity_x[row] = if self.solids[row + 1] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row] = t;

            let b = self.right_boundary.to_grid(dx);
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vx, b.vy,
                                           self.velocity_x[row + w - 3], self.velocity_y[row + w - 2]);
            self.velocity_x[row + w - 2] = if self.solids[row + w - 2] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row + w - 1] = t;
        }

        // haut / bas : normale = velocity_y (face de bord), tangentielle = velocity_x (fantôme)
        for x in 1..w - 1 {
            let last = (h - 2) * w;

            let b = self.top_boundary.to_grid(dx);
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vy, b.vx,
                                           self.velocity_y[x + w], self.velocity_x[x + w]);
            self.velocity_y[x] = if self.solids[x + w] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x] = t;

            let b = self.bottom_boundary.to_grid(dx);
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vy, b.vx,
                                           self.velocity_y[x + last - w], self.velocity_x[x + last]);
            self.velocity_y[x + last] = if self.solids[x + last] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x + last + w] = t;
        }
    }

    /// Vitesse maximale autorisée pour que le nombre de sous-pas de scalaires reste <= MAX_SUBSTEPS.
    /// Partagée avec le GPU (limit_velocity dans le shader).
    pub(crate) fn velocity_limit(&self, dt: f32) -> f32 {
        CFL * MAX_SUBSTEPS as f32 / (4.0 * dt.max(1e-6))
    }

    fn limit_velocity(&mut self, dt: f32) {
        let vmax = self.velocity_limit(dt);
        let fix = |v: &mut f32| {
            *v = if v.is_finite() { v.clamp(-vmax, vmax) } else { 0.0 };
        };
        self.velocity_x.par_iter_mut().for_each(fix);
        self.velocity_y.par_iter_mut().for_each(fix);
    }

    fn apply_special_velocity(&mut self, dt: f32) {
        let dx = self.dx;
        for &(idx, effect) in self.special_cells.iter() {
            match effect.to_grid(dx) {
                CellEffect::Emitter { vx, vy, .. } => {
                    self.velocity_x[idx] = vx;
                    self.velocity_y[idx] = vy;
                }
                CellEffect::Accelerator { ax, ay } => {
                    self.velocity_x[idx] += ax * dt;
                    self.velocity_y[idx] += ay * dt;
                }
                CellEffect::Teleporter { dest_idx } => {
                    if dest_idx != idx && self.solids[dest_idx] != CellType::Wall {
                        self.velocity_x[dest_idx] = self.velocity_x[idx];
                        self.velocity_y[dest_idx] = self.velocity_y[idx];
                    }
                }
                CellEffect::Sink { rate } => {
                    self.velocity_x[idx] = -rate;
                    self.velocity_x[idx - 1] = rate;
                    self.velocity_y[idx] = -rate;
                    self.velocity_y[idx - self.width] = rate;
                    self.pressure[idx] = 0.0;
                }
            }
        }
    }

    // ------------------------------------------------------------------ low-Mach : divergence cible

    /// Flux volumique sortant par les faces du bord (somme sur les faces de bord, cellules/s·cellule).
    /// Les faces fermées par un solide valent déjà 0 (apply_velocity_boundaries).
    fn boundary_outflow(&self) -> f64 {
        let (w, h) = (self.width, self.height);
        let (u, v) = (&self.velocity_x, &self.velocity_y);
        let mut q = 0.0f64;
        for y in 1..h - 1 {
            q += (-u[y * w] + u[(w - 2) + y * w]) as f64;
        }
        for x in 1..w - 1 {
            q += (-v[x] + v[x + (h - 2) * w]) as f64;
        }
        q
    }

    /// Même flux sortant que `boundary_outflow`, mais déduit des seules conditions aux limites
    /// (valeurs de bord après apply_velocity_boundaries + limit_velocity), donc calculable AVANT
    /// d'avoir les vitesses du pas. Valable en boîte fermée (aucun Outlet) : bord = 0 (Wall)
    /// ou facteur·vitesse (Inlet), 0 si la cellule voisine est solide.
    pub(crate) fn boundary_outflow_analytic(&self, dt: f32) -> f64 {
        let (w, h) = (self.width, self.height);
        let vmax = self.velocity_limit(dt);
        let s = &self.solids;
        let dx = self.dx;
        let face = |b: &GridBoundary, f: f32, comp: f32, solid_neighbor: bool| -> f64 {
            if solid_neighbor || b.boundary_type != BoundaryType::Inlet {
                return 0.0;
            }
            let v = comp * f;
            (if v.is_finite() { v.clamp(-vmax, vmax) } else { 0.0 }) as f64
        };
        let mut q = 0.0f64;
        for y in 1..h - 1 {
            let row = y * w;
            let (bl, br) = (self.left_boundary.to_grid(dx), self.right_boundary.to_grid(dx));
            q -= face(&bl, Self::offset_factor(y, h, bl.offset), bl.vx, s[row + 1] == CellType::Wall);
            q += face(&br, Self::offset_factor(y, h, br.offset), br.vx, s[row + w - 2] == CellType::Wall);
        }
        for x in 1..w - 1 {
            let last = (h - 2) * w;
            let (bt, bb) = (self.top_boundary.to_grid(dx), self.bottom_boundary.to_grid(dx));
            q -= face(&bt, Self::offset_factor(x, w, bt.offset), bt.vy, s[x + w] == CellType::Wall);
            q += face(&bb, Self::offset_factor(x, w, bb.offset), bb.vy, s[x + last] == CellType::Wall);
        }
        q
    }

    // Gaz low-Mach : on impose  div u = S  avec
    //   S = (∇·(k∇T) + q)/(ρ·cp·T)  −  (1/(γ·P_th))·dP_th/dt
    // - domaine ouvert (Outlet ou Sink) : P_th = P_ref, dP_th/dt = 0
    // - boîte fermée : dP_th/dt est choisi pour que le problème de Poisson (Neumann) soit
    //   résoluble, i.e. Σ S = flux sortant par les faces de bord (Inlets).
    // Incompressible / Boussinesq : S = 0.
    //
    // `q_out` : flux de bord déjà connu (chemin GPU, où les vitesses ne sont pas encore calculées) ;
    // None = on le lit dans les tableaux de vitesse (chemin CPU).
    pub(crate) fn compute_source(&mut self, dt: f32, q_out: Option<f64>) {
        if !self.low_mach_active() {
            self.source.fill(0.0);
            self.dpdt = 0.0;
            self.p_th = self.material.p_ref;
            return;
        }
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let (kf, ks, cp) = (m.conductivity, m.solid_conductivity, m.cp);

        let (sum_a, n_active) = {
            let (t, s, rho, q) = (&self.temperature, &self.solids, &self.rho, &self.heat_source);
            let k_of = |i: usize| if s[i] == CellType::Wall { ks } else { kf };
            let k_face = |a: usize, b: usize| {
                let (ka, kb) = (k_of(a), k_of(b));
                let sum = ka + kb;
                if sum <= 0.0 { 0.0 } else { 2.0 * ka * kb / sum }
            };

            self.source.par_chunks_mut(w).enumerate().map(|(y, row)| {
                row.fill(0.0);
                if y == 0 || y >= h - 1 {
                    return (0.0f64, 0u64);
                }
                let (mut sum, mut n) = (0.0f64, 0u64);
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] == CellType::Wall || s[i] == CellType::Sink {
                        continue;
                    }
                    let ti = t[i];
                    let mut cond = 0.0;
                    if x + 2 < w { cond += k_face(i, i + 1) * (t[i + 1] - ti); }
                    if x >= 2 { cond += k_face(i, i - 1) * (t[i - 1] - ti); }
                    if y + 2 < h { cond += k_face(i, i + w) * (t[i + w] - ti); }
                    if y >= 2 { cond += k_face(i, i - w) * (t[i - w] - ti); }

                    let a = (cond * inv_dx2 + q[i]) / (rho[i] * cp * ti.max(1.0));
                    row[x] = a;
                    sum += a as f64;
                    n += 1;
                }
                (sum, n)
            }).reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        };

        if self.is_closed() && n_active > 0 {
            let q_out = match q_out {
                Some(q) => q,
                None => {
                    let q = self.boundary_outflow();
                    debug_assert!(
                        (q - self.boundary_outflow_analytic(dt)).abs() <= 1e-3 * (1.0 + q.abs()),
                        "flux de bord analytique != flux lu dans les vitesses"
                    );
                    q
                }
            };
            let delta = ((sum_a - q_out) / n_active as f64) as f32; // = dP_th/dt / (γ·P_th)
            let gamma = m.gamma();
            self.dpdt = gamma * self.p_th * delta;
            self.p_th = (self.p_th + self.dpdt * dt).max(1.0);

            let s = &self.solids;
            self.source.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] != CellType::Wall && s[i] != CellType::Sink {
                        row[x] -= delta;
                    }
                }
            });
        } else {
            self.dpdt = 0.0;
            self.p_th = m.p_ref;
        }
    }

    // ------------------------------------------------------------------ projection

    fn calculate_divergence(&mut self) {
        let (w, h) = (self.width, self.height);
        let (u, v, s) = (&self.velocity_x, &self.velocity_y, &self.solids);

        self.divergence.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                row[x] = if s[i] == CellType::Wall || s[i] == CellType::Sink {
                    0.0
                } else {
                    u[i] - u[i - 1] + v[i] - v[i - w]
                };
            }
        });
    }

    // Poisson à coefficients variables : Σ_faces c_f·(p_voisin − p_i) = div_i − S_i
    // (c_f = conductance de la face, voir rebuild_conductances), Red-Black SOR.
    fn solve_pressure(&mut self, iterations: usize) {
        let (w, h) = (self.width, self.height);
        let omega = self.sor_omega;
        let shared = SharedMut(self.pressure.as_mut_ptr());
        let (div, src, cu, cv, s) = (&self.divergence, &self.source, &self.cond_u, &self.cond_v, &self.solids);

        for _ in 0..iterations {
            for color in 0..2usize {
                (1..h - 1).into_par_iter().for_each(|y| {
                    let p = shared.ptr();
                    let mut x = if ((1 + y) & 1) == color { 1 } else { 2 };
                    while x < w - 1 {
                        let i = x + y * w;
                        if s[i] != CellType::Wall && s[i] != CellType::Sink {
                            let a = cu[i];
                            let b = cu[i - 1];
                            let c = cv[i];
                            let d = cv[i - w];
                            let n = a + b + c + d;
                            if n > 0.0 {
                                unsafe {
                                    let sum = a * *p.add(i + 1) + b * *p.add(i - 1)
                                        + c * *p.add(i + w) + d * *p.add(i - w);
                                    let p_new = (sum - (div[i] - src[i])) / n;
                                    let old = *p.add(i);
                                    *p.add(i) = old + omega * (p_new - old);
                                }
                            }
                        }
                        x += 2;
                    }
                });
            }
        }
    }

    // Problème de Neumann pur (boîte fermée) : la pression n'est définie qu'à une constante près.
    // On retire la moyenne pour qu'elle ne dérive pas.
    pub(crate) fn remove_pressure_mean(&mut self) {
        let (w, h) = (self.width, self.height);
        let s = &self.solids;
        let p = &self.pressure;
        let (sum, n) = (1..h - 1).into_par_iter().map(|y| {
            let (mut sum, mut n) = (0.0f64, 0u64);
            for x in 1..w - 1 {
                let i = x + y * w;
                if s[i] != CellType::Wall && s[i] != CellType::Sink {
                    sum += p[i] as f64;
                    n += 1;
                }
            }
            (sum, n)
        }).reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        if n == 0 {
            return;
        }
        let mean = (sum / n as f64) as f32;
        self.pressure.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if s[i] != CellType::Wall && s[i] != CellType::Sink {
                    row[x] -= mean;
                }
            }
        });
    }

    fn correct_velocity(&mut self) {
        let (w, h) = (self.width, self.height);
        let (p, cu, cv) = (&self.pressure, &self.cond_u, &self.cond_v);

        self.velocity_x.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 0..w - 1 {
                let i = x + y * w;
                if cu[i] > 0.0 {
                    row[x] -= cu[i] * (p[i + 1] - p[i]);
                }
            }
        });

        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if cv[i] > 0.0 {
                    row[x] -= cv[i] * (p[i + w] - p[i]);
                }
            }
        });
    }

    pub fn enforce_incompressibility_cpu(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        if self.is_closed() {
            self.remove_pressure_mean();
        }
        self.correct_velocity();
        self.calculate_divergence(); // divergence résiduelle : doit valoir S
    }

    // ------------------------------------------------------------------ tracer (colorant)

    // fantômes du tracer : Inlet = valeur, Wall = copie, Outlet = 0 avant l'advection, copie après
    fn ghost_tracer(&mut self, before_advection: bool) {
        let (w, h) = (self.width, self.height);
        let ghost = |b: &Boundary, inner: f32| match b.boundary_type {
            BoundaryType::Inlet => b.tracer,
            BoundaryType::Outlet if before_advection => 0.0,
            _ => inner,
        };
        for y in 1..h - 1 {
            let (l, r) = (y * w, (w - 1) + y * w);
            self.tracer[l] = ghost(&self.left_boundary, self.tracer[l + 1]);
            self.tracer[r] = ghost(&self.right_boundary, self.tracer[r - 1]);
        }
        for x in 1..w - 1 {
            let (t, b) = (x, x + (h - 1) * w);
            self.tracer[t] = ghost(&self.top_boundary, self.tracer[t + w]);
            self.tracer[b] = ghost(&self.bottom_boundary, self.tracer[b - w]);
        }
    }

    /// Nombre de sous-pas des scalaires (tracer + température), garantissant
    /// dt_sub · (flux sortant + 4·diffusion max) <= CFL.
    pub(crate) fn scalar_substeps(&self, dt: f32) -> usize {
        let (w, h) = (self.width, self.height);
        let (u, v, s) = (&self.velocity_x, &self.velocity_y, &self.solids);

        let rate = (1..h - 1)
            .into_par_iter()
            .map(|y| {
                let mut m = 0.0f32;
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] == CellType::Wall {
                        continue;
                    }
                    let out = u[i].max(0.0) + (-u[i - 1]).max(0.0) + v[i].max(0.0) + (-v[i - w]).max(0.0);
                    m = m.max(out);
                }
                m
            })
            .reduce(|| 0.0, f32::max);

        let diffusion = self.thermal_alpha_max_grid().max(self.tracer_diff_grid());
        let n = (dt * (rate + 4.0 * diffusion) / CFL).ceil();
        (n as usize).clamp(1, MAX_SUBSTEPS)
    }

    // Tracer + température en sous-pas. Le tracer est une FRACTION (pas une densité) :
    // avec div u ≠ 0 (low-Mach) on advecte sa forme non conservative (flux − Y·div u),
    // sinon il serait comprimé/dilué par l'expansion du gaz.
    pub fn advect_diffuse_scalars(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.ghost_tracer(true);
        self.update_ghost_temperature(true);
        let n = self.scalar_substeps(dt);
        let dts = dt / n as f32;
        let expansion = self.low_mach_active();
        let d = self.tracer_diff_grid();

        for _ in 0..n {
            self.tracer_temp.copy_from_slice(&self.tracer); // conserve les cellules fantômes

            let kernel = TracerKernel { w, h, u: &self.velocity_x, v: &self.velocity_y,
                open_u: &self.open_u, open_v: &self.open_v, d };
            let (tr, s) = (&self.tracer, &self.solids);
            let (u, v) = (&self.velocity_x, &self.velocity_y);

            self.tracer_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y == h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    row[x] = if s[i] == CellType::Wall {
                        0.0
                    } else {
                        let mut val = tr[i] - dts * (kernel.fx(tr, x, y) - kernel.fx(tr, x - 1, y)
                            + kernel.fy(tr, x, y) - kernel.fy(tr, x, y - 1));
                        if expansion {
                            val += dts * tr[i] * (u[i] - u[i - 1] + v[i] - v[i - w]);
                        }
                        val
                    };
                }
            });

            std::mem::swap(&mut self.tracer, &mut self.tracer_temp);
            self.temperature_substep(dts);
        }
        self.heat_source.fill(0.0); // la source de chaleur du pinceau est consommée
    }

    fn apply_special_tracer(&mut self, _dt: f32) {
        for &(idx, effect) in self.special_cells.iter() {
            match effect {
                CellEffect::Emitter { tracer, .. } => {
                    self.tracer[idx] = tracer;
                }
                CellEffect::Sink { .. } => {
                    self.tracer[idx] = 0.0;
                }
                CellEffect::Teleporter { dest_idx } => {
                    if dest_idx != idx && self.solids[dest_idx] != CellType::Wall {
                        self.tracer[dest_idx] = (self.tracer[dest_idx] + self.tracer[idx]).min(1.0); //todo attention au surplus
                        self.tracer[idx] = 0.0;
                    }
                }
                CellEffect::Accelerator { .. } => {}
            }
        }
    }

    // ------------------------------------------------------------------ pas complet

    pub(crate) fn step_cpu(&mut self, dt: f32, pressure_iterations: usize) {
        self.update_rho(); // ρ(T, P_th)
        self.rebuild_masks(); // solides | types de bordure -> faces ouvertes
        self.rebuild_conductances(); // faces ouvertes * ρ_ref/ρ_face

        self.advect_velocity(dt); // advection semi-lagrangienne des faces
        self.diffuse_velocity(dt);
        self.apply_buoyancy(dt);
        self.apply_special_velocity(dt);

        self.apply_velocity_boundaries(); // CL + faces fermées = 0
        self.limit_velocity(dt);

        self.compute_source(dt, None); // low-Mach : S et dP_th/dt (0 sinon)
        self.enforce_incompressibility_cpu(pressure_iterations); // div u = S

        self.advect_diffuse_scalars(dt); // tracer + température, sous-pas CFL
        self.apply_special_tracer(dt);
        self.ghost_tracer(false); // fantômes, affichage
        self.update_ghost_temperature(false);

        self.update_pressure_pa(dt);
    }
}