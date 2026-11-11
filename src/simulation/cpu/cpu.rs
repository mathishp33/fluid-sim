use rayon::prelude::*;
use crate::simulation::fluid::{Boundary, BoundaryType, CellEffect, CellType, FluidSim};

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

struct TemperatureKernel<'a> {
    w: usize,
    h: usize,
    u: &'a [f32],
    v: &'a [f32],
    solids: &'a [CellType],
    alpha_fluid: f32,
    alpha_solid: f32,
}

impl TemperatureKernel<'_> {
    #[inline]
    fn alpha(&self, i: usize) -> f32 {
        if self.solids[i] == CellType::Wall { self.alpha_solid } else { self.alpha_fluid }
    }

    #[inline]
    fn face_alpha(&self, i: usize, j: usize) -> f32 {
        let (a, b) = (self.alpha(i), self.alpha(j));
        let s = a + b;
        if s <= 0.0 { 0.0 } else { 2.0 * a * b / s } // moyenne harmonique
    }

    #[inline]
    fn fx(&self, t: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let uf = self.u[i];
        let adv = uf * if uf > 0.0 { t[i] } else { t[i + 1] };
        let dif = if x >= 1 && x + 2 < self.w {
            -self.face_alpha(i, i + 1) * (t[i + 1] - t[i])
        } else {
            0.0
        };
        adv + dif
    }

    #[inline]
    fn fy(&self, t: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let vf = self.v[i];
        let adv = vf * if vf > 0.0 { t[i] } else { t[i + self.w] };
        let dif = if y >= 1 && y + 2 < self.h {
            -self.face_alpha(i, i + self.w) * (t[i + self.w] - t[i])
        } else {
            0.0
        };
        adv + dif
    }
}

// Flux de densité (advection + diffusion) les faces de la grille MAC.
struct DensityKernel<'a> {
    w: usize,
    h: usize,
    u: &'a [f32],
    v: &'a [f32],
    open_u: &'a [u8],
    open_v: &'a [u8],
    d: f32,
}

impl DensityKernel<'_> {
    #[inline]
    fn fx(&self, rho: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let uf = self.u[i];
        let adv = uf * if uf > 0.0 {
            rho[i]
        } else {
            rho[i + 1]
        };
        let dif = if x >= 1 && x + 2 < self.w && self.open_u[i] != 0 {
            -self.d * (rho[i + 1] - rho[i])
        } else {
            0.0
        };
        adv + dif
    }

    #[inline]
    fn fy(&self, rho: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let vf = self.v[i];
        let adv = vf * if vf > 0.0 {
            rho[i]
        } else {
            rho[i + self.w]
        };
        let dif = if y >= 1 && y + 2 < self.h && self.open_v[i] != 0 {
            -self.d * (rho[i + self.w] - rho[i])
        } else {
            0.0
        };
        adv + dif
    }
}

impl FluidSim {

    pub(crate) fn viscosity_substeps(&self, dt: f32) -> usize {
        ((4.0 * self.viscosity * dt / CFL).ceil() as usize).clamp(1, MAX_VISC_SUBSTEPS)
    }

    pub fn diffuse_velocity(&mut self, dt: f32) {
        if self.viscosity <= 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let n = self.viscosity_substeps(dt);
        let k = self.viscosity * dt / n as f32;

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
                        let mut lap = 0.0;
                        if ou[i + 1] != 0 { lap += u[i + 1] - c; }
                        if ou[i - 1] != 0 { lap += u[i - 1] - c; }
                        if ou[i + w] != 0 { lap += u[i + w] - c; }
                        if ou[i - w] != 0 { lap += u[i - w] - c; }
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
                        let mut lap = 0.0;
                        if ov[i + 1] != 0 { lap += v[i + 1] - c; }
                        if ov[i - 1] != 0 { lap += v[i - 1] - c; }
                        if ov[i + w] != 0 { lap += v[i + w] - c; }
                        if ov[i - w] != 0 { lap += v[i - w] - c; }
                        row[x] = c + k * lap;
                    }
                });
            }

            std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
            std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
        }
    }

    // dv/dt = -buoyancy * T (T > 0 monte, y pointe vers le bas)
    fn apply_buoyancy(&mut self, dt: f32) {
        if self.buoyancy == 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let (t, ov, g) = (&self.temperature, &self.open_v, self.buoyancy * dt);
        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 2 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if ov[i] != 0 {
                    row[x] -= g * 0.5 * (t[i] + t[i + w]);
                }
            }
        });
    }

    // fantômes : Inlet = valeur, Wall = adiabatique,
    // Outlet = ambiante (0) avant l'advection copie après
    fn update_ghost_temperature(&mut self, before_advection: bool) {
        let (w, h) = (self.width, self.height);
        let ghost = |b: &Boundary, inner: f32| match b.boundary_type {
            BoundaryType::Inlet => b.temperature,
            BoundaryType::Outlet if before_advection => 0.0,
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

    fn temperature_substep(&mut self, dts: f32) {
        let (w, h) = (self.width, self.height);
        self.temperature_temp.copy_from_slice(&self.temperature); // conserve les fantômes

        let kernel = TemperatureKernel {
            w, h,
            u: &self.velocity_x,
            v: &self.velocity_y,
            solids: &self.solids,
            alpha_fluid: self.thermal_diffusion,
            alpha_solid: self.solid_thermal_diffusion,
        };
        let t = &self.temperature;

        self.temperature_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                return;
            }
            for x in 1..w - 1 {
                row[x] = t[x + y * w]
                    - dts * (kernel.fx(t, x, y) - kernel.fx(t, x - 1, y)
                    + kernel.fy(t, x, y) - kernel.fy(t, x, y - 1));
            }
        });
        std::mem::swap(&mut self.temperature, &mut self.temperature_temp);
    }

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

    //  Advection semi-lagrangienne sur les faces

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

    //CL des faces à appliquer avant la projection
    fn edge_values(b: &Boundary, factor: f32, inlet_n: f32, inlet_t: f32, inner_n: f32, inner_t: f32) -> (f32, f32) {
        match b.boundary_type {
            BoundaryType::Wall => (0.0, inner_t), // free slip
            BoundaryType::Outlet => (inner_n, inner_t), // grad = 0
            BoundaryType::Inlet => (inlet_n * factor, inlet_t * factor),
        }
    }

    fn apply_velocity_boundaries(&mut self) {
        let (w, h) = (self.width, self.height);

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

            let b = self.left_boundary;
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, f, b.velocity_x, b.velocity_y,
                                           self.velocity_x[row + 1], self.velocity_y[row + 1]);
            self.velocity_x[row] = if self.solids[row + 1] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row] = t;

            let b = self.right_boundary;
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, f, b.velocity_x, b.velocity_y,
                                           self.velocity_x[row + w - 3], self.velocity_y[row + w - 2]);
            self.velocity_x[row + w - 2] = if self.solids[row + w - 2] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row + w - 1] = t;
        }

        // haut / bas : normale = velocity_y (face de bord), tangentielle = velocity_x (fantôme)
        for x in 1..w - 1 {
            let last = (h - 2) * w;

            let b = self.top_boundary;
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, f, b.velocity_y, b.velocity_x,
                                           self.velocity_y[x + w], self.velocity_x[x + w]);
            self.velocity_y[x] = if self.solids[x + w] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x] = t;

            let b = self.bottom_boundary;
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, f, b.velocity_y, b.velocity_x,
                                           self.velocity_y[x + last - w], self.velocity_x[x + last]);
            self.velocity_y[x + last] = if self.solids[x + last] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x + last + w] = t;
        }
    }

    /// Vitesse maximale autorisée pour que le nombre de sous-pas de densité reste <= MAX_SUBSTEPS.
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
        for &(idx, effect) in self.special_cells.iter() {
            match effect {
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

    fn solve_pressure(&mut self, iterations: usize) {
        let (w, h) = (self.width, self.height);
        let omega = self.sor_omega;
        let shared = SharedMut(self.pressure.as_mut_ptr());
        let (div, ou, ov, s) = (&self.divergence, &self.open_u, &self.open_v, &self.solids);

        for _ in 0..iterations {
            for color in 0..2usize {
                (1..h - 1).into_par_iter().for_each(|y| {
                    let p = shared.ptr();
                    let mut x = if ((1 + y) & 1) == color { 1 } else { 2 };
                    while x < w - 1 {
                        let i = x + y * w;
                        if s[i] != CellType::Wall && s[i] != CellType::Wall {
                            let a = ou[i] as f32;
                            let b = ou[i - 1] as f32;
                            let c = ov[i] as f32;
                            let d = ov[i - w] as f32;
                            let n = a + b + c + d;
                            if n > 0.0 {
                                unsafe {
                                    let sum = a * *p.add(i + 1) + b * *p.add(i - 1)
                                        + c * *p.add(i + w) + d * *p.add(i - w);
                                    let p_new = (sum - div[i]) / n;
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

    fn correct_velocity(&mut self) {
        let (w, h) = (self.width, self.height);
        let (p, ou, ov) = (&self.pressure, &self.open_u, &self.open_v);

        self.velocity_x.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 0..w - 1 {
                let i = x + y * w;
                if ou[i] != 0 {
                    row[x] -= p[i + 1] - p[i];
                }
            }
        });

        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if ov[i] != 0 {
                    row[x] -= p[i + w] - p[i];
                }
            }
        });
    }

    pub fn enforce_incompressibility_cpu(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        self.correct_velocity();
        self.calculate_divergence(); // divergence résiduelle
    }

    fn prepare_ghost_density(&mut self) {
        let (w, h) = (self.width, self.height);

        for y in 1..h - 1 {
            let (l, r) = (y * w, (w - 1) + y * w);
            self.density[l] = match self.left_boundary.boundary_type {
                BoundaryType::Inlet => self.left_boundary.density,
                BoundaryType::Outlet => 0.0,
                BoundaryType::Wall => self.density[l + 1],
            };
            self.density[r] = match self.right_boundary.boundary_type {
                BoundaryType::Inlet => self.right_boundary.density,
                BoundaryType::Outlet => 0.0,
                BoundaryType::Wall => self.density[r - 1],
            };
        }
        for x in 1..w - 1 {
            let (t, b) = (x, x + (h - 1) * w);
            self.density[t] = match self.top_boundary.boundary_type {
                BoundaryType::Inlet => self.top_boundary.density,
                BoundaryType::Outlet => 0.0,
                BoundaryType::Wall => self.density[t + w],
            };
            self.density[b] = match self.bottom_boundary.boundary_type {
                BoundaryType::Inlet => self.bottom_boundary.density,
                BoundaryType::Outlet => 0.0,
                BoundaryType::Wall => self.density[b - w],
            };
        }
    }

    /// nbr de sous-pas garantissant dt_sub * (flux sortant + diffusion) <= CFL
    pub(crate) fn density_substeps(&self, dt: f32) -> usize {
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

        let alpha = self.diffusion_rate.max(self.thermal_diffusion).max(self.solid_thermal_diffusion);
        let n = (dt * (rate + 4.0 * alpha) / CFL).ceil();
        (n as usize).clamp(1, MAX_SUBSTEPS)
    }

    pub fn advect_diffuse_density(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.prepare_ghost_density();
        self.update_ghost_temperature(true);
        let n = self.density_substeps(dt);
        let dts = dt / n as f32;

        for _ in 0..n {
            self.density_temp.copy_from_slice(&self.density); // conserve les cellules fantômes

            let kernel = DensityKernel {w, h, u: &self.velocity_x, v: &self.velocity_y,
                open_u: &self.open_u, open_v: &self.open_v, d: self.diffusion_rate};
            let (rho, s) = (&self.density, &self.solids);

            self.density_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y == h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    row[x] = if s[i] == CellType::Wall {
                        0.0
                    } else {
                        rho[i] - dts * (kernel.fx(rho, x, y) - kernel.fx(rho, x - 1, y)
                            + kernel.fy(rho, x, y) - kernel.fy(rho, x, y - 1))
                    };
                }
            });

            std::mem::swap(&mut self.density, &mut self.density_temp);
            self.temperature_substep(dts);
        }
    }

    /// Cellules fantômes de la densité après advection
    fn apply_density_boundaries(&mut self) {
        //gauche
        for y in 1..self.height - 1 {
            let idx = self.idx(0, y);

            match self.left_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => { self.density[idx] = self.density[self.idx(1, y)]; }
                BoundaryType::Inlet => { self.density[idx] = self.left_boundary.density; }
            }
        }

        //droite
        for y in 1..self.height - 1 {
            let idx = self.idx(self.width - 1, y);

            match self.right_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => { self.density[idx] = self.density[self.idx(self.width - 2, y)]; }
                BoundaryType::Inlet => { self.density[idx] = self.right_boundary.density; }
            }
        }

        //haut
        for x in 1..self.width - 1 {
            let idx = self.idx(x, 0);

            match self.top_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => { self.density[idx] = self.density[self.idx(x, 1)]; }
                BoundaryType::Inlet => { self.density[idx] = self.top_boundary.density; }
            }
        }

        //bas
        for x in 1..self.width - 1 {
            let idx = self.idx(x, self.height - 1);

            match self.bottom_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => { self.density[idx] = self.density[self.idx(x, self.height - 2)]; }
                BoundaryType::Inlet => { self.density[idx] = self.bottom_boundary.density; }
            }
        }
    }

    fn apply_special_density(&mut self, dt: f32) {
        for &(idx, effect) in self.special_cells.iter() {
            match effect {
                CellEffect::Emitter { density, .. } => {
                    self.density[idx] = density;
                }
                CellEffect::Sink { rate } => {
                    self.density[idx] = 0.0;
                }
                CellEffect::Teleporter { dest_idx } => {
                    if dest_idx != idx && self.solids[dest_idx] != CellType::Wall {
                        self.density[dest_idx] = (self.density[dest_idx] + self.density[idx]).min(1.0); //todo attention au surplus
                        self.density[idx] = 0.0;
                    }
                }
                CellEffect::Accelerator { .. } => {}
            }
        }
    }

    pub(crate) fn step_cpu(&mut self, dt: f32, pressure_iterations: usize) {
        self.rebuild_masks(); // solides | types de bordure -> faces ouvertes

        self.advect_velocity(dt); // advection semi-lagrangienne des faces
        self.diffuse_velocity(dt);
        self.apply_buoyancy(dt);
        self.apply_special_velocity(dt);

        self.apply_velocity_boundaries(); // CL + faces fermées = 0
        self.limit_velocity(dt);
        self.enforce_incompressibility_cpu(pressure_iterations); // plus aucune CL de vitesse

        self.advect_diffuse_density(dt); // flux conservatif, sous-pas CFL
        self.apply_special_density(dt);
        self.apply_density_boundaries(); // fantômes, affichage
        self.update_ghost_temperature(false);
    }
}