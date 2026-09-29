use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryType {
    Wall,
    Inlet,
    Outlet,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Boundary {
    pub boundary_type: BoundaryType,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub density: f32,
    pub offset: usize,
}

impl Boundary {
    pub fn wall() -> Self {
        Self {
            boundary_type: BoundaryType::Wall,
            velocity_x: 0.0,
            velocity_y: 0.0,
            density: 0.0,
            offset: 1,
        }
    }
}

pub struct FluidSim {
    pub width: usize,
    pub height: usize,

    pub density: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub diffusion_rate: f32,
    pub pressure: Vec<f32>,
    pub divergence: Vec<f32>,
    pub solids: Vec<bool>,
    pub sor_omega: f32,
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub friction: f32,

    density_temp: Vec<f32>,
    velocity_x_temp: Vec<f32>,
    velocity_y_temp: Vec<f32>,
    pressure_temp: Vec<f32>,

    pub left_boundary: Boundary,
    pub right_boundary: Boundary,
    pub top_boundary: Boundary,
    pub bottom_boundary: Boundary,

    density_rhs: Vec<f32>,
    open: Vec<bool>,
}

impl FluidSim {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    pub fn new(width: usize, height: usize, start_density: f32, diffusion_rate: f32, gravity_x: f32, gravity_y: f32, friction: f32) -> Self {
        let size = width * height;
        let n = width.min(height).max(2) as f32;
        let sor_omega = 2.0 / (1.0 + (std::f32::consts::PI / n).sin());
        FluidSim {
            width,
            height,

            density: vec![start_density; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            diffusion_rate,
            pressure: vec![0.0; size],
            divergence: vec![0.0; size],
            solids: vec![false; size],
            sor_omega,
            gravity_x,
            gravity_y,
            friction,

            density_temp: vec![0.0; size],
            velocity_x_temp: vec![0.0; size],
            velocity_y_temp: vec![0.0; size],
            pressure_temp: vec![0.0; size],

            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),

            density_rhs: vec![0.0; size],
            open: vec![false; size],
        }
    }

    pub fn set_solid(&mut self, x: usize, y: usize, solid: bool) {
        if x == 0 || x >= self.width - 1 || y == 0 || y >= self.height - 1 {
            return;
        }
        let idx = self.idx(x, y);

        if self.solids[idx] == solid {
            return;
        }

        self.solids[idx] = solid;

        if solid {
            self.density[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
            self.divergence[idx] = 0.0;

            self.density_temp[idx] = 0.0;
            self.velocity_x_temp[idx] = 0.0;
            self.velocity_y_temp[idx] = 0.0;
            self.pressure_temp[idx] = 0.0;

            return;
        }

        let neighbors = [
            self.idx(x - 1, y),
            self.idx(x + 1, y),
            self.idx(x, y - 1),
            self.idx(x, y + 1),
        ];

        let mut density_sum = 0.0;
        let mut velocity_x_sum = 0.0;
        let mut velocity_y_sum = 0.0;
        let mut pressure_sum = 0.0;
        let mut count = 0.0;

        for neighbor in neighbors {
            if self.solids[neighbor] {
                continue;
            }

            density_sum += self.density[neighbor];
            velocity_x_sum += self.velocity_x[neighbor];
            velocity_y_sum += self.velocity_y[neighbor];
            pressure_sum += self.pressure[neighbor];
            count += 1.0;
        }

        if count > 0.0 {
            self.density[idx] = density_sum / count;
            self.velocity_x[idx] = velocity_x_sum / count;
            self.velocity_y[idx] = velocity_y_sum / count;
            self.pressure[idx] = pressure_sum / count;
        } else {
            self.density[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
        }

        self.divergence[idx] = 0.0;

        self.density_temp[idx] = self.density[idx];
        self.velocity_x_temp[idx] = self.velocity_x[idx];
        self.velocity_y_temp[idx] = self.velocity_y[idx];
        self.pressure_temp[idx] = self.pressure[idx];
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

    pub fn pressure_residual(&self) -> f32 {
        // résidu = |div(v) - (pression appliquée)|
        self.divergence
            .iter()
            .zip(self.solids.iter())
            .filter(|(_, &s)| !s)
            .map(|(d, _)| d.abs())
            .fold(0.0, f32::max)
    }

    pub fn randomize_density_smoothed(&mut self, seed_count: usize) { //O(n)
        let mut rng = rand::thread_rng();

        self.density.fill(0.0);

        for _ in 0..seed_count {
            let x = rng.gen_range(1..self.width - 1);
            let y = rng.gen_range(1..self.height - 1);
            let idx = self.idx(x, y);
            self.density[idx] = rng.gen_range(0.5..1.0);
        }

        for _ in 0..20 {
            self.diffuse_density(0.1, 1);
        }
    }

    pub fn get_density(&self, x: usize, y: usize) -> f32 {
        self.density[self.idx(x, y)]
    } //O(1)

    fn diffuse_field(field: &mut Vec<f32>, temp: &mut Vec<f32>, rhs: &mut Vec<f32>, solids: &[bool], width: usize, height: usize, a: f32, iters: usize, ) {
        rhs.copy_from_slice(field);
        temp.copy_from_slice(field);

        for _ in 0..iters {
            {
                let cur: &Vec<f32> = field;
                let rhs: &Vec<f32> = rhs;
                temp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                    if y == 0 || y == height - 1 {
                        return;
                    }
                    for x in 1..width - 1 {
                        let idx = x + y * width;
                        if solids[idx] {
                            row[x] = 0.0;
                            continue;
                        }
                        let c = cur[idx];
                        let r = if solids[idx + 1] { c } else { cur[idx + 1] };
                        let l = if solids[idx - 1] { c } else { cur[idx - 1] };
                        let d = if solids[idx + width] { c } else { cur[idx + width] };
                        let u = if solids[idx - width] { c } else { cur[idx - width] };
                        row[x] = (rhs[idx] + a * (r + l + d + u)) / (1.0 + 4.0 * a);
                    }
                });
            }
            std::mem::swap(field, temp);
        }
    }

    pub fn diffuse_density(&mut self, dt: f32, diffusion_iters: usize) {
        let a = self.diffusion_rate * dt;
        Self::diffuse_field(&mut self.density, &mut self.density_temp, &mut self.density_rhs,
                            &self.solids, self.width, self.height, a, diffusion_iters);
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    } //O(1)

    fn sample_field(field: &Vec<f32>, width: usize, height: usize, x: f32, y: f32) -> f32 { //O(1)
        let w = width as isize;
        let h = height as isize;

        let x0 = x.floor().clamp(0.0, (w - 1) as f32) as isize;
        let y0 = y.floor().clamp(0.0, (h - 1) as f32) as isize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);

        let sx = x - x0 as f32;
        let sy = y - y0 as f32;

        let idx_00 = (x0 as usize) + (y0 as usize) * width;
        let idx_10 = (x1 as usize) + (y0 as usize) * width;
        let idx_01 = (x0 as usize) + (y1 as usize) * width;
        let idx_11 = (x1 as usize) + (y1 as usize) * width;

        let v00 = field[idx_00];
        let v10 = field[idx_10];
        let v01 = field[idx_01];
        let v11 = field[idx_11];

        let a = Self::lerp(v00, v10, sx);
        let b = Self::lerp(v01, v11, sx);
        Self::lerp(a, b, sy)
    }

    fn sample_field_solid_aware(field: &[f32], solids: &[bool], width: usize, height: usize, x: f32, y: f32, fallback: f32) -> f32 {
        let x = x.clamp(0.0, (width - 1) as f32);
        let y = y.clamp(0.0, (height - 1) as f32);

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;

        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let sx = x - x0 as f32;
        let sy = y - y0 as f32;

        let idx00 = x0 + y0 * width;
        let idx10 = x1 + y0 * width;
        let idx01 = x0 + y1 * width;
        let idx11 = x1 + y1 * width;

        let w00 = (1.0 - sx) * (1.0 - sy);
        let w10 = sx * (1.0 - sy);
        let w01 = (1.0 - sx) * sy;
        let w11 = sx * sy;

        let v00 = if solids[idx00] { fallback } else { field[idx00] };
        let v10 = if solids[idx10] { fallback } else { field[idx10] };
        let v01 = if solids[idx01] { fallback } else { field[idx01] };
        let v11 = if solids[idx11] { fallback } else { field[idx11] };

        v00 * w00
            + v10 * w10
            + v01 * w01
            + v11 * w11
    }

    pub fn advect_density(&mut self, dt: f32) {
        self.density_temp.copy_from_slice(&self.density);

        let width = self.width;
        let height = self.height;

        let density = &self.density;
        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;
        let solids = &self.solids;

        let left_boundary = self.left_boundary;
        let right_boundary = self.right_boundary;
        let top_boundary = self.top_boundary;
        let bottom_boundary = self.bottom_boundary;

        self.density_temp
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y == height - 1 {
                    return;
                }

                for x in 1..width - 1 {
                    let idx = x + y * width;
                    if solids[idx] {
                        row[x] = 0.0;
                        continue;
                    }

                    let vx = velocity_x[idx];
                    let vy = velocity_y[idx];

                    let px = x as f32 - vx * dt;
                    let py = y as f32 - vy * dt;

                    let value = if px < 0.0 {
                        match left_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,
                            BoundaryType::Inlet => { left_boundary.density }
                            BoundaryType::Wall => { density[1 + y * width] }
                        }
                    } else if px >= width as f32 {
                        match right_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,
                            BoundaryType::Inlet => { right_boundary.density }
                            BoundaryType::Wall => { density[(width - 2) + y * width] }
                        }
                    } else if py < 0.0 {
                        match top_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,
                            BoundaryType::Inlet => { top_boundary.density }
                            BoundaryType::Wall => { density[x + width] }
                        }
                    } else if py >= height as f32 {
                        match bottom_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,
                            BoundaryType::Inlet => { bottom_boundary.density }
                            BoundaryType::Wall => { density[x + (height - 2) * width] }
                        }
                    }
                    else {
                        Self::sample_field_solid_aware(density, solids, width, height, px, py, density[idx])
                    };

                    row[x] = value;
                }
            });

        std::mem::swap(
            &mut self.density,
            &mut self.density_temp,
        );
    }

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

    pub fn advect_velocity(&mut self, dt: f32) {
        self.velocity_x_temp.copy_from_slice(&self.velocity_x);
        self.velocity_y_temp.copy_from_slice(&self.velocity_y);

        let width = self.width;
        let height = self.height;

        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;
        let solids = &self.solids;

        self.velocity_x_temp
            .par_chunks_mut(width)
            .zip(self.velocity_y_temp.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, (row_x, row_y))| {
                if y == 0 || y == height - 1 {
                    return;
                }

                for x in 1..width - 1 {
                    let idx = x + y * width;

                    if solids[idx] {
                        row_x[x] = 0.0;
                        row_y[x] = 0.0;
                        continue;
                    }

                    let vx = velocity_x[idx];
                    let vy = velocity_y[idx];

                    let px = (x as f32 - vx * dt).clamp(0.0, (width - 1) as f32);

                    let py = (y as f32 - vy * dt).clamp(0.0, (height - 1) as f32);

                    row_x[x] = Self::sample_field_solid_aware(velocity_x, solids, width, height, px, py, vx);
                    row_y[x] = Self::sample_field_solid_aware(velocity_y, solids, width, height, px, py, vy);
                }
            });

        std::mem::swap(
            &mut self.velocity_x,
            &mut self.velocity_x_temp,
        );

        std::mem::swap(
            &mut self.velocity_y,
            &mut self.velocity_y_temp,
        );
    }

    fn apply_solid_velocity_boundaries(&mut self) {
        let width = self.width;
        let height = self.height;

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = self.idx(x, y);

                if self.solids[idx] {
                    self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                    continue;
                }

                let solid_left = self.solids[self.idx(x - 1, y)];
                let solid_right = self.solids[self.idx(x + 1, y)];
                let solid_top = self.solids[self.idx(x, y - 1)];
                let solid_bottom = self.solids[self.idx(x, y + 1)];

                if (solid_left && self.velocity_x[idx] < 0.0) || (solid_right && self.velocity_x[idx] > 0.0) {
                    self.velocity_x[idx] = 0.0;
                }

                if (solid_top && self.velocity_y[idx] < 0.0) || (solid_bottom && self.velocity_y[idx] > 0.0) {
                    self.velocity_y[idx] = 0.0;
                }
            }
        }
    }

    fn apply_velocity_boundaries(&mut self) {
        //gauche
        for y in 1..self.height - 1 {
            let idx = self.idx(0, y);

            match self.left_boundary.boundary_type {
                BoundaryType::Wall => {
                    self.velocity_x[idx] = 0.0;
                    //self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    let factor = Self::offset_factor(y, self.height, self.left_boundary.offset);
                    self.velocity_x[idx] = self.left_boundary.velocity_x * factor;
                    self.velocity_y[idx] = self.left_boundary.velocity_y * factor;
                }

                BoundaryType::Outlet => {
                    let inside = self.idx(1, y);
                    self.velocity_x[idx] = self.velocity_x[inside];
                    self.velocity_y[idx] = self.velocity_y[inside];
                }
            }
        }

        //droite
        for y in 1..self.height - 1 {
            let idx = self.idx(self.width - 1, y);

            match self.right_boundary.boundary_type {
                BoundaryType::Wall => {
                    self.velocity_x[idx] = 0.0;
                    //self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    let factor = Self::offset_factor(y, self.height, self.right_boundary.offset);
                    self.velocity_x[idx] = self.right_boundary.velocity_x * factor;
                    self.velocity_y[idx] = self.right_boundary.velocity_y * factor;
                }

                BoundaryType::Outlet => {
                    let inside = self.idx(self.width - 2, y);
                    self.velocity_x[idx] = self.velocity_x[inside];
                    self.velocity_y[idx] = self.velocity_y[inside];
                }
            }
        }

        //haut
        for x in 1..self.width - 1 {
            let idx = self.idx(x, 0);

            match self.top_boundary.boundary_type {
                BoundaryType::Wall => {
                    //self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    let factor = Self::offset_factor(x, self.width, self.top_boundary.offset);
                    self.velocity_x[idx] = self.top_boundary.velocity_x * factor;
                    self.velocity_y[idx] = self.top_boundary.velocity_y * factor;
                }

                BoundaryType::Outlet => {
                    let inside = self.idx(x, 1);
                    self.velocity_x[idx] = self.velocity_x[inside];
                    self.velocity_y[idx] = self.velocity_y[inside];
                }
            }
        }

        //bas
        for x in 1..self.width - 1 {
            let idx = self.idx(x, self.height - 1);

            match self.bottom_boundary.boundary_type {
                BoundaryType::Wall => {
                    //self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    let factor = Self::offset_factor(x, self.width, self.bottom_boundary.offset, );
                    self.velocity_x[idx] = self.bottom_boundary.velocity_x * factor;
                    self.velocity_y[idx] = self.bottom_boundary.velocity_y * factor;
                }

                BoundaryType::Outlet => {
                    let inside = self.idx(x, self.height - 2);
                    self.velocity_x[idx] = self.velocity_x[inside];
                    self.velocity_y[idx] = self.velocity_y[inside];
                }
            }
        }
    }

    fn calculate_divergence(&mut self) {
        let width = self.width;
        let height = self.height;

        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;

        self.divergence
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y == height - 1 {
                    return;
                }

                for x in 1..width - 1 {
                    let idx = x + y * width;
                    if self.solids[idx] {
                        row[x] = 0.0;
                        continue;
                    }

                    let vx_right = if self.solids[idx + 1] { 0.0 } else { velocity_x[idx + 1] };
                    let vx_left = if self.solids[idx - 1] { 0.0 } else { velocity_x[idx - 1] };
                    let vy_down = if self.solids[idx + width] { 0.0 } else { velocity_y[idx + width] };
                    let vy_up = if self.solids[idx - width] { 0.0 } else { velocity_y[idx - width] };

                    let divergence = 0.5 * (vx_right - vx_left + vy_down - vy_up);
                    row[x] = -divergence;
                }
            });
    }

    fn solve_pressure(&mut self, iterations: usize) {
        let width = self.width;
        let height = self.height;
        let omega = self.sor_omega;

        self.apply_pressure_boundaries();

        for _ in 0..iterations {
            for color in 0..2 {
                let pressure = &self.pressure;
                let divergence = &self.divergence;
                let solids = &self.solids;

                self.pressure_temp
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(y, row)| {
                        if y == 0 || y == height - 1 {
                            return;
                        }

                        for x in 1..width - 1 {
                            let idx = x + y * width;

                            if solids[idx] {
                                row[x] = 0.0;
                                continue;
                            }

                            if (x + y) % 2 != color {
                                row[x] = pressure[idx];
                                continue;
                            }

                            let mut sum = divergence[idx];
                            let neighbors = [idx + 1, idx - 1, idx + width, idx - width];

                            for neighbor in neighbors {
                                if solids[neighbor] {
                                    sum += pressure[idx]; // dp/dn = 0
                                } else {
                                    sum += pressure[neighbor];
                                }
                            }

                            let gs_value = sum * 0.25;
                            row[x] = pressure[idx] + omega * (gs_value - pressure[idx]);
                        }
                    });

                std::mem::swap(&mut self.pressure, &mut self.pressure_temp);
                self.apply_pressure_boundaries();
            }
        }
    }

    fn apply_pressure_boundaries(&mut self) {
        let width = self.width;
        let height = self.height;

        // gauche
        for y in 1..height - 1 {
            let boundary = self.idx(0, y);
            let inside = self.idx(1, y);

            if self.solids[boundary] {
                self.pressure[boundary] = 0.0;
            } else {
                self.pressure[boundary] =
                    match self.left_boundary.boundary_type {
                        BoundaryType::Outlet => 0.0,
                        BoundaryType::Wall | BoundaryType::Inlet => { self.pressure[inside] }
                    };
            }
        }

        // droite
        for y in 1..height - 1 {
            let boundary = self.idx(width - 1, y);
            let inside = self.idx(width - 2, y);

            if self.solids[boundary] {
                self.pressure[boundary] = 0.0;
            } else {
                self.pressure[boundary] =
                    match self.right_boundary.boundary_type {
                        BoundaryType::Outlet => 0.0,
                        BoundaryType::Wall | BoundaryType::Inlet => { self.pressure[inside] }
                    };
            }
        }

        // haut
        for x in 1..width - 1 {
            let boundary = self.idx(x, 0);
            let inside = self.idx(x, 1);

            if self.solids[boundary] {
                self.pressure[boundary] = 0.0;
            } else {
                self.pressure[boundary] =
                    match self.top_boundary.boundary_type {
                        BoundaryType::Outlet => 0.0,
                        BoundaryType::Wall | BoundaryType::Inlet => { self.pressure[inside] }
                    };
            }
        }

        // bas
        for x in 1..width - 1 {
            let boundary = self.idx(x, height - 1);
            let inside = self.idx(x, height - 2);

            if self.solids[boundary] {
                self.pressure[boundary] = 0.0;
            } else {
                self.pressure[boundary] =
                    match self.bottom_boundary.boundary_type {
                        BoundaryType::Outlet => 0.0,
                        BoundaryType::Wall | BoundaryType::Inlet => { self.pressure[inside] }
                    };
            }
        }
    }

    fn correct_velocity(&mut self) {
        let width = self.width;
        let height = self.height;

        let pressure = &self.pressure;

        self.velocity_x
            .par_chunks_mut(width)
            .zip(self.velocity_y.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, (row_x, row_y))| {
                if y == 0 || y == height - 1 {
                    return;
                }

                for x in 1..width - 1 {
                    let idx = x + y * width;
                    if self.solids[idx] {
                        row_x[x] = 0.0;
                        row_y[x] = 0.0;
                        continue;
                    }

                    let pressure_right = if self.solids[idx + 1] { pressure[idx] } else { pressure[idx + 1] };
                    let pressure_left = if self.solids[idx - 1] { pressure[idx] } else { pressure[idx - 1] };
                    let pressure_down = if self.solids[idx + width] { pressure[idx] } else { pressure[idx + width] };
                    let pressure_up = if self.solids[idx - width] { pressure[idx] } else { pressure[idx - width] };

                    let pressure_gradient_x = 0.5 * (pressure_right - pressure_left);
                    let pressure_gradient_y = 0.5 * (pressure_down - pressure_up);

                    row_x[x] -= pressure_gradient_x;
                    row_y[x] -= pressure_gradient_y;
                }
            });
    }

    fn apply_external_forces(&mut self, dt: f32) {
        let width = self.width;
        let height = self.height;
        let gravity_x = self.gravity_x;
        let gravity_y = self.gravity_y;
        let damping = (1.0 - self.friction * dt).max(0.0);
        let solids = &self.solids;

        self.velocity_x
            .par_chunks_mut(width)
            .zip(self.velocity_y.par_chunks_mut(width))
            .enumerate()
            .for_each(|(y, (row_x, row_y))| {
                if y == 0 || y == height - 1 {
                    return;
                }

                for x in 1..width - 1 {
                    let idx = x + y * width;
                    if solids[idx] {
                        continue;
                    }

                    row_x[x] = (row_x[x] + gravity_x * dt) * damping;
                    row_y[x] = (row_y[x] + gravity_y * dt) * damping;
                }
            });
    }

    pub fn enforce_incompressibility(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        self.correct_velocity();
    }

    pub fn step(&mut self, dt: f32, pressure_iterations: usize, diffusion_iterations: usize) {
        self.apply_external_forces(dt);
        self.advect_velocity(dt); //advection vitesse (v)
        self.apply_velocity_boundaries(); //attention aux bordures (v)

        self.apply_solid_velocity_boundaries(); //attentions aux solides (v)

        self.enforce_incompressibility(pressure_iterations); //divergence et pression -> vitesse
        self.apply_velocity_boundaries(); //attention aux bordures (v)

        self.apply_solid_velocity_boundaries(); //attentions aux solides (v)

        self.diffuse_density(dt, diffusion_iterations); //diffuser la densité (d)
        self.advect_density(dt); //advection de la densité (d)
        self.apply_density_boundaries(); //attention aux bordures (d)
    }
}