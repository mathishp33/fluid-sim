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
}

impl Boundary {
    pub fn wall() -> Self {
        Self {
            boundary_type: BoundaryType::Wall,
            velocity_x: 0.0,
            velocity_y: 0.0,
            density: 0.0,
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

    density_temp: Vec<f32>,
    velocity_x_temp: Vec<f32>,
    velocity_y_temp: Vec<f32>,
    pressure_temp: Vec<f32>,

    pub left_boundary: Boundary,
    pub right_boundary: Boundary,
    pub top_boundary: Boundary,
    pub bottom_boundary: Boundary,
}

impl FluidSim {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    pub fn new(width: usize, height: usize, start_density: f32, diffusion_rate: f32) -> Self {
        let size = width * height;
        FluidSim {
            width,
            height,

            density: vec![start_density; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            diffusion_rate,

            pressure: vec![0.0; size],
            divergence: vec![0.0; size],
            density_temp: vec![0.0; size],
            velocity_x_temp: vec![0.0; size],
            velocity_y_temp: vec![0.0; size],
            pressure_temp: vec![0.0; size],

            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),
        }
    }

    pub fn randomize_density_smoothed(&mut self, seed_count: usize) {
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
    }

    pub fn diffuse_density(&mut self, dt: f32, diffusion_iters: usize) {
        let a = self.diffusion_rate * dt;

        for _ in 0..diffusion_iters {
            // Apply diffusion using swap buffer - split interior and boundary
            // Interior cells use diffusion formula
            let width = self.width;

            self.density_temp
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    if y == 0 || y == self.height - 1 {
                        return;
                    }

                    for x in 1..width - 1 {
                        let idx = x + y * width;

                        row[x] = (self.density[idx] + a * (self.density[idx + 1] + self.density[idx - 1] +
                                    self.density[idx + width] + self.density[idx - width])) / (1.0 + 4.0 * a);
                    }
                });
            
            // Copy boundaries (Neumann boundary condition)
            for y in 0..self.height {
                let idx_left = self.idx(0, y);
                let idx_right = self.idx(self.width - 1, y);
                self.density_temp[idx_left] = self.density[idx_left];
                self.density_temp[idx_right] = self.density[idx_right];
            }
            for x in 0..self.width {
                let idx_top = self.idx(x, 0);
                let idx_bottom = self.idx(x, self.height - 1);
                self.density_temp[idx_top] = self.density[idx_top];
                self.density_temp[idx_bottom] = self.density[idx_bottom];
            }
            
            std::mem::swap(&mut self.density, &mut self.density_temp);
        }
    }

    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    fn sample_field(field: &Vec<f32>, width: usize, height: usize, x: f32, y: f32) -> f32 {
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

    fn sample_density(&self, x: f32, y: f32) -> f32 {
        if x < 0.0 {
            return match self.left_boundary.boundary_type {
                BoundaryType::Outlet => 0.0,
                BoundaryType::Inlet => self.left_boundary.density,
                BoundaryType::Wall => self.density[self.idx(0, y.clamp(0.0, (self.height - 1) as f32) as usize)],
            };
        }

        if x >= self.width as f32 {
            return match self.right_boundary.boundary_type {
                BoundaryType::Outlet => 0.0,
                BoundaryType::Inlet => self.right_boundary.density,
                BoundaryType::Wall => self.density[self.idx(self.width - 1,
                    y.clamp(0.0, (self.height - 1) as f32) as usize,
                )],
            };
        }

        if y < 0.0 {
            return match self.top_boundary.boundary_type {
                BoundaryType::Outlet => 0.0,
                BoundaryType::Inlet => self.top_boundary.density,
                BoundaryType::Wall => self.density[self.idx(
                    x.clamp(0.0, (self.width - 1) as f32) as usize, 0,
                )],
            };
        }

        if y >= self.height as f32 {
            return match self.bottom_boundary.boundary_type {
                BoundaryType::Outlet => 0.0,
                BoundaryType::Inlet => self.bottom_boundary.density,
                BoundaryType::Wall =>
                    self.density[self.idx(x.clamp(0.0, (self.width - 1) as f32) as usize,
                                          self.height - 1,
                )],
            };
        }

        Self::sample_field(&self.density, self.width, self.height, x, y)
    }

    pub fn advect_density(&mut self, dt: f32) {
        self.density_temp.copy_from_slice(&self.density);

        let width = self.width;
        let height = self.height;

        let density = &self.density;
        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;

        // Copie des conditions aux limites pour éviter d'emprunter `self`
        // dans la closure Rayon.
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

                    let vx = velocity_x[idx];
                    let vy = velocity_y[idx];

                    let px = x as f32 - vx * dt;
                    let py = y as f32 - vy * dt;

                    let value = if px < 0.0 {
                        match left_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,

                            BoundaryType::Inlet => {
                                left_boundary.density
                            }

                            BoundaryType::Wall => {
                                density[1 + y * width]
                            }
                        }
                    } else if px >= width as f32 {
                        match right_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,

                            BoundaryType::Inlet => {
                                right_boundary.density
                            }

                            BoundaryType::Wall => {
                                density[(width - 2) + y * width]
                            }
                        }
                    } else if py < 0.0 {
                        match top_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,

                            BoundaryType::Inlet => {
                                top_boundary.density
                            }

                            BoundaryType::Wall => {
                                density[x + width]
                            }
                        }
                    } else if py >= height as f32 {
                        match bottom_boundary.boundary_type {
                            BoundaryType::Outlet => 0.0,

                            BoundaryType::Inlet => {
                                bottom_boundary.density
                            }

                            BoundaryType::Wall => {
                                density[x + (height - 2) * width]
                            }
                        }
                    }
                    else {
                        Self::sample_field(density, width, height, px, py)
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
                BoundaryType::Wall | BoundaryType::Outlet => {
                    self.density[idx] = self.density[self.idx(1, y)];
                }

                BoundaryType::Inlet => {
                    self.density[idx] = self.left_boundary.density;
                }
            }
        }

        //droite
        for y in 1..self.height - 1 {
            let idx = self.idx(self.width - 1, y);

            match self.right_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => {
                    self.density[idx] = self.density[self.idx(self.width - 2, y)];
                }

                BoundaryType::Inlet => {
                    self.density[idx] = self.right_boundary.density;
                }
            }
        }

        //haut
        for x in 1..self.width - 1 {
            let idx = self.idx(x, 0);

            match self.top_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => {
                    self.density[idx] = self.density[self.idx(x, 1)];
                }

                BoundaryType::Inlet => {
                    self.density[idx] = self.top_boundary.density;
                }
            }
        }

        //bas
        for x in 1..self.width - 1 {
            let idx = self.idx(x, self.height - 1);

            match self.bottom_boundary.boundary_type {
                BoundaryType::Wall | BoundaryType::Outlet => {
                    self.density[idx] = self.density[self.idx(x, self.height - 2)];
                }

                BoundaryType::Inlet => {
                    self.density[idx] = self.bottom_boundary.density;
                }
            }
        }
    }

    pub fn advect_velocity(&mut self, dt: f32) {
        // Copy current velocity to temp buffers
        self.velocity_x_temp.copy_from_slice(&self.velocity_x);
        self.velocity_y_temp.copy_from_slice(&self.velocity_y);

        let width = self.width;
        let height = self.height;

        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;

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

                    let vx = velocity_x[idx];
                    let vy = velocity_y[idx];
                    let px = x as f32 - vx * dt;
                    let py = y as f32 - vy * dt;

                    row_x[x] = Self::sample_field(velocity_x, width, height, px, py);

                    row_y[x] = Self::sample_field(velocity_y, width, height, px, py);
                }
            });

        std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
        std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);

    }

    fn apply_velocity_boundaries(&mut self) {
        //gauche
        for y in 1..self.height - 1 {
            let idx = self.idx(0, y);

            match self.left_boundary.boundary_type {
                BoundaryType::Wall => {
                    self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    self.velocity_x[idx] = self.left_boundary.velocity_x;
                    self.velocity_y[idx] = self.left_boundary.velocity_y;
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
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    self.velocity_x[idx] = self.right_boundary.velocity_x;
                    self.velocity_y[idx] = self.right_boundary.velocity_y;
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
                    self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    self.velocity_x[idx] = self.top_boundary.velocity_x;
                    self.velocity_y[idx] = self.top_boundary.velocity_y;
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
                    self.velocity_x[idx] = 0.0;
                    self.velocity_y[idx] = 0.0;
                }

                BoundaryType::Inlet => {
                    self.velocity_x[idx] = self.bottom_boundary.velocity_x;
                    self.velocity_y[idx] = self.bottom_boundary.velocity_y;
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

                    let divergence = 0.5 * (
                        velocity_x[idx + 1] - velocity_x[idx - 1] + velocity_y[idx + width] - velocity_y[idx - width]
                    );

                    row[x] = -divergence;
                }
            });
    }

    fn solve_pressure(&mut self, iterations: usize) {
        self.pressure.fill(0.0);

        let width = self.width;
        let height = self.height;

        for _ in 0..iterations {
            let pressure = &self.pressure;
            let divergence = &self.divergence;

            self.pressure_temp
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(y, row)| {
                    if y == 0 || y == height - 1 {
                        return;
                    }

                    for x in 1..width - 1 {
                        let idx = x + y * width;
                        let neighbors = pressure[idx + 1] + pressure[idx - 1] + pressure[idx + width] + pressure[idx - width];
                        row[x] = (neighbors + divergence[idx]) * 0.25;
                    }
                });

            std::mem::swap(&mut self.pressure, &mut self.pressure_temp);
            self.apply_pressure_boundaries();
        }
    }

    fn apply_pressure_boundaries(&mut self) {
        let width = self.width;
        let height = self.height;

        for y in 1..height - 1 {
            let l0 = self.idx(0, y);
            self.pressure[l0] =
                match self.left_boundary.boundary_type {
                    BoundaryType::Outlet => 0.0,
                    BoundaryType::Wall | BoundaryType::Inlet => {
                        self.pressure[self.idx(1, y)]
                    }
                };

            let l1 = self.idx(width - 1, y);
            self.pressure[l1] =
                match self.right_boundary.boundary_type {
                    BoundaryType::Outlet => 0.0,
                    BoundaryType::Wall | BoundaryType::Inlet => {
                        self.pressure[self.idx(width - 2, y)]
                    }
                };
        }


        for x in 1..width - 1 {
            let l2 = self.idx(x, 0);
            self.pressure[l2] =
                match self.top_boundary.boundary_type {
                    BoundaryType::Outlet => 0.0,
                    BoundaryType::Wall | BoundaryType::Inlet => {
                        self.pressure[self.idx(x, 1)]
                    }
                };

            let l3 = self.idx(x, height - 1);
            self.pressure[l3] =
                match self.bottom_boundary.boundary_type {
                    BoundaryType::Outlet => 0.0,
                    BoundaryType::Wall | BoundaryType::Inlet => {
                        self.pressure[self.idx(x, height - 2)]
                    }
                };
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

                    let pressure_gradient_x = 0.5 * (pressure[idx + 1] - pressure[idx - 1]);
                    let pressure_gradient_y = 0.5 * (pressure[idx + width] - pressure[idx - width]);

                    row_x[x] -= pressure_gradient_x;
                    row_y[x] -= pressure_gradient_y;
                }
            });
    }

    pub fn enforce_incompressibility(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        self.correct_velocity();
    }


    pub fn step(&mut self, dt: f32, pressure_iterations: usize, diffusion_iterations: usize) {
        self.advect_velocity(dt);
        self.apply_velocity_boundaries();

        self.enforce_incompressibility(pressure_iterations);
        self.apply_velocity_boundaries();

        self.diffuse_density(dt, diffusion_iterations);
        self.advect_density(dt);
        self.apply_density_boundaries();
    }
}