use rand::Rng;
use rayon::prelude::*;

pub struct FluidSim {
    pub width: usize,
    pub height: usize,
    pub density: Vec<f64>,
    pub velocity_x: Vec<f64>,
    pub velocity_y: Vec<f64>,
    pub diffusion_rate: f64,
    pub pressure: Vec<f64>,
    pub divergence: Vec<f64>,

    density_temp: Vec<f64>,
    velocity_x_temp: Vec<f64>,
    velocity_y_temp: Vec<f64>,
}

impl FluidSim {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    pub fn new(width: usize, height: usize, start_density: f64, diffusion_rate: f64) -> Self {
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

    pub fn get_density(&self, x: usize, y: usize) -> f64 {
        self.density[self.idx(x, y)]
    }

    pub fn diffuse_density(&mut self, dt: f64, diffusion_iters: usize) {
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

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    fn sample_field(field: &Vec<f64>, width: usize, height: usize, x: f64, y: f64) -> f64 {
        let w = width as isize;
        let h = height as isize;

        let x0 = x.floor().clamp(0.0, (w - 1) as f64) as isize;
        let y0 = y.floor().clamp(0.0, (h - 1) as f64) as isize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);

        let sx = x - x0 as f64;
        let sy = y - y0 as f64;

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

    pub fn advect_density(&mut self, dt: f64) {
        // Copy current density to temp buffer
        self.density_temp.copy_from_slice(&self.density);

        let width = self.width;
        let height = self.height;

        let density = &self.density;
        let velocity_x = &self.velocity_x;
        let velocity_y = &self.velocity_y;

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
                    let px = x as f64 - vx * dt;
                    let py = y as f64 - vy * dt;

                    row[x] = Self::sample_field(density, width, height, px, py);
                }
            });

        std::mem::swap(&mut self.density, &mut self.density_temp);
        
        // Clear boundaries
        for y in 0..self.height {
            let idx_left = self.idx(0, y);
            let idx_right = self.idx(self.width - 1, y);
            self.density[idx_left] = 0.0;
            self.density[idx_right] = 0.0;
        }
        for x in 0..self.width {
            let idx_top = self.idx(x, 0);
            let idx_bottom = self.idx(x, self.height - 1);
            self.density[idx_top] = 0.0;
            self.density[idx_bottom] = 0.0;
        }
    }

    pub fn advect_velocity(&mut self, dt: f64) {
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
                    let px = x as f64 - vx * dt;
                    let py = y as f64 - vy * dt;

                    row_x[x] = Self::sample_field(velocity_x, width, height, px, py);

                    row_y[x] = Self::sample_field(velocity_y, width, height, px, py);
                }
            });

        std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
        std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
        
        for y in 0..self.height {
            let idx_left = self.idx(0, y);
            let idx_right = self.idx(self.width - 1, y);
            self.velocity_x[idx_left] = 0.0;
            self.velocity_x[idx_right] = 0.0;
            self.velocity_y[idx_left] = 0.0;
            self.velocity_y[idx_right] = 0.0;
        }
        for x in 0..self.width {
            let idx_top = self.idx(x, 0);
            let idx_bottom = self.idx(x, self.height - 1);
            self.velocity_x[idx_top] = 0.0;
            self.velocity_x[idx_bottom] = 0.0;
            self.velocity_y[idx_top] = 0.0;
            self.velocity_y[idx_bottom] = 0.0;
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
        
        for _ in 0..iterations {
            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let idx = self.idx(x, y);
                    let idx_right = self.idx(x + 1, y);
                    let idx_left = self.idx(x - 1, y);
                    let idx_up = self.idx(x, y + 1);
                    let idx_down = self.idx(x, y - 1);
                    
                    let neighbors = self.pressure[idx_right] + self.pressure[idx_left] +
                                   self.pressure[idx_up] + self.pressure[idx_down];
                    self.pressure[idx] = (neighbors + self.divergence[idx]) / 4.0;
                }
            }

            // Clear boundaries
            for y in 0..self.height {
                let idx_left = self.idx(0, y);
                let idx_right = self.idx(self.width - 1, y);
                self.pressure[idx_left] = 0.0;
                self.pressure[idx_right] = 0.0;
            }
            for x in 0..self.width {
                let idx_top = self.idx(x, 0);
                let idx_bottom = self.idx(x, self.height - 1);
                self.pressure[idx_top] = 0.0;
                self.pressure[idx_bottom] = 0.0;
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


    pub fn step(&mut self, dt: f64, pressure_iterations: usize, diffusion_iterations: usize) {
        self.advect_velocity(dt);
        self.enforce_incompressibility(pressure_iterations);

        self.diffuse_density(dt, diffusion_iterations);
        self.advect_density(dt);
    }
}