use ndarray::Array2;

pub struct Fluid {
    pub width: usize,
    pub height: usize,
    pub density: Array2<f64>,
    pub velocity_x: Array2<f64>,
    pub velocity_y: Array2<f64>,
    pub diffusion_rate: f64,
    pub pressure: Array2<f64>,
    pub divergence: Array2<f64>,
}

impl Fluid {
    pub fn new(width: usize, height: usize, start_density: f64, diffusion_rate: f64) -> Self {
        Fluid {
            width,
            height,
            density: Array2::from_elem((width, height), start_density),
            velocity_x: Array2::zeros((width, height)),
            velocity_y: Array2::zeros((width, height)),
            diffusion_rate,
            pressure: Array2::zeros((width, height)),
            divergence: Array2::zeros((width, height)),
        }
    }

    pub fn get_density(&self, x: usize, y: usize) -> f64 {
        self.density[(x, y)]
    }

    pub fn diffuse_density(&mut self, dt: f64, iterations: usize) {
        let a = self.diffusion_rate * dt;

        for _ in 0..iterations {
            for x in 1..self.width - 1 {
                for y in 1..self.height - 1 {
                    self.density[(x, y)] = (
                        self.density[(x, y)] +
                        a * (
                            self.density[(x + 1, y)] +
                            self.density[(x - 1, y)] +
                            self.density[(x, y + 1)] +
                            self.density[(x, y - 1)]
                        )
                    ) / (1.0 + 4.0 * a);
                }
            }
        }
    }

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    fn sample_field(field: &Array2<f64>, x: f64, y: f64) -> f64 {
        let w = field.shape()[0] as isize;
        let h = field.shape()[1] as isize;

        let x0 = x.floor().clamp(0.0, (w - 1) as f64) as isize;
        let y0 = y.floor().clamp(0.0, (h - 1) as f64) as isize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);

        let sx = x - x0 as f64;
        let sy = y - y0 as f64;

        let v00 = field[(x0 as usize, y0 as usize)];
        let v10 = field[(x1 as usize, y0 as usize)];
        let v01 = field[(x0 as usize, y1 as usize)];
        let v11 = field[(x1 as usize, y1 as usize)];

        let a = Self::lerp(v00, v10, sx);
        let b = Self::lerp(v01, v11, sx);
        Self::lerp(a, b, sy)
    }

    pub fn advect_density(&mut self, dt: f64) {
        let mut new_density = self.density.clone();

        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                let vx = self.velocity_x[(x, y)];
                let vy = self.velocity_y[(x, y)];

                let px = x as f64 - vx * dt;
                let py = y as f64 - vy * dt;

                new_density[(x, y)] = Self::sample_field(&self.density, px, py);
            }
        }

        self.density = new_density;
    }

    pub fn advect_velocity(&mut self, dt: f64) {
        let mut new_vx = self.velocity_x.clone();
        let mut new_vy = self.velocity_y.clone();

        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                let vx = self.velocity_x[(x, y)];
                let vy = self.velocity_y[(x, y)];

                let px = x as f64 - vx * dt;
                let py = y as f64 - vy * dt;

                new_vx[(x, y)] = Self::sample_field(&self.velocity_x, px, py);
                new_vy[(x, y)] = Self::sample_field(&self.velocity_y, px, py);
            }
        }

        self.velocity_x = new_vx;
        self.velocity_y = new_vy;
    }

    /// Calculate velocity divergence at each grid cell
    fn calculate_divergence(&mut self) {
        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                let divergence = 0.5 * (
                    self.velocity_x[(x + 1, y)] - self.velocity_x[(x - 1, y)] +
                    self.velocity_y[(x, y + 1)] - self.velocity_y[(x, y - 1)]
                );
                self.divergence[(x, y)] = -divergence;
            }
        }
    }

    fn solve_pressure(&mut self, iterations: usize) {
        self.pressure = Array2::zeros((self.width, self.height));
        
        for _ in 0..iterations {
            for x in 1..self.width - 1 {
                for y in 1..self.height - 1 {
                    let neighbors = self.pressure[(x + 1, y)] +
                                   self.pressure[(x - 1, y)] +
                                   self.pressure[(x, y + 1)] +
                                   self.pressure[(x, y - 1)];
                    self.pressure[(x, y)] = (neighbors + self.divergence[(x, y)]) / 4.0;
                }
            }

            for y in 0..self.height {
                self.pressure[(0, y)] = 0.0;
                self.pressure[(self.width - 1, y)] = 0.0;
            }
            for x in 0..self.width {
                self.pressure[(x, 0)] = 0.0;
                self.pressure[(x, self.height - 1)] = 0.0;
            }
        }
    }

    fn correct_velocity(&mut self) {
        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                let pressure_gradient_x = 0.5 * (self.pressure[(x + 1, y)] - self.pressure[(x - 1, y)]);
                let pressure_gradient_y = 0.5 * (self.pressure[(x, y + 1)] - self.pressure[(x, y - 1)]);
                
                self.velocity_x[(x, y)] -= pressure_gradient_x;
                self.velocity_y[(x, y)] -= pressure_gradient_y;
            }
        }
    }

    pub fn enforce_incompressibility(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        self.correct_velocity();
    }


    pub fn step(&mut self, dt: f64) {
        self.advect_velocity(dt);
        self.enforce_incompressibility(20);

        self.diffuse_density(dt, 10);
        self.advect_density(dt);
    }
}