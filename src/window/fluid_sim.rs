use ndarray::Array2;

pub struct Fluid {
    pub width: usize,
    pub height: usize,
    pub density: Array2<f64>,
    pub velocity_x: Array2<f64>,
    pub velocity_y: Array2<f64>,
}

impl Fluid {
    pub fn new(width: usize, height: usize, start_density: f64) -> Self {
        Fluid {
            width,
            height,
            density: Array2::from_elem((width, height), start_density),
            velocity_x: Array2::zeros((width, height)),
            velocity_y: Array2::zeros((width, height)),
        }

    }

    pub fn get_density(&mut self, x: usize, y: usize) -> f64 {
        return self.density[(x, y)];
    }

    pub fn diffusion(&mut self) {
        let k = 0.1; // diffusion rate
        let mut new_density = self.density.clone();
        for x in 1..self.width - 1 {
            for y in 1..self.height - 1 {
                new_density[(x, y)] += k * (
                    self.density[(x + 1, y)] +
                    self.density[(x - 1, y)] +
                    self.density[(x, y + 1)] +
                    self.density[(x, y - 1)] -
                    4.0 * self.density[(x, y)]
                );
            }
        }
        self.density = new_density;
    }

    pub fn lerp(&mut self, a: f64, b: f64, k: f64) -> f64 {
        a + k * (b - a)
    }

    pub fn advection(&mut self) {
        let k = 0.5; //friction rate
        let mut new_density = self.density.clone();
        let mut new_velocity_x = self.velocity_x.clone();
        let mut new_velocity_y = self.velocity_y.clone();
        for x in 0..self.width {
            for y in 0..self.height {
                if self.velocity_x[(x, y)] == 0.0 && self.velocity_y[(x, y)] == 0.0 {
                    continue;
                }
                if self.density[(x, y)] <= 0.0 {
                    new_density[(x, y)] = 0.0;
                    new_velocity_x[(x, y)] = 0.0;
                    new_velocity_y[(x, y)] = 0.0;
                    continue;
                }
                let x1 = x as f64 + self.velocity_x[(x, y)].round();
                let y1 = y as f64 + self.velocity_y[(x, y)].round();
                if x1 <= 0.0 || y1 <= 0.0 || x1 >= self.width as f64 - 1.0 || y1 >= self.height as f64 - 1.0 {
                    continue;
                }
                let f = (x1 - self.velocity_x[(x, y)], y1 - self.velocity_y[(x, y)]);
                let i = (f.0.floor(), f.1.floor());
                let j = (f.0.fract(), f.1.fract());
                if i.0 < 0.0 || i.1 < 0.0 || i.0 >= self.width as f64 - 1.0 || i.1 >= self.height as f64 - 1.0 {
                    continue;
                }
                let z1 = self.lerp(self.density[(i.0 as usize, i.1 as usize)], self.density[(i.0 as usize + 1, i.1 as usize)], j.0);
                let z2 = self.lerp(self.density[(i.0 as usize, i.1 as usize + 1)], self.density[(i.0 as usize + 1, i.1 as usize + 1)], j.0);

                let mut density_to_give = self.lerp(z1, z2, j.1);
                if density_to_give > self.density[(x, y)] {
                    density_to_give = self.density[(x, y)];
                }
                new_density[(x1 as usize, y1 as usize)] += density_to_give;
                new_density[(x, y)] -= density_to_give; //enlever si on veux un truc + beau
                new_velocity_x[(x, y)] = 0.0;
                new_velocity_y[(x, y)] = 0.0;
                new_velocity_x[(x1 as usize, y1 as usize)] = self.velocity_x[(x, y)] * k;
                new_velocity_y[(x1 as usize, y1 as usize)] = self.velocity_y[(x, y)] * k;
            }
        }
        self.density = new_density;
        self.velocity_x = new_velocity_x;
        self.velocity_y = new_velocity_y;
    }
}