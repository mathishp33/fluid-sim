use minifb::{Window, WindowOptions};
use ndarray::Array2;

fn main() {
    let width = 800;
    let height = 600;
    const FPS: i32 = 120;
    let mut last_update = std::time::Instant::now();
    let particle_radius = 5;
    let precision = 5;
    let mut window = Window::new(
        "Fluid Simulation", 
        width as usize,
        height as usize, 
        WindowOptions {
            resize: false,
            ..WindowOptions::default()
        })
        .unwrap_or_else(|e| {
            panic!("Unable to create window: {}", e);
        });

    let mut fluid = Fluid::new((width / precision) as usize, (height / precision) as usize);
    fluid.density.fill(0.0);
    fluid.velocity_x.fill(0.0);
    fluid.velocity_y.fill(0.0);

    let mut mouse_pos0 = (0, 0);

    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let now = std::time::Instant::now();

        if now.duration_since(last_update) >= std::time::Duration::from_secs_f64(1.0f64 / FPS as f64) {
            let mut buffer: Vec<u32> = vec![0; width * height];
            last_update = now;
            
            window.update();
            let (mouse_x, mouse_y) = window.get_mouse_pos(minifb::MouseMode::Clamp).unwrap_or((0.0, 0.0));
            let mouse_x = mouse_x as usize;
            let mouse_y = mouse_y as usize;
            if window.get_mouse_down(minifb::MouseButton::Left) {
                for dx in -(particle_radius as isize)..=(particle_radius as isize) {
                    for dy in -(particle_radius as isize)..=(particle_radius as isize) {
                        let nx = (mouse_x as isize + dx) as usize;
                        let ny = (mouse_y as isize + dy) as usize;

                        if nx < width && ny < height {
                            let distance_squared = (dx * dx + dy * dy) as f32;
                            if distance_squared < (particle_radius * particle_radius) as f32 {
                                fluid.density[(nx / precision, ny / precision)] = 1.0;
                                fluid.velocity_x[(nx / precision, ny / precision)] = 0.0;
                                fluid.velocity_y[(nx / precision, ny / precision)] = 0.0;
                            }
                        }
                    }
                }
            }
            if window.get_mouse_down(minifb::MouseButton::Right) {
                for dx in -(particle_radius as isize)..=(particle_radius as isize) {
                    for dy in -(particle_radius as isize)..=(particle_radius as isize) {
                        let nx = (mouse_x as isize + dx) as usize;
                        let ny = (mouse_y as isize + dy) as usize;

                        if nx < width && ny < height {
                            let distance_squared = (dx * dx + dy * dy) as f32;
                            if distance_squared < (particle_radius * particle_radius) as f32 {
                                fluid.velocity_x[(nx / precision, ny / precision)] = (mouse_x as f64 - mouse_pos0.0 as f64) * 0.5f64;
                                fluid.velocity_y[(nx / precision, ny / precision)] = (mouse_y as f64 - mouse_pos0.1 as f64) * 0.5f64;
                            }
                        }
                    }
                }
                mouse_pos0 = (mouse_x, mouse_y);
            } else {
                mouse_pos0 = (mouse_x, mouse_y);
            }
            

            fluid.diffusion();
            fluid.advection();

            for y in 0..fluid.height {
                for x in 0..fluid.width {
                    let color = (fluid.get_density(x, y) * 255.0f64) as i32;
                    let u32_color = ((color as u32) << 16) | ((color as u32) << 8) | (color as u32);
                    for i in 0..precision {
                        for j in 0..precision {
                            let buffer_index = ((y * precision + j) * width + (x * precision + i)) as usize;
                            buffer[buffer_index] = u32_color; 
                        }
                    }
                }
            }
            window.update_with_buffer(&buffer, width, height).unwrap();
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

struct Fluid {
    width: usize,
    height: usize,
    density: Array2<f64>,
    velocity_x: Array2<f64>,
    velocity_y: Array2<f64>,
}

impl Fluid {
    fn new(width: usize, height: usize) -> Self {
        Fluid {
            width,
            height,
            density: Array2::zeros((width, height)),
            velocity_x: Array2::zeros((width, height)),
            velocity_y: Array2::zeros((width, height)),
        }

    }

    fn get_density(&mut self, x: usize, y: usize) -> f64 {
        return self.density[(x, y)];
    }

    fn diffusion(&mut self) {
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

    fn lerp(&mut self, a: f64, b: f64, k: f64) -> f64 {
        a + k * (b - a)
    }

    fn advection(&mut self) {
        let k = 0.5; //friction rate
        let mut new_density = self.density.clone();
        let mut new_velocity_x = self.velocity_x.clone();
        let mut new_velocity_y = self.velocity_y.clone();
        for x in 0..self.width {
            for y in 0..self.height {
                let x1 = x as f64 + self.velocity_x[(x, y)].round();
                let y1 = y as f64 + self.velocity_y[(x, y)].round();
                if x1 < 0.0 || y1 < 0.0 || x1 >= self.width as f64 - 1.0 || y1 >= self.height as f64 - 1.0 {
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

                new_density[(x1 as usize, y1 as usize)] = self.lerp(z1, z2, j.1);
                new_velocity_x[(x1 as usize, y1 as usize)] = self.velocity_x[(x, y)] * k;
                new_velocity_y[(x1 as usize, y1 as usize)] = self.velocity_y[(x, y)] * k;
            }
        }
        self.density = new_density;
        self.velocity_x = new_velocity_x;
        self.velocity_y = new_velocity_y;
    }
}