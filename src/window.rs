use minifb::{Window, WindowOptions};

mod fluid_sim;


pub struct FluidWindow {
    pub width: usize,
    pub height: usize,
    pub particle_radius: usize,
    pub precision : usize,
    pub window: Window,
    pub start_density: f64,
    pub diffusion_rate: f64,
    pub max_color: u32,
    pub randomize: bool,
    pub random_smoothing: usize,
}

impl FluidWindow {
    pub fn new(width: usize, height: usize, particle_radius: usize, precision: usize, start_density: f64, diffusion_rate: f64, max_color: u32, randomize: bool, random_smoothing: usize) -> Self {
        FluidWindow {
            width,
            height,
            particle_radius,
            precision,
            window: Window::new(
                "Fluid Simulation", 
                width,
                height,
                WindowOptions {
                    resize: false,
                    ..WindowOptions::default()
                })
                .unwrap_or_else(|e| {
                    panic!("Unable to create window: {}", e);
                }),
            start_density,
            diffusion_rate,
            max_color,
            randomize,
            random_smoothing,
        }
    }

    pub fn run(&mut self) {
        let mut fluid = fluid_sim::Fluid::new(
            (self.width / self.precision) as usize,
            (self.height / self.precision) as usize,
            self.start_density,
            self.diffusion_rate,
        );

        if self.randomize {
            fluid.randomize_density_smoothed(self.random_smoothing);
        }

        let mut last_mouse = (0usize, 0usize);
        let mut last_time = std::time::Instant::now();

        while self.window.is_open() && !self.window.is_key_down(minifb::Key::Escape) {
            let now = std::time::Instant::now();
            let dt = (now - last_time).as_secs_f64();
            last_time = now;

            if dt <= 0.0 {
                continue;
            }

            let mut buffer = vec![0u32; self.width * self.height];

            let (mx, my) = self
                .window
                .get_mouse_pos(minifb::MouseMode::Clamp)
                .unwrap_or((0.0, 0.0));

            let mx = mx as usize;
            let my = my as usize;

            let fx = (mx as f64 - last_mouse.0 as f64) / dt;
            let fy = (my as f64 - last_mouse.1 as f64) / dt;

            let gx = mx / self.precision;
            let gy = my / self.precision;

            if gx > 1 && gx < fluid.width - 1 && gy > 1 && gy < fluid.height - 1 {
                let r = self.particle_radius / self.precision.max(1);

                for dx in -(r as isize)..=(r as isize) {
                    for dy in -(r as isize)..=(r as isize) {
                        let x = gx as isize + dx;
                        let y = gy as isize + dy;

                        if x <= 0 || y <= 0 ||
                        x >= fluid.width as isize - 1 ||
                        y >= fluid.height as isize - 1 {
                            continue;
                        }

                        let x = x as usize;
                        let y = y as usize;

                        if self.window.get_mouse_down(minifb::MouseButton::Left) {
                            fluid.density[(x, y)] += 2.0 * dt;
                        }

                        if self.window.get_mouse_down(minifb::MouseButton::Right) {
                            fluid.velocity_x[(x, y)] += fx * 0.05;
                            fluid.velocity_y[(x, y)] += fy * 0.05;
                        }
                    }
                }
            }

            last_mouse = (mx, my);

            fluid.step(dt.min(0.05)); // clamp dt for stability


            for y in 0..fluid.height {
                for x in 0..fluid.width {
                    let d = fluid.get_density(x, y).clamp(0.0, 1.0);

                    let r = (d * ((self.max_color >> 0) & 0xFF) as f64) as u8;
                    let g = (d * ((self.max_color >> 8) & 0xFF) as f64) as u8;
                    let b = (d * ((self.max_color >> 16) & 0xFF) as f64) as u8;

                    let color =
                        ((b as u32) << 16) |
                        ((g as u32) << 8) |
                        (r as u32);

                    for py in 0..self.precision {
                        for px in 0..self.precision {
                            let i = (y * self.precision + py) * self.width
                                + (x * self.precision + px);
                            buffer[i] = color;
                        }
                    }
                }
            }

            self.window
                .update_with_buffer(&buffer, self.width, self.height)
                .unwrap();
        }
    }
}

