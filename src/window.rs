use minifb::{Window, WindowOptions};

use crate::simulation::fluid_sim;


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
    pub pressure_iters: usize,
    pub diffusion_iters: usize,
    buffer: Vec<u32>,
    fps: f64,
    frame_count: usize,
    last_fps_update: std::time::Instant,
    paused: bool,
    step_frame: usize,
}

impl FluidWindow {
    pub fn new(width: usize, height: usize, particle_radius: usize, precision: usize, start_density: f64, diffusion_rate: f64, max_color: u32, randomize: bool, 
        random_smoothing: usize, pressure_iters: usize, diffusion_iters: usize) -> Self {
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
            pressure_iters,
            diffusion_iters,
            buffer: vec![0u32; width * height],
            fps: 0.0,
            frame_count: 0,
            last_fps_update: std::time::Instant::now(),
            paused: false,
            step_frame: 0,
        }
    }

    pub fn run(&mut self) {
        let mut fluid = fluid_sim::FluidSim::new(
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

            self.frame_count += 1;
            let elapsed = now.duration_since(self.last_fps_update).as_secs_f64();
            if elapsed >= 0.5 {
                self.fps = self.frame_count as f64 / elapsed;
                self.frame_count = 0;
                self.last_fps_update = now;
                let title = format!("Fluid Simulation - FPS: {:.1} ({})", self.fps, if self.paused { "PAUSED" } else { "RUNNING" });
                self.window.set_title(&title);
            }

            if self.window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
                self.paused = !self.paused;
            }

            if self.paused {
                if self.window.is_key_pressed(minifb::Key::Right, minifb::KeyRepeat::No) {
                    self.step_frame = 1;
                }
                if self.window.is_key_pressed(minifb::Key::Up, minifb::KeyRepeat::No) {
                    self.step_frame = 10;
                }
            }

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
                            let idx = x + y * fluid.width;
                            fluid.density[idx] = (fluid.density[idx] + 2.0 * dt).min(1.0);
                        }

                        if self.window.get_mouse_down(minifb::MouseButton::Right) {
                            let idx = x + y * fluid.width;
                            fluid.velocity_x[idx] += fx * 0.05;
                            fluid.velocity_y[idx] += fy * 0.05;
                        }
                    }
                }
            }

            last_mouse = (mx, my);

            if !self.paused || self.step_frame > 0 {
                fluid.step(dt.min(0.05), self.pressure_iters, self.diffusion_iters); // clamp dt for stability
                self.step_frame = self.step_frame.saturating_sub(1);
            }


            self.buffer.fill(0);

            for y in 0..fluid.height {
                let base_y = y * self.precision * self.width;
                for x in 0..fluid.width {
                    let d = fluid.get_density(x, y).clamp(0.0, 1.0);

                    let r = (d * ((self.max_color >> 0) & 0xFF) as f64) as u8;
                    let g = (d * ((self.max_color >> 8) & 0xFF) as f64) as u8;
                    let b = (d * ((self.max_color >> 16) & 0xFF) as f64) as u8;

                    let color =
                        ((b as u32) << 16) |
                        ((g as u32) << 8) |
                        (r as u32);

                    let base_x = x * self.precision;
                    for py in 0..self.precision {
                        let row_offset = base_y + py * self.width;
                        for px in 0..self.precision {
                            self.buffer[row_offset + base_x + px] = color;
                        }
                    }
                }
            }

            self.window
                .update_with_buffer(&self.buffer, self.width, self.height)
                .unwrap();
        }
    }
}

