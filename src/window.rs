use std::time::Instant;
use minifb::{Window, WindowOptions};

mod fluid_sim;


pub struct FluidWindow {
    pub width: usize,
    pub height: usize,
    pub fps: i32,
    pub last_update: Instant,
    pub particle_radius: usize,
    pub precision : usize,
    pub window: Window,
    pub start_density: f64,
}

impl FluidWindow {
    pub fn new(width: usize, height: usize, fps: i32, particle_radius: usize, precision: usize, start_density: f64) -> Self {
        FluidWindow {
            width,
            height,
            fps,
            last_update: std::time::Instant::now(),
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
        }
    }

    pub fn run(&mut self) {
        let mut fluid = fluid_sim::Fluid::new((self.width / self.precision) as usize, (self.height / self.precision) as usize, self.start_density);
    
        let mut mouse_pos0 = (0, 0);
    
        while self.window.is_open() && !self.window.is_key_down(minifb::Key::Escape) {
            let now = std::time::Instant::now();
    
            if now.duration_since(self.last_update) >= std::time::Duration::from_secs_f64(1.0f64 / self.fps as f64) {
                let mut buffer: Vec<u32> = vec![0; self.width * self.height];
                self.last_update = now;
                
                self.window.update();
                let (mouse_x, mouse_y) = self.window.get_mouse_pos(minifb::MouseMode::Clamp).unwrap_or((0.0, 0.0));
                let mouse_x = mouse_x as usize;
                let mouse_y = mouse_y as usize;
                if self.window.get_mouse_down(minifb::MouseButton::Left) {
                    for dx in -(self.particle_radius as isize)..=(self.particle_radius as isize) {
                        for dy in -(self.particle_radius as isize)..=(self.particle_radius as isize) {
                            let nx = (mouse_x as isize + dx) as usize;
                            let ny = (mouse_y as isize + dy) as usize;
    
                            if nx < self.width && ny < self.height {
                                let distance_squared = (dx * dx + dy * dy) as f32;
                                if distance_squared < (self.particle_radius * self.particle_radius) as f32 {
                                    fluid.density[(nx / self.precision, ny / self.precision)] = 1.0;
                                    fluid.velocity_x[(nx / self.precision, ny / self.precision)] = 0.0;
                                    fluid.velocity_y[(nx / self.precision, ny / self.precision)] = 0.0;
                                }
                            }
                        }
                    }
                }
                if self.window.get_mouse_down(minifb::MouseButton::Right) {
                    for dx in -(self.particle_radius as isize)..=(self.particle_radius as isize) {
                        for dy in -(self.particle_radius as isize)..=(self.particle_radius as isize) {
                            let nx = (mouse_x as isize + dx) as usize;
                            let ny = (mouse_y as isize + dy) as usize;
    
                            if nx < self.width && ny < self.height {
                                let distance_squared = (dx * dx + dy * dy) as f32;
                                if distance_squared < (self.particle_radius * self.particle_radius) as f32 {
                                    fluid.velocity_x[(nx / self.precision, ny / self.precision)] = (mouse_x as f64 - mouse_pos0.0 as f64) * 0.5f64;
                                    fluid.velocity_y[(nx / self.precision, ny / self.precision)] = (mouse_y as f64 - mouse_pos0.1 as f64) * 0.5f64;
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
                        for i in 0..self.precision {
                            for j in 0..self.precision {
                                let buffer_index = ((y * self.precision + j) * self.width + (x * self.precision + i)) as usize;
                                buffer[buffer_index] = u32_color; 
                            }
                        }
                    }
                }
                self.window.update_with_buffer(&buffer, self.width, self.height).unwrap();
            } else {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

