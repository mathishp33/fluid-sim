use minifb::{Window, WindowOptions};

use crate::simulation::fluid_sim;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
    Density,
    Pressure,
    Divergence,
    Velocity,
}

impl DisplayMode {
    fn next(self) -> Self {
        match self {
            Self::Density => Self::Pressure,
            Self::Pressure => Self::Divergence,
            Self::Divergence => Self::Velocity,
            Self::Velocity => Self::Density,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Density => "DENSITY",
            Self::Pressure => "PRESSURE",
            Self::Divergence => "DIVERGENCE",
            Self::Velocity => "VELOCITY",
        }
    }
}

fn signed_color(value: f32, max_abs: f32) -> u32 {
    if max_abs <= 1e-12 {
        return 0;
    }

    let v = (value / max_abs).clamp(-1.0, 1.0);

    let r;
    let g;
    let b;

    if v < 0.0 {
        let t = -v;
        r = 0;
        g = (t * 255.0) as u8;
        b = (t * 255.0) as u8;
    } else {
        let t = v;
        r = (t * 255.0) as u8;
        g = 0;
        b = 0;
    }

    ((b as u32) << 16) | ((g as u32) << 8) | (r as u32)
}

// fn max_abs(field: &[f32]) -> f32 {
//     field
//         .iter()
//         .map(|v| v.abs())
//         .fold(0.0, f32::max)
// }

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> u32 {
    let h = (h % 360.0 + 360.0) % 360.0;

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h {
        h if h < 60.0 => (c, x, 0.0),
        h if h < 120.0 => (x, c, 0.0),
        h if h < 180.0 => (0.0, c, x),
        h if h < 240.0 => (0.0, x, c),
        h if h < 300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let r = ((r + m) * 255.0) as u32;
    let g = ((g + m) * 255.0) as u32;
    let b = ((b + m) * 255.0) as u32;

    (b << 16) | (g << 8) | r
}

fn velocity_color(vx: f32, vy: f32, max_speed: f32) -> u32 {
    let speed = (vx * vx + vy * vy).sqrt();

    if speed < 1e-12 || max_speed <= 1e-12 {
        return 0;
    }

    // direction vecteur -> angle -> "teine"
    let angle = vy.atan2(vx);
    let hue = angle.to_degrees() + 180.0;

    // norme vitesse -> luminosité
    let brightness = (speed / max_speed).clamp(0.0, 1.0);

    hsv_to_rgb(hue, 1.0, brightness)
}

pub struct FluidWindow {
    pub width: usize,
    pub height: usize,
    pub particle_radius: usize,
    pub precision : usize,
    pub window: Window,
    pub start_density: f32,
    pub diffusion_rate: f32,
    pub max_color: u32,
    pub randomize: bool,
    pub random_smoothing: usize,
    pub pressure_iters: usize,
    pub diffusion_iters: usize,
    buffer: Vec<u32>,
    fps: f32,
    frame_count: usize,
    last_fps_update: std::time::Instant,
    paused: bool,
    step_frame: usize,
    display_mode: DisplayMode,
}

impl FluidWindow {
    pub fn new(width: usize, height: usize, particle_radius: usize, precision: usize, start_density: f32, diffusion_rate: f32, max_color: u32, randomize: bool,
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
            display_mode: DisplayMode::Density,
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
            let dt = (now - last_time).as_secs_f32();
            last_time = now;

            if dt <= 0.0 {
                continue;
            }

            self.frame_count += 1;
            let elapsed = now.duration_since(self.last_fps_update).as_secs_f32();
            if elapsed >= 0.5 {
                self.fps = self.frame_count as f32 / elapsed;
                self.frame_count = 0;
                self.last_fps_update = now;
                let title = format!("Fluid Simulation - \
                FPS: {:.1} ({}) | {}", self.fps, if self.paused { "PAUSED" } else { "RUNNING" }, self.display_mode.name());
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

            if self.window.is_key_pressed(minifb::Key::V, minifb::KeyRepeat::No) {
                self.display_mode = self.display_mode.next();
            }

            let (mx, my) = self
                .window
                .get_mouse_pos(minifb::MouseMode::Clamp)
                .unwrap_or((0.0, 0.0));

            let mx = mx as usize;
            let my = my as usize;

            let fx = (mx as f32 - last_mouse.0 as f32) / dt;
            let fy = (my as f32 - last_mouse.1 as f32) / dt;

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

            let pressure_max = fluid.pressure
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f32::max);

            let divergence_max = fluid.divergence
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f32::max);

            let velocity_max = fluid.velocity_x
                .iter()
                .zip(fluid.velocity_y.iter())
                .map(|(&vx, &vy)| (vx * vx + vy * vy).sqrt())
                .fold(0.0, f32::max);

            let precision = self.precision;
            let max_color = self.max_color;
            let display_mode = self.display_mode;

            self.buffer
                .par_chunks_mut(self.width * precision)
                .enumerate()
                .for_each(|(y, screen_rows)| {
                    for x in 0..fluid.width {
                        let idx = x + y * fluid.width;

                        let color = match display_mode {
                            DisplayMode::Density => {
                                let d = fluid.density[idx].clamp(0.0, 1.0);

                                let r = (d * ((max_color >> 0) & 0xFF) as f32) as u8;
                                let g = (d * ((max_color >> 8) & 0xFF) as f32) as u8;
                                let b = (d * ((max_color >> 16) & 0xFF) as f32) as u8;

                                ((b as u32) << 16) |
                                    ((g as u32) << 8) |
                                    r as u32
                            }

                            DisplayMode::Pressure => {
                                signed_color(fluid.pressure[idx], pressure_max)
                            }

                            DisplayMode::Divergence => {
                                signed_color(fluid.divergence[idx], divergence_max)
                            }

                            DisplayMode::Velocity => {
                                velocity_color(fluid.velocity_x[idx], fluid.velocity_y[idx], velocity_max)
                            }
                        };

                        let base_x = x * precision;

                        for row in screen_rows.chunks_exact_mut(self.width) {
                            for px in 0..precision {
                                row[base_x + px] = color;
                            }
                        }
                    }
                });

            self.window
                .update_with_buffer(&self.buffer, self.width, self.height)
                .unwrap();
        }
    }
}

// c'est cool nan ? OwO