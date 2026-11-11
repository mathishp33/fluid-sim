use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use egui::{Color32, ColorImage, PointerButton, Sense, TextureHandle, TextureOptions};
use rayon::prelude::*;
use rfd::FileDialog;

use crate::simulation::fluid::{CellEffect, FluidSim};
use crate::simulation::fluid::CellType;
use crate::window::starting_window::SimulationSettings;
use crate::window::starting_window::boundary_ui;

pub enum WindowAction {
    None,
    BackToSettings,
}

#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
    Density,
    Pressure,
    Divergence,
    Velocity,
}

impl DisplayMode {
    const ALL: [Self; 4] = [Self::Density, Self::Pressure, Self::Divergence, Self::Velocity];

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

#[derive(Clone, Copy, PartialEq)]
enum MouseMode {
    Rectangle,
    Circle,
}

impl MouseMode {
    const ALL: [Self; 2] = [Self::Rectangle, Self::Circle];

    fn next(self) -> Self {
        match self {
            Self::Rectangle => Self::Circle,
            Self::Circle => Self::Rectangle,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Rectangle => "RECTANGLE",
            Self::Circle => "CIRCLE",
        }
    }
}

const RADII: &[usize] = &[
    1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350,
];

fn cycle_radius(current: usize, reverse: bool) -> usize {
    if reverse {
        RADII.iter().rev().copied().find(|&r| r < current).unwrap_or(RADII[RADII.len() - 1])
    } else {
        RADII.iter().copied().find(|&r| r > current).unwrap_or(RADII[0])
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

    // direction vecteur -> angle -> "teinte"
    let angle = vy.atan2(vx);
    let hue = angle.to_degrees() + 180.0;

    // norme vitesse -> luminosité
    let brightness = (speed / max_speed).clamp(0.0, 1.0);

    hsv_to_rgb(hue, 1.0, brightness)
}

pub fn save_vec(data: &[CellType]) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let path = FileDialog::new()
        .add_filter("Fluid Sim File", &["fsf"])
        .set_file_name("simulation.fsf")
        .save_file();

    if let Some(path) = path {
        let encoded = bincode::serialize(data)?;
        fs::write(&path, encoded)?;
        return Ok(Some(path));
    }

    Ok(None)
}

pub fn load_vec() -> Result<Option<Vec<CellType>>, Box<dyn std::error::Error>> {
    let path = FileDialog::new()
        .add_filter("Fluid Sim File", &["fsf"])
        .pick_file();

    if let Some(path) = path {
        let bytes = fs::read(path)?;
        let decoded: Vec<CellType> = bincode::deserialize(&bytes)?;

        return Ok(Some(decoded));
    }

    Ok(None)
}


fn pressed_once(input: &egui::InputState, key: egui::Key) -> bool {
    input.events.iter().any(|e| {
        matches!(e, egui::Event::Key { key: k, pressed: true, repeat: false, .. } if *k == key)
    })
}

#[derive(Clone, Copy)]
struct MouseInput {
    left: bool,
    right: bool,
    middle: bool,
    ctrl: bool,
}

#[derive(Clone, Copy, Default)]
struct Stats {
    pressure_max: f32,
    divergence_max: f32,
    velocity_max: f32,
    total_density: f64,
    kinetic_energy: f32,
}


pub struct FluidWindow {
    fluid: FluidSim,
    texture: Option<TextureHandle>,
    rgba: Vec<u8>,

    precision: usize,
    mouse_radius: usize,
    pressure_iters: usize,
    max_color: u32,

    fps: f32,
    frame_count: usize,
    last_fps_update: Instant,
    last_frame: Instant,

    paused: bool,
    step_frame: usize,
    display_mode: DisplayMode,
    mouse_mode: MouseMode,
    inverse_solid_color: bool,
    selected_cell_type: CellType,
    current_cell_effect: Option<CellEffect>,

    last_mouse: Option<(f32, f32)>,
    hovered_cell: Option<(usize, usize)>,
    stats: Stats,
    status: String,
    show_help: bool,
}

impl FluidWindow {
    pub fn new(settings: &SimulationSettings) -> Self {
        let precision = settings.precision.max(1);

        let mut fluid = FluidSim::new(
            settings.width / precision,
            settings.height / precision,
            settings.start_density,
            settings.diffusion_rate,
        );
        fluid.left_boundary = settings.left_boundary;
        fluid.right_boundary = settings.right_boundary;
        fluid.top_boundary = settings.top_boundary;
        fluid.bottom_boundary = settings.bottom_boundary;

        let size = fluid.width * fluid.height * 4;
        let now = Instant::now();

        Self {
            fluid,
            texture: None,
            rgba: vec![255; size],

            precision,
            mouse_radius: 20,
            pressure_iters: settings.pressure_iters,
            max_color: settings.max_color(),

            fps: 0.0,
            frame_count: 0,
            last_fps_update: now,
            last_frame: now,

            paused: false,
            step_frame: 0,
            display_mode: DisplayMode::Density,
            mouse_mode: MouseMode::Rectangle,
            inverse_solid_color: true,
            selected_cell_type: CellType::Wall,
            current_cell_effect: None,

            last_mouse: None,
            hovered_cell: None,
            stats: Stats::default(),
            status: String::from("Ready"),
            show_help: false,
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> WindowAction {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().max(1e-4);
        self.last_frame = now;
        self.update_fps(now);

        let mut action = WindowAction::None;

        self.handle_shortcuts(ctx, &mut action);

        self.top_bar(ctx, &mut action);
        self.status_bar(ctx);
        self.controls_panel(ctx);
        self.info_panel(ctx);
        self.canvas(ctx, dt);
        self.help_window(ctx);

        ctx.request_repaint();
        action
    }

    fn update_fps(&mut self, now: Instant) {
        self.frame_count += 1;
        let elapsed = now.duration_since(self.last_fps_update).as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_fps_update = now;
        }
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        if ctx.wants_keyboard_input() {
            return;
        }

        let key = |k: egui::Key| ctx.input(|i| pressed_once(i, k));
        let shift = ctx.input(|i| i.modifiers.shift);

        if key(egui::Key::Space) {
            self.paused = !self.paused;
        }

        if self.paused {
            if key(egui::Key::ArrowRight) {
                self.step_frame = 1;
            }
            if key(egui::Key::ArrowUp) {
                self.step_frame = 10;
            }
        }

        if key(egui::Key::S) {
            self.save();
        }
        if key(egui::Key::L) {
            self.load();
        }
        if key(egui::Key::I) {
            self.inverse_solid_color = !self.inverse_solid_color;
        }
        if key(egui::Key::V) {
            self.display_mode = self.display_mode.next();
        }
        if key(egui::Key::M) {
            self.mouse_mode = self.mouse_mode.next();
        }
        if key(egui::Key::R) {
            self.reset_flow();
        }
        // C se répète quand on le maintient (comme KeyRepeat::Yes)
        if ctx.input(|i| i.key_pressed(egui::Key::C)) {
            self.mouse_radius = cycle_radius(self.mouse_radius, shift);
        }
        if key(egui::Key::Escape) {
            *action = WindowAction::BackToSettings;
        }
    }

    fn reset_flow(&mut self) {
        self.fluid.pressure.fill(0.0);
        self.fluid.divergence.fill(0.0);
        self.fluid.velocity_x.fill(0.0);
        self.fluid.velocity_y.fill(0.0);
        self.status = "Flow reset".into();
    }

    fn save(&mut self) {
        match save_vec(&self.fluid.solids) {
            Ok(Some(path)) => self.status = format!("Saved to {}", path.display()),
            Ok(None) => {}
            Err(e) => self.status = format!("Save error: {e}"),
        }
        self.after_blocking_dialog();
    }

    fn load(&mut self) {
        match load_vec() {
            Ok(Some(data)) if data.len() == self.fluid.solids.len() => {
                self.fluid.solids = data;
                self.status = "Solids loaded".into();
            }
            Ok(Some(_)) => {
                self.status = "Invalid vector size (grid dimensions differ)".into();
            }
            Ok(None) => {}
            Err(e) => self.status = format!("Load error: {e}"),
        }
        self.after_blocking_dialog();
    }


    fn after_blocking_dialog(&mut self) {
        self.last_frame = Instant::now();
        self.last_mouse = None;
    }


    fn top_bar(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Save solids (S)").clicked() {
                    self.save();
                }
                if ui.button("Load solids (L)").clicked() {
                    self.load();
                }

                ui.separator();

                let label = if self.paused { "Resume (Space)" } else { "Pause (Space)" };
                if ui.button(label).clicked() {
                    self.paused = !self.paused;
                }
                if ui
                    .add_enabled(self.paused, egui::Button::new("Step (Right)"))
                    .clicked()
                {
                    self.step_frame = 1;
                }
                if ui.button("Reset flow (R)").clicked() {
                    self.reset_flow();
                }

                ui.separator();

                if ui.button("Help").clicked() {
                    self.show_help = !self.show_help;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Back to settings (Esc)").clicked() {
                        *action = WindowAction::BackToSettings;
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("LMB: density (Ctrl: solid) | RMB: push | MMB: erase (Ctrl: remove solid)");
                });
            });
        });
    }

    fn controls_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls").default_width(240.0).show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.label("Display (V)");
            for mode in DisplayMode::ALL {
                ui.selectable_value(&mut self.display_mode, mode, mode.name());
            }

            ui.separator();
            ui.label("Brush shape (M)");
            for mode in MouseMode::ALL {
                ui.selectable_value(&mut self.mouse_mode, mode, mode.name());
            }
            ui.add(egui::Slider::new(&mut self.mouse_radius, 1..=350).text("Radius (C)"));

            ui.separator();
            ui.checkbox(&mut self.inverse_solid_color, "White solids (I)");

            ui.separator();
            ui.heading("Live Settings");
            ui.separator();
            ui.label("Solver");
            ui.add(egui::Slider::new(&mut self.pressure_iters, 0..=200).text("Pressure iters"));
            ui.add(egui::Slider::new(&mut self.fluid.sor_omega, 1.0..=1.95).text("SOR omega"));
            ui.add(egui::Slider::new(&mut self.fluid.diffusion_rate, 0.0..=5.0).text("Diffusion rate"));
            ui.separator();
            ui.label("Boundaries");
            boundary_ui(ui, "Left Boundary", &mut self.fluid.left_boundary);
            boundary_ui(ui, "Right Boundary", &mut self.fluid.right_boundary);
            boundary_ui(ui, "Top Boundary", &mut self.fluid.top_boundary);
            boundary_ui(ui, "Bottom Boundary", &mut self.fluid.bottom_boundary);

            ui.separator();
            ui.heading("Cell Effects");
            for cell_type in CellType::ALL_EXCEPT_ONE {
                ui.selectable_value(&mut self.selected_cell_type, cell_type, cell_type.to_string());
            }
            self.current_cell_effect = CellEffect::default_for(self.selected_cell_type);
            ui.collapsing(self.selected_cell_type.to_string(), |ui| {
                if (self.current_cell_effect != None) {
                    match self.current_cell_effect.unwrap() {
                        CellEffect::Sink { mut rate } => {
                            ui.add(egui::Slider::new(&mut rate, 0.0..=10.0).text("Density Rate"));
                        }
                        CellEffect::Emitter { mut density, mut vx, mut vy } => {
                            ui.add(egui::Slider::new(&mut density, 0.0..=1.0).text("Density"));
                            ui.add(egui::Slider::new(&mut vx, -200.0..=200.0).text("X velocity"));
                            ui.add(egui::Slider::new(&mut vy, -200.0..=200.0).text("Y velocity"));
                        }
                        CellEffect::Accelerator { mut ax, mut ay } => {
                            ui.add(egui::Slider::new(&mut ax, -50.0..=50.0).text("X acceleration"));
                            ui.add(egui::Slider::new(&mut ay, -50.0..=50.0).text("Y acceleration"));
                        }
                        CellEffect::Teleporter { .. } => {
                            //todo
                        }
                    }
                }
            });
        });
    }

    fn info_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("info").default_width(240.0).show(ctx, |ui| {
            ui.heading("Info");
            ui.separator();

            egui::Grid::new("info_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                let mut row = |name: &str, value: String| {
                    ui.label(name);
                    ui.label(value);
                    ui.end_row();
                };

                row("FPS", format!("{:.1}", self.fps));
                row("State", if self.paused { "PAUSED".to_string() } else { "RUNNING".to_string() });
                row("Grid", format!("{} x {}", self.fluid.width, self.fluid.height));
                row("Precision", format!("{} px", self.precision));
                row("Display", self.display_mode.name().to_string());
                row("Brush", format!("{} ({} px)", self.mouse_mode.name(), self.mouse_radius));
                row("Max |pressure|", format!("{:.3}", self.stats.pressure_max));
                row("Max |divergence|", format!("{:.3}", self.stats.divergence_max));
                row("Max speed", format!("{:.3}", self.stats.velocity_max));
                row("Total density", format!("{:.4}", self.stats.total_density));
                row("Total kinetic energy", format!("{:.2}", self.stats.kinetic_energy));
            });

            ui.separator();
            ui.heading("Cell under cursor");

            match self.hovered_cell {
                Some((x, y)) => {
                    let idx = x + y * self.fluid.width;
                    egui::Grid::new("cell_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                        let mut row = |name: &str, value: String| {
                            ui.label(name);
                            ui.label(value);
                            ui.end_row();
                        };
                        row("Position", format!("{x}, {y}"));
                        let (cvx, cvy) = self.fluid.cell_velocity(x, y);
                        row("Velocity X", format!("{:.3}", cvx));
                        row("Velocity Y", format!("{:.3}", cvy));
                        row("Density", format!("{:.3}", self.fluid.density[idx]));
                        row("Pressure", format!("{:.3}", self.fluid.pressure[idx]));
                        row("Divergence", format!("{:.3}", self.fluid.divergence[idx]));
                        row("Solid", format!("{}", self.fluid.solids[idx].to_string()));
                    });
                }
                None => {
                    ui.weak("—");
                }
            }
        });
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Shortcuts")
            .open(&mut self.show_help)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("help_grid").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                    for (k, v) in [
                        ("Space", "Pause / resume"),
                        ("Right / Up", "Step 1 / 10 frames (paused)"),
                        ("S / L", "Save / load solids"),
                        ("V", "Next display mode"),
                        ("M", "Brush shape"),
                        ("C / Shift+C", "Brush radius up / down"),
                        ("I", "Invert solid color"),
                        ("R", "Reset pressure, divergence and velocity"),
                        ("Esc", "Back to settings"),
                        ("LMB", "Add density (Ctrl/Cmd: draw solid)"),
                        ("RMB", "Push fluid"),
                        ("MMB", "Erase density (Ctrl/Cmd: remove solid)"),
                    ] {
                        ui.strong(k);
                        ui.label(v);
                        ui.end_row();
                    }
                });
            });
    }

    fn canvas(&mut self, ctx: &egui::Context, dt: f32) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (outer, response) = ui.allocate_exact_size(avail, Sense::click_and_drag());
            if outer.width() < 1.0 || outer.height() < 1.0 {
                return;
            }

            let gw = self.fluid.width as f32;
            let gh = self.fluid.height as f32;
            let scale = (outer.width() / gw).min(outer.height() / gh);
            let rect = egui::Rect::from_center_size(outer.center(), egui::vec2(gw * scale, gh * scale));

            self.handle_mouse(ctx, &response, rect, dt);
            self.simulate(dt);
            self.update_stats();
            self.update_texture(ctx);

            ui.painter().rect_filled(outer, 0.0, Color32::BLACK);
            if let Some(tex) = &self.texture {
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        });
    }

    fn handle_mouse(&mut self, ctx: &egui::Context, response: &egui::Response, rect: egui::Rect, dt: f32) {
        let (pos, input) = ctx.input(|i| {
            (
                i.pointer.latest_pos(),
                MouseInput {
                    left: i.pointer.button_down(PointerButton::Primary),
                    right: i.pointer.button_down(PointerButton::Secondary),
                    middle: i.pointer.button_down(PointerButton::Middle),
                    ctrl: i.modifiers.command, // Ctrl (Win/Linux) ou Cmd (macOS)
                },
            )
        });

        let over = match pos {
            Some(p) if response.contains_pointer() && rect.contains(p) => Some(p),
            _ => None,
        };

        let Some(p) = over else {
            self.hovered_cell = None;
            self.last_mouse = None;
            return;
        };

        let u = ((p.x - rect.min.x) / rect.width()).clamp(0.0, 0.9999);
        let v = ((p.y - rect.min.y) / rect.height()).clamp(0.0, 0.9999);

        let gw = self.fluid.width as f32;
        let gh = self.fluid.height as f32;
        let gx = (u * gw) as usize;
        let gy = (v * gh) as usize;
        self.hovered_cell = Some((gx, gy));

        let mx = u * gw * self.precision as f32;
        let my = v * gh * self.precision as f32;

        let (fx, fy) = match self.last_mouse {
            Some((lx, ly)) => ((mx - lx) / dt, (my - ly) / dt),
            None => (0.0, 0.0),
        };
        self.last_mouse = Some((mx, my));

        self.apply_brush(gx, gy, fx, fy, dt, input);
    }

    fn apply_brush(&mut self, gx: usize, gy: usize, fx: f32, fy: f32, dt: f32, input: MouseInput) {
        if !(input.left || input.right || input.middle) {
            return;
        }

        let fluid = &mut self.fluid;
        let (w, h) = (fluid.width, fluid.height);

        if gx <= 1 || gx >= w - 1 || gy <= 1 || gy >= h - 1 {
            return;
        }

        let precision = self.precision as f32;
        let radius = self.mouse_radius as f32;
        let mouse_mode = self.mouse_mode;
        let r = (radius / precision).ceil() as isize;

        for dx in -r..=r {
            for dy in -r..=r {
                let x = gx as isize + dx;
                let y = gy as isize + dy;

                if x <= 0 || y <= 0 || x >= w as isize - 1 || y >= h as isize - 1 {
                    continue;
                }

                let dist_x = dx as f32 * precision;
                let dist_y = dy as f32 * precision;

                let inside = match mouse_mode {
                    MouseMode::Rectangle => dist_x.abs() <= radius && dist_y.abs() <= radius,
                    MouseMode::Circle => dist_x * dist_x + dist_y * dist_y <= radius * radius,
                };
                if !inside {
                    continue;
                }

                let x = x as usize;
                let y = y as usize;
                let idx = x + y * w;

                if input.left {
                    if input.ctrl {
                        if self.selected_cell_type == CellType::Wall {
                            fluid.set_cell(x, y, CellType::Wall, None);
                        }
                        else {
                            fluid.set_cell(x, y, self.selected_cell_type, Option::from(self.current_cell_effect));
                        }
                    } else if fluid.solids[idx] == CellType::Fluid {
                        fluid.density[idx] = (fluid.density[idx] + 2.0 * dt).min(1.0);
                    }
                }

                if input.right {
                    fluid.velocity_x[idx] += fx * 0.07;
                    fluid.velocity_y[idx] += fy * 0.07;
                }

                if input.middle {
                    if input.ctrl {
                        fluid.set_cell(x, y, CellType::Fluid, None);
                    } else {
                        fluid.density[idx] = (fluid.density[idx] - 2.0 * dt).max(0.0);
                    }
                }
            }
        }
    }

    fn simulate(&mut self, dt: f32) {
        if !self.paused || self.step_frame > 0 {
            self.fluid.step(dt.min(0.05), self.pressure_iters); //todo!(baisser 0.05)
            self.step_frame = self.step_frame.saturating_sub(1);
        }
    }

    fn update_stats(&mut self) {
        let f = &self.fluid;
        let (w, h) = (f.width, f.height);

        let (speed_max, kinetic) = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut m = 0.0f32;
                let mut e = 0.0f32;
                for x in 0..w {
                    let (vx, vy) = FluidSim::cell_velocity_from(&f.velocity_x, &f.velocity_y, w, h, x, y);
                    let s2 = vx * vx + vy * vy;
                    m = m.max(s2);
                    e += f.density[x + y * w] * s2;
                }
                (m.sqrt(), 0.5 * e)
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0.max(b.0), a.1 + b.1));

        self.stats = Stats {
            pressure_max: f.pressure.par_iter().map(|v| v.abs()).reduce(|| 0.0, f32::max),
            divergence_max: f.divergence.par_iter().map(|v| v.abs()).reduce(|| 0.0, f32::max),
            velocity_max: speed_max,
            total_density: f.total_mass(),
            kinetic_energy: kinetic,
        };
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let w = self.fluid.width;
        let h = self.fluid.height;
        self.rgba.resize(w * h * 4, 255);

        let mode = self.display_mode;
        let max_color = self.max_color;
        let solid_color: u32 = if self.inverse_solid_color { 0xffffff } else { 0x000000 };
        let Stats { pressure_max, divergence_max, velocity_max, .. } = self.stats;

        let density = &self.fluid.density[..];
        let pressure = &self.fluid.pressure[..];
        let divergence = &self.fluid.divergence[..];
        let velocity_x = &self.fluid.velocity_x[..];
        let velocity_y = &self.fluid.velocity_y[..];
        let grid_h = h;
        let solids = &self.fluid.solids[..];

        self.rgba
            .par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, px) in row.chunks_exact_mut(4).enumerate() {
                    let idx = x + y * w;

                    let color: u32 = if solids[idx] == CellType::Wall {
                        match mode {
                            DisplayMode::Density => 0x000000,
                            _ => solid_color,
                        }
                    } else {
                        match mode {
                            DisplayMode::Density => {
                                let d = density[idx].clamp(0.0, 1.0);
                                let r = (d * ((max_color >> 16) & 0xFF) as f32) as u32;
                                let g = (d * ((max_color >> 8) & 0xFF) as f32) as u32;
                                let b = (d * (max_color & 0xFF) as f32) as u32;
                                (r << 16) | (g << 8) | b
                            }
                            DisplayMode::Pressure => signed_color(pressure[idx], pressure_max),
                            DisplayMode::Divergence => signed_color(divergence[idx], divergence_max),
                            DisplayMode::Velocity => {
                                let (vx, vy) = FluidSim::cell_velocity_from(velocity_x, velocity_y, w, grid_h, x, y);
                                velocity_color(vx, vy, velocity_max)
                            }
                        }
                    };

                    px[0] = (color >> 16) as u8;
                    px[1] = (color >> 8) as u8;
                    px[2] = color as u8;
                    px[3] = 255;
                }
            });

        let image = ColorImage::from_rgba_unmultiplied([w, h], &self.rgba);
        match &mut self.texture {
            Some(tex) => tex.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("fluid", image, TextureOptions::NEAREST)),
        }
    }
}