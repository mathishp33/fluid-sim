use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use egui::{Color32, ColorImage, PointerButton, Sense, TextureHandle, TextureOptions};
use rayon::prelude::*;
use rfd::FileDialog;

use crate::simulation::fluid::{BoundaryType, CellEffect, ComputeMethod, FluidSim, ThermalCell};
use crate::simulation::materials::FluidKind;
use crate::simulation::fluid::CellType;
use crate::window::starting_window::SimulationSettings;
use crate::window::starting_window::boundary_ui;

const MAX_DT: f32 = 0.05;
const HEAT_RATE: f32 = 5.0; // 1/s : vitesse à laquelle le pinceau tire T vers la cible
const PUSH_RATE: f32 = 20.0; // 1/s : vitesse à laquelle le fluide rejoint la vitesse du curseur (clic droit)

pub enum WindowAction {
    None,
    BackToSettings,
}

#[derive(Clone, Copy, PartialEq)]
enum DisplayMode {
    Tracer,
    Pressure,
    Divergence,
    Velocity,
    Temperature,
    Rho,
}

impl DisplayMode {
    const ALL: [Self; 6] = [Self::Tracer, Self::Pressure, Self::Divergence, Self::Velocity, Self::Temperature, Self::Rho];

    fn next(self) -> Self {
        match self {
            Self::Tracer => Self::Pressure,
            Self::Pressure => Self::Divergence,
            Self::Divergence => Self::Velocity,
            Self::Velocity => Self::Temperature,
            Self::Temperature => Self::Rho,
            Self::Rho => Self::Tracer,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Tracer => "TRACER",
            Self::Pressure => "PRESSURE (dyn)",
            Self::Divergence => "DIVERGENCE",
            Self::Velocity => "VELOCITY",
            Self::Temperature => "TEMPERATURE",
            Self::Rho => "RHO (kg/m³)",
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

#[derive(Clone, Copy, PartialEq)]
enum ThermalMode {
    Passive,
    Isothermal,
    Heater,
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

/// x = (T − T_ref) / échelle : bleu/cyan (froid) -> noir (ambiant) -> rouge -> jaune (chaud)
fn temperature_color(x: f32) -> u32 {
    let x = x.clamp(-1.0, 1.0);
    let (r, g, b) = if x >= 0.0 {
        ((2.0 * x).min(1.0), (2.0 * x - 1.0).max(0.0), 0.0)
    } else {
        let c = -x;
        (0.0, (2.0 * c - 1.0).max(0.0), (2.0 * c).min(1.0))
    };
    (((r * 255.0) as u32) << 16) | (((g * 255.0) as u32) << 8) | (b * 255.0) as u32
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
    shift: bool,
}

#[derive(Clone, Copy, Default)]
struct Stats {
    pressure_max: f32, // unités du solveur (sert à la colorisation)
    pressure_dyn_pa: f32, // Pa
    divergence_max: f32, // 1/s : divergence réelle
    velocity_max: f32, // cellules/s (colorisation, CFL)
    velocity_max_ms: f32, // m/s
    total_tracer: f64, // sans dimension
    total_mass: f64, // kg par mètre de profondeur
    kinetic_energy: f32, // J par mètre de profondeur
    pressure_residual: f32, // max |div − S| (1/s) sur les cellules vues par le solveur
    divergence_rms: f32, // RMS de (div − S) (1/s)
    velocity_rms: f32, // m/s
    cfl: f32, // max|v|·dt en cellules
    p_th: f32,
    dpdt: f32,
    rho_min: f32,
    rho_max: f32,
    rho_dev_max: f32, // max |ρ − ρ_ambiant|
    t_min: f32,
    t_max: f32,
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
    show_cell_effects: bool,
    teleporter_dest: Option<usize>,
    picking_destination: bool,
    last_effect_type: CellType,

    brush_temperature: f32,
    temperature_scale: f32, // K, échelle d'affichage de ΔT
    thermal_mode: ThermalMode,
    thermal_temp: f32,  // K (solide isotherme)
    thermal_power: f32, // W/m³ (solide chauffant)

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
            settings.dx,
            settings.material,
            settings.gravity,
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
            display_mode: DisplayMode::Tracer,
            mouse_mode: MouseMode::Rectangle,
            inverse_solid_color: true,
            selected_cell_type: CellType::Wall,
            current_cell_effect: None,
            show_cell_effects: true,
            teleporter_dest: None,
            picking_destination: false,
            last_effect_type: CellType::Wall,

            brush_temperature: settings.material.t_ref + 100.0,
            temperature_scale: 100.0,
            thermal_mode: ThermalMode::Passive,
            thermal_temp: settings.material.t_ref + 100.0,
            thermal_power: 1e6,

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
        self.fluid.source.fill(0.0);
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
                    ui.weak("LMB: tracer (Ctrl: solid) | RMB: push | MMB: erase (Ctrl: remove solid) | Shift+MMB: heat to brush T");
                });
            });
        });
    }

    fn controls_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls").default_width(240.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
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
                ui.horizontal(|ui| {
                    ui.label("Backend");
                    let current = self.fluid.compute_method();
                    let mut method = current;
                    ui.selectable_value(&mut method, ComputeMethod::CPU, "CPU");
                    ui.selectable_value(&mut method, ComputeMethod::GPU, "GPU");
                    if method != current {
                        self.fluid.set_compute_method(method);
                    }
                });
                if self.fluid.material.kind == FluidKind::Gas {
                    ui.checkbox(&mut self.fluid.low_mach, "Low-Mach (compressible gas)");
                }
                ui.checkbox(&mut self.fluid.no_slip, "No-slip walls (adhérence)");

                ui.separator();
                ui.heading("Fluid (SI)");
                ui.add(egui::Slider::new(&mut self.fluid.material.viscosity, 1e-7..=1e-2).logarithmic(true).text("Viscosity ν (m²/s)"));
                ui.add(egui::Slider::new(&mut self.fluid.material.tracer_diffusion, 1e-10..=1e-2).logarithmic(true).text("Tracer diffusion D (m²/s)"));

                ui.separator();
                ui.heading("Temperature");
                ui.add(egui::Slider::new(&mut self.fluid.material.conductivity, 0.01..=400.0).logarithmic(true).text("Fluid k (W/(m·K))"));
                ui.add(egui::Slider::new(&mut self.fluid.material.solid_conductivity, 0.01..=400.0).logarithmic(true).text("Solid k (W/(m·K))"));
                ui.add(egui::Slider::new(&mut self.fluid.gravity, 0.0..=100.0).text("Gravity (m/s²)"));
                ui.add(egui::Slider::new(&mut self.brush_temperature, 100.0..=2000.0).text("Brush temp (K)"));
                ui.add(egui::Slider::new(&mut self.temperature_scale, 1.0..=1000.0).logarithmic(true).text("Display ΔT scale (K)"));
                if ui.button("Reset temperature").clicked() {
                    self.fluid.temperature.fill(self.fluid.material.t_ref);
                    self.fluid.p_th = self.fluid.material.p_ref;
                    self.status = "Temperature reset".into();
                }

                ui.separator();
                ui.label("Boundaries");
                let dx = self.fluid.dx;
                boundary_ui(ui, "Left Boundary", &mut self.fluid.left_boundary, dx);
                boundary_ui(ui, "Right Boundary", &mut self.fluid.right_boundary, dx);
                boundary_ui(ui, "Top Boundary", &mut self.fluid.top_boundary, dx);
                boundary_ui(ui, "Bottom Boundary", &mut self.fluid.bottom_boundary, dx);

                ui.separator();
                ui.heading("Cell Effects");
                ui.checkbox(&mut self.show_cell_effects, "Show cell effects");
                ui.separator();
                for cell_type in CellType::PAINTABLE {
                    ui.selectable_value(&mut self.selected_cell_type, cell_type, cell_type.to_string());
                }
                ui.separator();
                if self.selected_cell_type != self.last_effect_type {
                    self.last_effect_type = self.selected_cell_type;
                    self.current_cell_effect = CellEffect::default_for(self.selected_cell_type);
                    self.picking_destination = false;
                }
                self.effect_ui(ui);

                ui.separator();
                ui.heading("Solid thermal (Wall + Ctrl+LMB)");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Passive, "Passive solid");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Isothermal, "Isothermal (fixed T)");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Heater, "Heater (W/m³)");
                match self.thermal_mode {
                    ThermalMode::Passive => {}
                    ThermalMode::Isothermal => {
                        ui.add(egui::Slider::new(&mut self.thermal_temp, 100.0..=2000.0).text("T (K)"));
                    }
                    ThermalMode::Heater => {
                        ui.add(egui::Slider::new(&mut self.thermal_power, 1e2..=1e9).logarithmic(true).text("Power (W/m³)"));
                        let dx = self.fluid.dx;
                        ui.weak(format!("= {:.2} W per metre of depth per cell", self.thermal_power * dx * dx));
                    }
                }
            });
        });
    }

    fn effect_ui(&mut self, ui: &mut egui::Ui) {
        if self.selected_cell_type == CellType::Teleporter {
            ui.label("1. Pick a destination\n2. Ctrl+LMB to paint teleporters");
            let label = if self.picking_destination { "Click on the canvas..." } else { "Pick destination" };
            if ui.button(label).clicked() {
                self.picking_destination = !self.picking_destination;
            }
            match self.teleporter_dest {
                Some(idx) => {
                    let w = self.fluid.width;
                    ui.label(format!("Destination: {}, {}", idx % w, idx / w));
                }
                None => {
                    ui.weak("No destination set");
                }
            }
            self.current_cell_effect =
                self.teleporter_dest.map(|d| CellEffect::Teleporter { dest_idx: d });
            return;
        }

        if let Some(effect) = &mut self.current_cell_effect {
            match effect {
                CellEffect::Sink { rate } => {
                    ui.add(egui::Slider::new(rate, 0.0..=10.0).text("Suction (m/s)"));
                }
                CellEffect::Emitter { tracer, vx, vy } => {
                    ui.add(egui::Slider::new(tracer, 0.0..=1.0).text("Tracer"));
                    ui.add(egui::Slider::new(vx, -10.0..=10.0).text("X velocity (m/s)"));
                    ui.add(egui::Slider::new(vy, -10.0..=10.0).text("Y velocity (m/s)"));
                }
                CellEffect::Accelerator { ax, ay } => {
                    ui.add(egui::Slider::new(ax, -20.0..=20.0).text("X acceleration (m/s²)"));
                    ui.add(egui::Slider::new(ay, -20.0..=20.0).text("Y acceleration (m/s²)"));
                }
                CellEffect::Teleporter { .. } => {}
            }
        }
    }

    fn info_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("info").default_width(240.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Info");
                ui.separator();
                ui.collapsing("Info", |ui| {
                egui::Grid::new("info_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    let mut row = |name: &str, value: String| {
                        ui.label(name);
                        ui.label(value);
                        ui.end_row();
                    };

                    let f = &self.fluid;
                    row("FPS", format!("{:.1}", self.fps));
                    row("State", if self.paused { "PAUSED".to_string() } else { "RUNNING".to_string() });
                    row("Fluid", format!("{:?}{}", f.material.kind, if f.low_mach_active() { " (low-Mach)" } else { "" }));
                    row("Backend", format!("{:?}", f.compute_method()));
                    row("Grid", format!("{} x {}", f.width, f.height));
                    row("Domain (m)", format!("{:.3} x {:.3}", f.width as f32 * f.dx, f.height as f32 * f.dx));
                    row("Cell size (m)", format!("{:.4}", f.dx));
                    row("Precision", format!("{} px", self.precision));
                    row("Display", self.display_mode.name().to_string());
                    row("Brush", format!("{} ({} px)", self.mouse_mode.name(), self.mouse_radius));
                    row("P_th (Pa)", format!("{:.1}", self.stats.p_th));
                    row("dP_th/dt (Pa/s)", format!("{:.2}", self.stats.dpdt));
                    row("Max |P dyn| (Pa)", format!("{:.4}", self.stats.pressure_dyn_pa));
                    row("Max |div| (1/s)", format!("{:.4}", self.stats.divergence_max));
                    row("Max speed (m/s)", format!("{:.4}", self.stats.velocity_max_ms));
                    row("rho min / max (kg/m³)", format!("{:.4} / {:.4}", self.stats.rho_min, self.stats.rho_max));
                    row("T min / max (K)", format!("{:.2} / {:.2}", self.stats.t_min, self.stats.t_max));
                    row("Total tracer", format!("{:.4}", self.stats.total_tracer));
                    row("Total mass (kg/m)", format!("{:.5}", self.stats.total_mass));
                    row("Kinetic energy (J/m)", format!("{:.4e}", self.stats.kinetic_energy));
                    row("|div − S| max (1/s)", format!("{:.4e}", self.stats.pressure_residual));
                    row("|div − S| RMS (1/s)", format!("{:.4e}", self.stats.divergence_rms));
                    row("Velocity RMS (m/s)", format!("{:.4}", self.stats.velocity_rms));
                    row("CFL (max|v|·dt, cells)", format!("{:.3}", self.stats.cfl));
                });
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
                            let dx = self.fluid.dx;
                            let (cvx, cvy) = self.fluid.cell_velocity(x, y);
                            row("Velocity X (m/s)", format!("{:.4}", cvx * dx));
                            row("Velocity Y (m/s)", format!("{:.4}", cvy * dx));
                            row("Tracer", format!("{:.3}", self.fluid.tracer[idx]));
                            row("rho (kg/m³)", format!("{:.4}", self.fluid.rho[idx]));
                            row("Temperature (K)", format!("{:.2}", self.fluid.temperature[idx]));
                            row("P total (Pa)", format!("{:.2}", self.fluid.pressure_pa[idx]));
                            row("P dyn (Pa)", format!("{:.4}", self.fluid.pressure[idx] * self.fluid.pressure_scale()));
                            row("Divergence (1/s)", format!("{:.4}", self.fluid.divergence[idx]));
                            row("Source S (1/s)", format!("{:.4}", self.fluid.source[idx]));
                            row("Solid", self.fluid.solids[idx].to_string());
                        });
                    }
                    None => {
                        ui.weak("—");
                    }
                }
            });
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
                        ("R", "Reset pressure, divergence, source and velocity"),
                        ("Esc", "Back to settings"),
                        ("LMB", "Add tracer (Ctrl/Cmd: draw solid)"),
                        ("RMB", "Drag fluid toward cursor velocity"),
                        ("MMB", "Erase tracer (Ctrl/Cmd: remove solid)"),
                        ("Shift+MMB", "Heat / cool toward brush temperature"),
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
            if !self.paused || self.step_frame > 0  {
                self.update_stats(dt.min(MAX_DT));
            }
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
                    shift: i.modifiers.shift,
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

        if self.picking_destination {
            if response.clicked() {
                let w = self.fluid.width;
                let h = self.fluid.height;
                let idx = gx + gy * w;
                if gx < 1 || gx >= w - 1 || gy < 1 || gy >= h - 1 {
                    self.status = "Destination must be inside the grid".into();
                } else if self.fluid.solids[idx] == CellType::Wall {
                    self.status = "Destination can't be a wall".into();
                } else {
                    self.teleporter_dest = Some(idx);
                    self.picking_destination = false;
                    self.status = format!("Teleporter destination: {gx}, {gy}");
                }
            }
            return;
        }

        self.apply_brush(gx, gy, fx, fy, dt, input);
    }

    fn apply_brush(&mut self, gx: usize, gy: usize, fx: f32, fy: f32, dt: f32, input: MouseInput) {
        if !(input.left || input.right || input.middle) {
            return;
        }
        if input.left && input.ctrl && self.selected_cell_type == CellType::Teleporter  && self.current_cell_effect.is_none() {
            self.status = "Teleporter: pick a destination first".into();
            return;
        }

        let thermal = match self.thermal_mode {
            ThermalMode::Passive => None,
            ThermalMode::Isothermal => Some(ThermalCell::Isothermal(self.thermal_temp)),
            ThermalMode::Heater => Some(ThermalCell::Heater(self.thermal_power)),
        };
        let fluid = &mut self.fluid;
        let (w, h) = (fluid.width, fluid.height);

        if gx <= 1 || gx >= w - 1 || gy <= 1 || gy >= h - 1 {
            return;
        }

        let precision = self.precision as f32;
        let radius = self.mouse_radius as f32;
        let mouse_mode = self.mouse_mode;
        let r = (radius / precision).ceil() as isize;
        let brush_t = self.brush_temperature;
        let heating = input.shift && !input.ctrl;
        // vitesse du curseur : pixels/s -> cellules/s (= m/s ÷ dx)
        let (cursor_x, cursor_y) = (fx / precision, fy / precision);
        let push = 1.0 - (-PUSH_RATE * dt).exp();

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
                            fluid.set_thermal(x, y, thermal);
                        } else {
                            fluid.set_cell(x, y, self.selected_cell_type, self.current_cell_effect);
                        }
                    } else if fluid.solids[idx] == CellType::Fluid {
                        fluid.tracer[idx] = (fluid.tracer[idx] + 2.0 * dt).min(1.0);
                    }
                }

                if input.right && (cursor_x != 0.0 || cursor_y != 0.0) {
                    // la vitesse du fluide rejoint celle du curseur (indépendant du framerate)
                    fluid.velocity_x[idx] += (cursor_x - fluid.velocity_x[idx]) * push;
                    fluid.velocity_y[idx] += (cursor_y - fluid.velocity_y[idx]) * push;
                }

                if input.middle {
                    if heating {
                        // source de chaleur (W/m³) : passe par l'équation d'énergie
                        fluid.heat_towards(idx, brush_t, HEAT_RATE);
                    } else if input.ctrl {
                        fluid.set_cell(x, y, CellType::Fluid, None);
                    } else {
                        fluid.tracer[idx] = (fluid.tracer[idx] - 2.0 * dt).max(0.0);
                    }
                }
            }
        }
    }

    fn simulate(&mut self, dt: f32) {
        if !self.paused || self.step_frame > 0 {
            self.fluid.step(dt.min(MAX_DT), self.pressure_iters);
            self.step_frame = self.step_frame.saturating_sub(1);
        }
    }

    fn update_stats(&mut self, dt: f32) {
        let f = &self.fluid;
        let (w, h) = (f.width, f.height);
        let s = &f.solids;
        let dx = f.dx;

        let lo = f.left_boundary.boundary_type == BoundaryType::Outlet;
        let ro = f.right_boundary.boundary_type == BoundaryType::Outlet;
        let to = f.top_boundary.boundary_type == BoundaryType::Outlet;
        let bo = f.bottom_boundary.boundary_type == BoundaryType::Outlet;
        let wall = |i: usize| s[i] == CellType::Wall;

        let open_u = |x: usize, y: usize| -> bool {
            if y == 0 || y >= h - 1 || x >= w - 1 { return false; }
            let i = x + y * w;
            if x == 0 { lo && !wall(i + 1) }
            else if x == w - 2 { ro && !wall(i) }
            else { !wall(i) && !wall(i + 1) }
        };
        let open_v = |x: usize, y: usize| -> bool {
            if y >= h - 1 || x == 0 || x >= w - 1 { return false; }
            let i = x + y * w;
            if y == 0 { to && !wall(i + w) }
            else if y == h - 2 { bo && !wall(i) }
            else { !wall(i) && !wall(i + w) }
        };

        // (max s², Σ ρ·s², Σ (div−S)², Σ |v|², nb cellules actives, max |div−S|)
        let (speed2_max, rho_s2, resid2_sum, v2_sum, count, residual) = (0..h)
            .into_par_iter()
            .map(|y| {
                let (mut m, mut e) = (0.0f32, 0.0f32);
                let (mut r2, mut v2, mut n_cells, mut res) = (0.0f64, 0.0f64, 0u64, 0.0f32);
                for x in 0..w {
                    let i = x + y * w;
                    let (vx, vy) = FluidSim::cell_velocity_from(&f.velocity_x, &f.velocity_y, w, h, x, y);
                    let s2 = vx * vx + vy * vy;
                    m = m.max(s2);
                    if s[i] != CellType::Wall {
                        e += f.rho[i] * s2;
                    }

                    let interior = x >= 1 && x < w - 1 && y >= 1 && y < h - 1;
                    if !interior || s[i] == CellType::Wall || s[i] == CellType::Sink {
                        continue;
                    }
                    let r = f.divergence[i] - f.source[i]; // résidu du Poisson : div u − S
                    r2 += (r * r) as f64;
                    v2 += s2 as f64;
                    n_cells += 1;

                    let n = open_u(x, y) as u8 + open_u(x - 1, y) as u8
                        + open_v(x, y) as u8 + open_v(x, y - 1) as u8;
                    if n > 0 {
                        res = res.max(r.abs());
                    }
                }
                (m, e, r2, v2, n_cells, res)
            })
            .reduce(|| (0.0, 0.0, 0.0, 0.0, 0, 0.0),
                    |a, b| (a.0.max(b.0), a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4, a.5.max(b.5)));

        // plages de rho et T sur les cellules intérieures non solides
        let rho_amb = f.material.rho_ambient();
        let (rho_min, rho_max, t_min, t_max) = (1..h - 1)
            .into_par_iter()
            .map(|y| {
                let (mut rmin, mut rmax) = (f32::INFINITY, f32::NEG_INFINITY);
                let (mut tmin, mut tmax) = (f32::INFINITY, f32::NEG_INFINITY);
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] == CellType::Wall {
                        continue;
                    }
                    rmin = rmin.min(f.rho[i]);
                    rmax = rmax.max(f.rho[i]);
                    tmin = tmin.min(f.temperature[i]);
                    tmax = tmax.max(f.temperature[i]);
                }
                (rmin, rmax, tmin, tmax)
            })
            .reduce(|| (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY),
                    |a, b| (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)));
        let fin = |v: f32| if v.is_finite() { v } else { 0.0 };
        let (rho_min, rho_max, t_min, t_max) = (fin(rho_min), fin(rho_max), fin(t_min), fin(t_max));

        let speed_max = speed2_max.sqrt(); // cellules/s
        let n = count.max(1) as f64;
        let pressure_max = f.pressure.par_iter().map(|v| v.abs()).reduce(|| 0.0, f32::max);

        self.stats = Stats {
            pressure_max,
            pressure_dyn_pa: pressure_max * f.pressure_scale(),
            divergence_max: f.divergence.par_iter().map(|v| v.abs()).reduce(|| 0.0, f32::max),
            velocity_max: speed_max,
            velocity_max_ms: speed_max * dx,
            total_tracer: f.total_tracer(),
            total_mass: f.total_mass(),
            kinetic_energy: 0.5 * rho_s2 * dx.powi(4), // ½·Σ ρ·v²·dx² (aire d'une cellule dx²), v en m/s
            pressure_residual: residual,
            divergence_rms: (resid2_sum / n).sqrt() as f32,
            velocity_rms: ((v2_sum / n).sqrt() as f32) * dx,
            cfl: speed_max * dt, // dx = 1 cellule (l'advection recule de v*dt en unités de cellule)
            p_th: f.p_th,
            dpdt: f.dpdt,
            rho_min,
            rho_max,
            rho_dev_max: (rho_max - rho_amb).abs().max((rho_min - rho_amb).abs()),
            t_min,
            t_max,
        };
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let w = self.fluid.width;
        let h = self.fluid.height;
        self.rgba.resize(w * h * 4, 255);

        let mode = self.display_mode;
        let max_color = self.max_color;
        let solid_color: u32 = if self.inverse_solid_color { 0xffffff } else { 0x000000 };
        let Stats { pressure_max, divergence_max, velocity_max, rho_dev_max, .. } = self.stats;

        let tracer = &self.fluid.tracer[..];
        let rho = &self.fluid.rho[..];
        let rho_amb = self.fluid.material.rho_ambient();
        let t_ref = self.fluid.material.t_ref;
        let t_scale = self.temperature_scale.max(1e-3);
        let pressure = &self.fluid.pressure[..];
        let divergence = &self.fluid.divergence[..];
        let velocity_x = &self.fluid.velocity_x[..];
        let velocity_y = &self.fluid.velocity_y[..];
        let grid_h = h;
        let solids = &self.fluid.solids[..];
        let show_effects = self.show_cell_effects;
        let temperature = &self.fluid.temperature[..];
        let thermal_kind = &self.fluid.thermal_kind[..];

        self.rgba
            .par_chunks_exact_mut(w * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, px) in row.chunks_exact_mut(4).enumerate() {
                    let idx = x + y * w;

                    let kind = match solids[idx] {
                        k if k.is_special() && !show_effects => CellType::Fluid,
                        k => k,
                    };
                    let color: u32 = match kind {
                        CellType::Fluid => {
                            match mode {
                                DisplayMode::Tracer => {
                                    let d = tracer[idx].clamp(0.0, 1.0);
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
                                DisplayMode::Temperature => temperature_color((temperature[idx] - t_ref) / t_scale),
                                DisplayMode::Rho => signed_color(rho[idx] - rho_amb, rho_dev_max),
                            }
                        }
                        CellType::Wall => match (mode, thermal_kind[idx]) {
                            (DisplayMode::Temperature, _) => temperature_color((temperature[idx] - t_ref) / t_scale),
                            (_, 1) if show_effects => 0x2060d0, // isotherme : bleu
                            (_, 2) if show_effects => 0xd08020, // chauffage : orange
                            (DisplayMode::Tracer, _) => 0x000000,
                            _ => solid_color,
                        }
                        CellType::Sink => { //red
                            0xff0000
                        }
                        CellType::Emitter => { //green
                            0x00ff00
                        }
                        CellType::Accelerator => { //violet
                            0xff00ff
                        }
                        CellType::Teleporter => {
                            0x0000ff
                        }
                    };

                    px[0] = (color >> 16) as u8;
                    px[1] = (color >> 8) as u8;
                    px[2] = color as u8;
                    px[3] = 255;
                }
            });

        if show_effects {
            let mark = |rgba: &mut Vec<u8>, idx: usize| {
                let p = idx * 4;
                if p + 2 < rgba.len() && solids[idx] == CellType::Fluid {
                    rgba[p] = 0x00;
                    rgba[p + 1] = 0xff;
                    rgba[p + 2] = 0xff;
                }
            };
            for &(_, effect) in &self.fluid.special_cells {
                if let CellEffect::Teleporter { dest_idx } = effect {
                    mark(&mut self.rgba, dest_idx);
                }
            }
            if self.selected_cell_type == CellType::Teleporter {
                if let Some(d) = self.teleporter_dest {
                    mark(&mut self.rgba, d);
                }
            }
        }
        let image = ColorImage::from_rgba_unmultiplied([w, h], &self.rgba);
        match &mut self.texture {
            Some(tex) => tex.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("fluid", image, TextureOptions::NEAREST)),
        }
    }
}