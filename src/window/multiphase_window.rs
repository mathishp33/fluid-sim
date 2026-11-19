//! Fenêtre egui du solveur multiphasique (CPU f64 ou GPU f32). Toute la logique (scénarios, pinceaux, rendu, stats) est dans
//! `simulation::mp_view` (testée) ; ce fichier ne fait que l'interface. Mêmes conventions que `main_window.rs`.
//!
//! ATTENTION : ce fichier n'a pas pu être compilé ici (egui/rfd absents de l'environnement hors ligne) ; il n'utilise que des appels egui
//! déjà présents dans `main_window.rs`. `mp_view.rs`, lui, est compilé et testé (tests `mp_view::tests`).

use std::time::{Duration, Instant};

use egui::{Color32, ColorImage, PointerButton, Sense, TextureHandle, TextureOptions};

use crate::simulation::fluid::{BoundaryType};
use crate::simulation::gpu_multiphase::MultiphaseGpu;
use crate::simulation::mp_view::{
    all_tools, apply_tool, brush_cells, build_scenario, cell_info, compute_stats, render_rgba, tool_name, MpDisplay, MpScenario, MpStats, MpTool,
    RenderOpts,
};
use crate::simulation::multiphase::{Bc, MultiphaseSim};
use crate::window::main_window::WindowAction;
use crate::window::starting_window::SimulationSettings;

const P_ATM: f64 = 101325.0;
const FRAME_BUDGET: Duration = Duration::from_millis(14);

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Cpu,
    Gpu,
}

pub struct MultiphaseWindow {
    sim: MultiphaseSim,
    gpu: Option<MultiphaseGpu>,
    backend: Backend,
    settings: SimulationSettings,
    scenario: MpScenario,

    texture: Option<TextureHandle>,
    rgba: Vec<u8>,
    display: MpDisplay,
    t_scale: f64,
    p_scale: f64,

    paused: bool,
    step_frame: usize,
    /// vitesse de simulation visée : temps simulé / temps réel
    speed: f64,
    achieved: f64,
    last_frame: Instant,
    last_mouse: Option<(f32, f32)>,
    hovered: Option<(usize, usize)>,

    tool: MpTool,
    circle: bool,
    radius: usize,
    brush_temp: f64,

    stats: MpStats,
    stats_age: usize,
    status: String,
    show_help: bool,
    fps: f32,
    frames: usize,
    last_fps: Instant,
}

impl MultiphaseWindow {
    pub fn new(settings: &SimulationSettings) -> Self {
        let scenario = MpScenario::from_index(settings.mp_scenario);
        let mut w = Self {
            sim: Self::make(settings, scenario),
            gpu: None,
            backend: Backend::Cpu,
            settings: settings.clone(),
            scenario,
            texture: None,
            rgba: vec![],
            display: MpDisplay::Phases,
            t_scale: 100.0,
            p_scale: 0.0,
            paused: false,
            step_frame: 0,
            speed: 1.0,
            achieved: 0.0,
            last_frame: Instant::now(),
            last_mouse: None,
            hovered: None,
            tool: MpTool::Phase(crate::simulation::mp_view::EAU),
            circle: true,
            radius: 3,
            brush_temp: 293.15,
            stats: MpStats::default(),
            stats_age: 0,
            status: format!("Scénario : {}", scenario.name()),
            show_help: false,
            fps: 0.0,
            frames: 0,
            last_fps: Instant::now(),
        };
        w.stats = compute_stats(&w.sim);
        w
    }

    fn make(settings: &SimulationSettings, sc: MpScenario) -> MultiphaseSim {
        let p = settings.precision.max(1);
        let (nx, ny) = ((settings.width / p).max(16), (settings.height / p).max(16));
        let g = if settings.gravity > 0.0 { settings.gravity as f64 } else { 9.81 };
        let mut s = build_scenario(sc, nx, ny, settings.dx as f64, g);
        // un bord réglé sur « Outlet » dans l'écran de démarrage devient une ouverture à 1 atm (gauche, droite, bas, haut)
        let sides = [&settings.left_boundary, &settings.right_boundary, &settings.bottom_boundary, &settings.top_boundary];
        for (k, b) in sides.iter().enumerate() {
            if b.boundary_type == BoundaryType::Outlet {
                s.bc[k] = Bc::Open { p: P_ATM };
            }
        }
        s
    }

    fn reset(&mut self) {
        self.sim = Self::make(&self.settings, self.scenario);
        self.gpu = None;
        self.backend = Backend::Cpu;
        self.stats = compute_stats(&self.sim);
        self.status = format!("Scénario : {}", self.scenario.name());
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> WindowAction {
        let now = Instant::now();
        let dt_wall = (now - self.last_frame).as_secs_f64().clamp(1e-4, 0.1);
        self.last_frame = now;
        self.frames += 1;
        let el = now.duration_since(self.last_fps).as_secs_f32();
        if el >= 0.5 {
            self.fps = self.frames as f32 / el;
            self.frames = 0;
            self.last_fps = now;
        }

        let mut action = WindowAction::None;
        self.shortcuts(ctx, &mut action);
        self.top_bar(ctx, &mut action);
        egui::TopBottomPanel::bottom("mp_status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("LMB : outil | RMB : pousser | MMB : effacer le solide | Shift+LMB : chauffer");
                });
            });
        });
        self.controls(ctx);
        self.info(ctx);
        self.canvas(ctx, dt_wall);
        self.help(ctx);
        ctx.request_repaint();
        action
    }

    fn shortcuts(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let pressed = |k: egui::Key| {
            ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { key, pressed: true, repeat: false, .. } if *key == k)))
        };
        if pressed(egui::Key::Space) {
            self.paused = !self.paused;
        }
        if self.paused && pressed(egui::Key::ArrowRight) {
            self.step_frame = 1;
        }
        if self.paused && pressed(egui::Key::ArrowUp) {
            self.step_frame = 10;
        }
        if pressed(egui::Key::V) {
            self.display = self.display.next();
        }
        if pressed(egui::Key::M) {
            self.circle = !self.circle;
        }
        if pressed(egui::Key::R) {
            self.reset();
        }
        if pressed(egui::Key::Escape) {
            *action = WindowAction::BackToSettings;
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        egui::TopBottomPanel::top("mp_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let label = if self.paused { "Reprendre (Espace)" } else { "Pause (Espace)" };
                if ui.button(label).clicked() {
                    self.paused = !self.paused;
                }
                if ui.add_enabled(self.paused, egui::Button::new("Pas (→)")).clicked() {
                    self.step_frame = 1;
                }
                if ui.button("Recommencer (R)").clicked() {
                    self.reset();
                }
                ui.separator();
                if ui.button("Aide").clicked() {
                    self.show_help = !self.show_help;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Retour aux réglages (Échap)").clicked() {
                        *action = WindowAction::BackToSettings;
                    }
                });
            });
        });
    }

    fn controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("mp_controls").default_width(260.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Multiphasique");
                ui.separator();

                let mut chosen = None;
                egui::ComboBox::from_label("Scénario préréglé").selected_text(self.scenario.name()).show_ui(ui, |ui| {
                    for sc in MpScenario::ALL {
                        if ui.selectable_label(sc == self.scenario, sc.name()).clicked() {
                            chosen = Some(sc);
                        }
                    }
                });
                if let Some(sc) = chosen {
                    self.scenario = sc;
                    self.reset();
                }

                ui.separator();
                ui.label("Affichage (V)");
                for m in MpDisplay::ALL {
                    ui.selectable_value(&mut self.display, m, m.name());
                }
                if self.display == MpDisplay::Temperature {
                    ui.add(egui::Slider::new(&mut self.t_scale, 1.0..=500.0).logarithmic(true).text("Échelle ΔT (K)"));
                }
                if self.display == MpDisplay::Pressure {
                    ui.add(egui::Slider::new(&mut self.p_scale, 0.0..=1e6).text("Échelle p (Pa, 0 = auto)"));
                }

                ui.separator();
                ui.heading("Pinceau");
                for t in all_tools(&self.sim.reg) {
                    let name = tool_name(&self.sim.reg, t);
                    ui.selectable_value(&mut self.tool, t, name);
                }
                ui.checkbox(&mut self.circle, "Rond (M)");
                ui.add(egui::Slider::new(&mut self.radius, 1..=30).text("Rayon (cellules)"));
                ui.add(egui::Slider::new(&mut self.brush_temp, 200.0..=900.0).text("T pinceau (K)"));
                ui.weak("T pinceau : température des phases ajoutées, des solides, et cible du chauffage");

                ui.separator();
                ui.heading("Simulation");
                ui.add(egui::Slider::new(&mut self.speed, 0.001..=10.0).logarithmic(true).text("Vitesse (×temps réel)"));
                ui.horizontal(|ui| {
                    ui.label("Calcul");
                    let cur = self.backend;
                    let mut b = cur;
                    ui.selectable_value(&mut b, Backend::Cpu, "CPU f64");
                    ui.selectable_value(&mut b, Backend::Gpu, "GPU f32");
                    if b != cur {
                        self.set_backend(b);
                    }
                });
                if self.backend == Backend::Gpu {
                    if let Some(g) = &mut self.gpu {
                        let mut it = g.pcg_iter as usize;
                        ui.add(egui::Slider::new(&mut it, 20..=2000).text("Itérations PCG (GPU)"));
                        g.pcg_iter = it as u32;
                    }
                }
                ui.add(egui::Slider::new(&mut self.sim.cfl, 0.1..=0.9).text("CFL"));
                ui.add(egui::Slider::new(&mut self.sim.dt_max, 1e-5..=2e-2).logarithmic(true).text("dt max (s)"));
                let mut g = -self.sim.gravity[1];
                if ui.add(egui::Slider::new(&mut g, 0.0..=30.0).text("Gravité (m/s²)")).changed() {
                    self.sim.gravity = [0.0, -g];
                }
                ui.checkbox(&mut self.sim.surface_tension, "Tension de surface");
                ui.checkbox(&mut self.sim.viscous, "Viscosité");
                ui.checkbox(&mut self.sim.conduction, "Conduction thermique");
                ui.checkbox(&mut self.sim.phase_change, "Changement de phase (ébullition / cavitation)");
                ui.add_enabled(self.backend == Backend::Cpu, egui::Checkbox::new(&mut self.sim.conservative_momentum, "Quantité de mouvement conservative (CPU)"));
                ui.weak("GPU : l'advection est toujours conservative.");

                ui.separator();
                ui.heading("Bords");
                let names = ["Gauche", "Droite", "Bas", "Haut"];
                for k in 0..4 {
                    let mut open = matches!(self.sim.bc[k], Bc::Open { .. });
                    if ui.checkbox(&mut open, format!("{} ouvert (1 atm)", names[k])).changed() {
                        self.sim.bc[k] = if open { Bc::Open { p: P_ATM } } else { Bc::Wall };
                    }
                }
            });
        });
    }

    fn set_backend(&mut self, b: Backend) {
        match b {
            Backend::Cpu => {
                self.gpu = None;
                self.backend = Backend::Cpu;
                self.status = "Calcul CPU (f64)".into();
            }
            Backend::Gpu => match MultiphaseGpu::try_new(&self.sim) {
                Some(g) => {
                    self.gpu = Some(g);
                    self.backend = Backend::Gpu;
                    self.status = "Calcul GPU (f32)".into();
                }
                None => {
                    self.backend = Backend::Cpu;
                    self.status = "GPU indisponible (adaptateur absent ou grille trop grande pour un seul buffer)".into();
                }
            },
        }
    }

    fn info(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("mp_info").default_width(260.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Info");
                ui.separator();
                let st = &self.stats;
                egui::Grid::new("mp_info_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                    let mut row = |a: &str, b: String| {
                        ui.label(a);
                        ui.label(b);
                        ui.end_row();
                    };
                    row("FPS", format!("{:.1}", self.fps));
                    row("État", if self.paused { "PAUSE".into() } else { "EN COURS".to_string() });
                    row("Calcul", if self.backend == Backend::Gpu { "GPU f32".into() } else { "CPU f64".to_string() });
                    row("Grille", format!("{} × {}", self.sim.nx, self.sim.ny));
                    row("Domaine (m)", format!("{:.3} × {:.3}", self.sim.nx as f64 * self.sim.dx, self.sim.ny as f64 * self.sim.dx));
                    row("Temps simulé (s)", format!("{:.4}", st.time));
                    row("Pas", format!("{}", st.steps));
                    row("dt (s)", format!("{:.2e}", st.dt));
                    row("Vitesse atteinte", format!("×{:.3}", self.achieved));
                    row("Itérations pression", format!("{}", st.pressure_iters));
                    row("Vitesse max (m/s)", format!("{:.3}", st.speed_max));
                    row("p min / max (Pa)", format!("{:.0} / {:.0}", st.p_min, st.p_max));
                    row("T min / max (K)", format!("{:.2} / {:.2}", st.t_min, st.t_max));
                    row("Énergie cinétique (J/m)", format!("{:.4e}", st.kinetic));
                    row("Énergie interne (J/m)", format!("{:.6e}", st.internal));
                });
                ui.separator();
                ui.label("Masse (kg/m) · aire (m²)");
                egui::Grid::new("mp_mass_grid").num_columns(3).spacing([12.0, 4.0]).show(ui, |ui| {
                    for (n, m, a) in &st.phases {
                        ui.label(n);
                        ui.label(format!("{:.5e}", m));
                        ui.label(format!("{:.4e}", a));
                        ui.end_row();
                    }
                });
                ui.separator();
                ui.heading("Cellule sous le curseur");
                match self.hovered {
                    Some((i, j)) => {
                        egui::Grid::new("mp_cell_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                            for (a, b) in cell_info(&self.sim, i, j) {
                                ui.label(a);
                                ui.label(b);
                                ui.end_row();
                            }
                        });
                    }
                    None => {
                        ui.weak("—");
                    }
                }
            });
        });
    }

    fn help(&mut self, ctx: &egui::Context) {
        egui::Window::new("Raccourcis").open(&mut self.show_help).resizable(false).show(ctx, |ui| {
            egui::Grid::new("mp_help").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                for (k, v) in [
                    ("Espace", "Pause / reprise"),
                    ("→ / ↑", "1 / 10 pas (en pause)"),
                    ("V", "Mode d'affichage suivant"),
                    ("M", "Pinceau rond / carré"),
                    ("R", "Recommencer le scénario"),
                    ("Échap", "Retour aux réglages"),
                    ("Clic gauche", "Applique l'outil (phase, solide, chauffage, poussée)"),
                    ("Maj + clic gauche", "Chauffer / refroidir vers T pinceau"),
                    ("Clic droit", "Pousse le fluide avec le curseur"),
                    ("Clic milieu", "Efface le solide"),
                ] {
                    ui.strong(k);
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
    }

    fn canvas(&mut self, ctx: &egui::Context, dt_wall: f64) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (outer, response) = ui.allocate_exact_size(avail, Sense::click_and_drag());
            if outer.width() < 1.0 || outer.height() < 1.0 {
                return;
            }
            let (gw, gh) = (self.sim.nx as f32, self.sim.ny as f32);
            let scale = (outer.width() / gw).min(outer.height() / gh);
            let rect = egui::Rect::from_center_size(outer.center(), egui::vec2(gw * scale, gh * scale));

            self.mouse(ctx, &response, rect, dt_wall);
            self.simulate(dt_wall);
            self.stats_age += 1;
            if self.stats_age >= 5 || self.paused {
                self.stats = compute_stats(&self.sim);
                self.stats_age = 0;
                if !self.stats.finite && !self.paused {
                    self.paused = true;
                    self.status = "Divergence numérique : simulation mise en pause (réduire dt max / CFL, ou recommencer)".into();
                }
            }
            self.update_texture(ctx);

            ui.painter().rect_filled(outer, 0.0, Color32::BLACK);
            if let Some(tex) = &self.texture {
                ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            }
        });
    }

    fn mouse(&mut self, ctx: &egui::Context, response: &egui::Response, rect: egui::Rect, dt: f64) {
        let (pos, left, right, middle, shift) = ctx.input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.button_down(PointerButton::Primary),
                i.pointer.button_down(PointerButton::Secondary),
                i.pointer.button_down(PointerButton::Middle),
                i.modifiers.shift,
            )
        });
        let over = match pos {
            Some(p) if response.contains_pointer() && rect.contains(p) => Some(p),
            _ => None,
        };
        let Some(p) = over else {
            self.hovered = None;
            self.last_mouse = None;
            return;
        };
        let (nx, ny) = (self.sim.nx, self.sim.ny);
        let u = ((p.x - rect.min.x) / rect.width()).clamp(0.0, 0.9999);
        let v = ((p.y - rect.min.y) / rect.height()).clamp(0.0, 0.9999);
        let ci = (u * nx as f32) as usize;
        let cj = ny - 1 - (v * ny as f32) as usize; // l'axe j croît vers le haut
        self.hovered = Some((ci, cj));

        // vitesse du curseur en m/s (y vers le haut)
        let (mx, my) = (u * nx as f32 * self.sim.dx as f32, (1.0 - v) * ny as f32 * self.sim.dx as f32);
        let cursor = match self.last_mouse {
            Some((lx, ly)) => (((mx - lx) as f64) / dt, ((my - ly) as f64) / dt),
            None => (0.0, 0.0),
        };
        self.last_mouse = Some((mx, my));

        if !(left || right || middle) {
            return;
        }
        let cells = brush_cells(nx, ny, ci, cj, self.radius, self.circle);
        let mut edited = false;
        if left && shift {
            apply_tool(&mut self.sim, MpTool::Heat, &cells, self.brush_temp, cursor, dt);
            edited = true;
        } else if left {
            apply_tool(&mut self.sim, self.tool, &cells, self.brush_temp, cursor, dt);
            edited = true;
        }
        if right {
            apply_tool(&mut self.sim, MpTool::Push, &cells, self.brush_temp, cursor, dt);
            edited = true;
        }
        if middle {
            apply_tool(&mut self.sim, MpTool::EraseSolid, &cells, self.brush_temp, cursor, dt);
            edited = true;
        }
        if edited {
            if let Some(g) = &mut self.gpu {
                g.upload(&self.sim); // l'état CPU est relu à chaque pas : on renvoie l'état édité
            }
        }
    }

    fn simulate(&mut self, dt_wall: f64) {
        if self.paused && self.step_frame == 0 {
            self.achieved = 0.0;
            return;
        }
        let start = Instant::now();
        let target = dt_wall * self.speed;
        let t0 = self.sim.time;
        loop {
            match (&mut self.gpu, self.backend) {
                (Some(g), Backend::Gpu) => {
                    g.step(&mut self.sim);
                }
                _ => {
                    self.sim.step();
                }
            }
            if self.paused {
                self.step_frame = self.step_frame.saturating_sub(1);
                if self.step_frame == 0 {
                    break;
                }
            }
            if self.sim.time - t0 >= target || start.elapsed() >= FRAME_BUDGET {
                break;
            }
        }
        self.achieved = (self.sim.time - t0) / dt_wall;
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let o = RenderOpts { display: self.display, t_scale: self.t_scale, p_scale: self.p_scale };
        render_rgba(&self.sim, o, &mut self.rgba);
        let image = ColorImage::from_rgba_unmultiplied([self.sim.nx, self.sim.ny], &self.rgba);
        match &mut self.texture {
            Some(tex) => tex.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("multiphase", image, TextureOptions::NEAREST)),
        }
    }
}

/// Fenêtre de simulation : mono-phase (FluidWindow) ou multiphasique selon `settings.multiphase`.
/// Dans le code qui possède la fenêtre (main.rs / app), remplacez `FluidWindow` par `SimWindow`.
pub enum SimWindow {
    Single(crate::window::main_window::FluidWindow),
    Multi(Box<MultiphaseWindow>),
}

impl SimWindow {
    pub fn new(settings: &SimulationSettings) -> Self {
        if settings.multiphase {
            SimWindow::Multi(Box::new(MultiphaseWindow::new(settings)))
        } else {
            SimWindow::Single(crate::window::main_window::FluidWindow::new(settings))
        }
    }
    pub fn ui(&mut self, ctx: &egui::Context) -> WindowAction {
        match self {
            SimWindow::Single(w) => w.ui(ctx),
            SimWindow::Multi(w) => w.ui(ctx),
        }
    }
}
