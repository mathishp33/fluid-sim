//! Fenêtre egui du solveur multiphasique (CPU f64 ou GPU f32). Toute la logique (scénarios, pinceaux, rendu, stats, scènes) est dans
//! `simulation::mp_view` / `mp_cells` (testées) ; ce fichier ne fait que l'interface. Mêmes conventions que `main_window.rs` :
//!
//!  * clic gauche : outil courant ; Ctrl + clic gauche : pose le type de cellule choisi (solide + comportement thermique, puits, émetteur,
//!    accélérateur, téléporteur) ; clic droit : pousse le fluide ; clic milieu (ou Ctrl + milieu) : efface solide et effet ;
//!    Maj + clic gauche ou Maj + clic milieu : chauffe / refroidit vers la T du pinceau.
//!  * raccourcis : Espace, → / ↑, S / L (sauver / charger la scène), I (solides blancs), V, M, C / Maj+C (rayon), R (reset flow),
//!    Maj+R (recommencer le scénario), Échap.

use std::time::{Duration, Instant};

use egui::{Color32, ColorImage, PointerButton, Sense, TextureHandle, TextureOptions};

use crate::simulation::fluid::{Boundary, BoundaryType};
use crate::simulation::gpu_multiphase::{gpu_unsupported_reason, MultiphaseGpu};
use crate::simulation::mp_cells::{MpEffect, MpInlet, MpThermal};
use crate::simulation::mp_materials::{MaterialLibrary, Roles};
use crate::simulation::mp_view::{
    apply_tool_with, brush_cells, build_scenario_with, cell_info, compute_stats, render_rgba, reset_flow, reset_temperature, scene_from_json,
    scene_to_json, sim_tools, tool_label, tool_name, MpDisplay, MpScenario, MpStats, MpTool, RenderOpts, ToolOpts, T_IGNITION,
};
use crate::simulation::multiphase::{Bc, MultiphaseSim};
use crate::window::main_window::WindowAction;
use crate::window::material_editor::MaterialEditor;
use crate::window::starting_window::{boundary_ui, save_settings, SimulationSettings};

const P_ATM: f64 = 101325.0;
const FRAME_BUDGET: Duration = Duration::from_millis(14);
/// rayons de pinceau (cellules) parcourus par la touche C
const RADII: &[usize] = &[1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 60];

fn cycle_radius(current: usize, reverse: bool) -> usize {
    if reverse {
        RADII.iter().rev().copied().find(|&r| r < current).unwrap_or(RADII[RADII.len() - 1])
    } else {
        RADII.iter().copied().find(|&r| r > current).unwrap_or(RADII[0])
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Cpu,
    Gpu,
}

/// Type de cellule posé par Ctrl + clic gauche (équivalent de `CellType::PAINTABLE` du mono-fluide)
#[derive(Clone, Copy, PartialEq)]
enum CellKind {
    Wall,
    Sink,
    Emitter,
    Accelerator,
    Teleporter,
}

impl CellKind {
    const ALL: [CellKind; 5] = [CellKind::Wall, CellKind::Sink, CellKind::Emitter, CellKind::Accelerator, CellKind::Teleporter];
    fn name(self) -> &'static str {
        match self {
            CellKind::Wall => "Solide (mur)",
            CellKind::Sink => "Puits",
            CellKind::Emitter => "Émetteur",
            CellKind::Accelerator => "Accélérateur",
            CellKind::Teleporter => "Téléporteur",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ThermalMode {
    Passive,
    Isothermal,
    Heater,
}

pub struct MultiphaseWindow {
    sim: MultiphaseSim,
    gpu: Option<MultiphaseGpu>,
    backend: Backend,
    settings: SimulationSettings,
    scenario: MpScenario,

    // matériaux : copie de travail éditable (appliquée au redémarrage)
    lib: MaterialLibrary,
    editor: MaterialEditor,
    lib_dirty: bool,
    build_warnings: Vec<String>,

    texture: Option<TextureHandle>,
    rgba: Vec<u8>,
    display: MpDisplay,
    t_scale: f64,
    p_scale: f64,
    show_effects: bool,
    solids_white: bool,

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
    /// température du noyau de flamme créé par l'outil « Allumer »
    ignite_temp: f64,

    // cellules spéciales / solides thermiques (Ctrl + clic gauche)
    cell_kind: CellKind,
    sel_solid: usize,
    thermal_mode: ThermalMode,
    thermal_temp: f64,
    thermal_power: f64,
    sink_rate: f64,
    emit_vx: f64,
    emit_vy: f64,
    emit_phase: usize,
    emit_temp: f64,
    acc_ax: f64,
    acc_ay: f64,
    tp_dest: Option<usize>,
    tp_speed: f64,
    picking_destination: bool,

    // bords modifiables en direct (gauche, droite, bas, haut)
    boundaries: [Boundary; 4],

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
        let mut settings = settings.clone();
        let (sim, status) = match Self::make(&settings, &settings.materials, scenario) {
            Ok(s) => (s, format!("Scénario : {}", scenario.name())),
            Err(e) => {
                // matériaux incompatibles avec le scénario : cuve vide ; si même cela échoue, matériaux par défaut
                match Self::make(&settings, &settings.materials, MpScenario::Empty) {
                    Ok(s) => (s, format!("{e} — cuve vide à la place")),
                    Err(_) => {
                        settings.materials = MaterialLibrary::default();
                        (Self::make(&settings, &settings.materials, MpScenario::Empty).expect("matériaux par défaut"), format!("{e} — matériaux par défaut rétablis"))
                    }
                }
            }
        };
        let lib = settings.materials.clone();
        let roles = Roles::resolve(&sim.reg, sim.ambient_phase);
        let boundaries = Self::boundaries_for(&settings, &sim);
        let tool = MpTool::Phase(roles.water.unwrap_or(0));
        let mut w = Self {
            sim,
            gpu: None,
            backend: Backend::Cpu,
            settings,
            scenario,
            lib,
            editor: MaterialEditor::default(),
            lib_dirty: false,
            build_warnings: vec![],
            texture: None,
            rgba: vec![],
            display: MpDisplay::Phases,
            t_scale: 100.0,
            p_scale: 0.0,
            show_effects: true,
            solids_white: false,
            paused: false,
            step_frame: 0,
            speed: 1.0,
            achieved: 0.0,
            last_frame: Instant::now(),
            last_mouse: None,
            hovered: None,
            tool,
            circle: true,
            radius: 3,
            brush_temp: 293.15,
            ignite_temp: T_IGNITION,
            cell_kind: CellKind::Wall,
            sel_solid: roles.plate.unwrap_or(0),
            thermal_mode: ThermalMode::Passive,
            thermal_temp: 400.0,
            thermal_power: 1e7,
            sink_rate: 0.2,
            emit_vx: 0.0,
            emit_vy: 1.0,
            emit_phase: roles.water.unwrap_or(0),
            emit_temp: 293.15,
            acc_ax: 5.0,
            acc_ay: 0.0,
            tp_dest: None,
            tp_speed: 0.5,
            picking_destination: false,
            boundaries,
            stats: MpStats::default(),
            stats_age: 0,
            status,
            show_help: false,
            fps: 0.0,
            frames: 0,
            last_fps: Instant::now(),
        };
        w.refresh_warnings();
        w.apply_boundaries();
        w.stats = compute_stats(&w.sim);
        w
    }

    fn make(settings: &SimulationSettings, lib: &MaterialLibrary, sc: MpScenario) -> Result<MultiphaseSim, String> {
        let p = settings.precision.max(1);
        let (nx, ny) = ((settings.width / p).max(16), (settings.height / p).max(16));
        let g = if settings.gravity > 0.0 { settings.gravity as f64 } else { 9.81 };
        build_scenario_with(lib, sc, nx, ny, settings.dx as f64, g)
    }

    /// bords de l'écran de démarrage, cohérents avec les ouvertures imposées par le scénario (haut ouvert de la cuve vide, de l'ébullition…)
    fn boundaries_for(settings: &SimulationSettings, sim: &MultiphaseSim) -> [Boundary; 4] {
        let mut b = [settings.left_boundary, settings.right_boundary, settings.bottom_boundary, settings.top_boundary];
        for k in 0..4 {
            if matches!(sim.bc[k], Bc::Open { .. }) && b[k].boundary_type == BoundaryType::Wall {
                b[k].boundary_type = BoundaryType::Outlet;
            }
        }
        b
    }

    fn refresh_warnings(&mut self) {
        self.build_warnings = match self.settings.materials.build() {
            Ok(b) => b.warnings,
            Err(e) => vec![e],
        };
    }

    /// Fluide qui entre par un bord « Inlet » : celui choisi à l'écran de démarrage (par nom), sinon le premier liquide.
    fn inlet_phase(&self) -> usize {
        self.sim
            .reg
            .phases
            .iter()
            .position(|p| p.name == self.settings.mp_inlet_fluid)
            .or_else(|| self.sim.liquid_phases().first().copied())
            .unwrap_or(self.sim.ambient_phase)
    }

    /// Reporte les bords de l'interface dans le solveur (Wall / Outlet = ouvert à 1 atm / Inlet = paroi à vitesse imposée). Retourne true si quelque chose a changé.
    fn apply_boundaries(&mut self) -> bool {
        let before = (self.sim.bc, self.sim.inlet);
        let phase = self.inlet_phase();
        for k in 0..4 {
            let b = self.boundaries[k];
            match b.boundary_type {
                BoundaryType::Wall => {
                    self.sim.bc[k] = Bc::Wall;
                    self.sim.inlet[k] = None;
                }
                BoundaryType::Outlet => {
                    self.sim.bc[k] = Bc::Open { p: P_ATM };
                    self.sim.inlet[k] = None;
                }
                BoundaryType::Inlet => {
                    self.sim.bc[k] = Bc::Wall;
                    self.sim.inlet[k] = Some(MpInlet { vx: b.velocity_x_ms as f64, vy: b.velocity_y_ms as f64, phase, temp: b.temperature as f64, offset: b.offset });
                }
            }
        }
        before != (self.sim.bc, self.sim.inlet)
    }

    /// Recommence le scénario avec `self.settings.materials`.
    fn reset(&mut self) {
        match Self::make(&self.settings, &self.settings.materials.clone(), self.scenario) {
            Ok(s) => {
                let was_gpu = self.backend == Backend::Gpu;
                self.sim = s;
                self.gpu = None;
                self.backend = Backend::Cpu;
                self.boundaries = Self::boundaries_for(&self.settings, &self.sim);
                self.apply_boundaries();
                self.fix_selection();
                self.stats = compute_stats(&self.sim);
                self.status = format!("Scénario : {}", self.scenario.name());
                if was_gpu {
                    // la nouvelle scène reste sur le GPU (le message d'erreur éventuel remplace le statut)
                    self.set_backend(Backend::Gpu);
                    if self.backend == Backend::Gpu {
                        self.status = format!("Scénario : {} (GPU)", self.scenario.name());
                    }
                }
                self.paused = false;
                self.step_frame = 0;
                self.tp_dest = None;
                self.picking_destination = false;
            }
            Err(e) => self.status = format!("Impossible de recommencer : {e}"),
        }
        self.refresh_warnings();
    }

    /// après un changement de matériaux, les indices d'outil / d'émetteur / de solide doivent rester valides
    fn fix_selection(&mut self) {
        let (np, ns) = (self.sim.reg.phases.len(), self.sim.reg.solids.len());
        let roles = Roles::resolve(&self.sim.reg, self.sim.ambient_phase);
        let ok = match self.tool {
            MpTool::Phase(k) => k < np,
            MpTool::Solid(s) => s < ns,
            MpTool::Mixture(m) => m < self.sim.mixtures.len(),
            MpTool::Ignite => !self.sim.reg.reactions.is_empty(),
            _ => true,
        };
        if !ok {
            self.tool = MpTool::Phase(roles.water.unwrap_or(0));
        }
        if self.emit_phase >= np {
            self.emit_phase = roles.water.unwrap_or(0);
        }
        if self.sel_solid >= ns {
            self.sel_solid = roles.plate.unwrap_or(0);
        }
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
                    ui.weak("LMB : outil (Ctrl : cellule spéciale) | RMB : pousser | MMB : effacer (solide + effet) | Maj+LMB / Maj+MMB : chauffer vers T pinceau");
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
        let shift = ctx.input(|i| i.modifiers.shift);
        if pressed(egui::Key::Space) {
            self.paused = !self.paused;
        }
        if self.paused && pressed(egui::Key::ArrowRight) {
            self.step_frame = 1;
        }
        if self.paused && pressed(egui::Key::ArrowUp) {
            self.step_frame = 10;
        }
        if pressed(egui::Key::S) {
            self.save_scene();
        }
        if pressed(egui::Key::L) {
            self.load_scene();
        }
        if pressed(egui::Key::I) {
            self.solids_white = !self.solids_white;
        }
        if pressed(egui::Key::V) {
            self.display = self.display.next();
        }
        if pressed(egui::Key::M) {
            self.circle = !self.circle;
        }
        // C se répète quand on le maintient
        if ctx.input(|i| i.key_pressed(egui::Key::C)) {
            self.radius = cycle_radius(self.radius, shift);
        }
        if pressed(egui::Key::R) {
            if shift {
                self.reset();
            } else {
                self.do_reset_flow();
            }
        }
        if pressed(egui::Key::Escape) {
            *action = WindowAction::BackToSettings;
        }
    }

    fn do_reset_flow(&mut self) {
        self.sync_host();
        reset_flow(&mut self.sim);
        self.upload_if_gpu();
        self.status = "Écoulement remis à zéro".into();
    }

    fn upload_if_gpu(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.upload(&self.sim); // l'état édité (relu au préalable par `sync_host`) remplace celui du GPU
        }
    }

    /// GPU résident : relit l'état complet dans `self.sim` s'il a avancé depuis la dernière relecture (avant une modification,
    /// une sauvegarde, des statistiques). Sans effet en calcul CPU.
    fn sync_host(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.ensure_host(&mut self.sim);
        }
    }

    /// Repasse sur le CPU si une modification dépasse les limites du GPU (plus de 32 fluides, réaction à plus de 6 réactifs / produits,
    /// plus de 256 destinations de téléporteurs). L'état vient d'être envoyé : CPU et GPU sont identiques à ce moment.
    fn enforce_backend(&mut self) {
        if self.backend == Backend::Gpu {
            if let Some(why) = gpu_unsupported_reason(&self.sim) {
                self.gpu = None;
                self.backend = Backend::Cpu;
                self.status = format!("Calcul repassé sur CPU : {why}");
            }
        }
    }

    fn after_blocking_dialog(&mut self) {
        self.last_frame = Instant::now();
        self.last_mouse = None;
    }

    fn save_scene(&mut self) {
        self.sync_host();
        let path = rfd::FileDialog::new().add_filter("Scène multiphasique", &["mps"]).set_file_name("scene.mps").save_file();
        if let Some(path) = path {
            self.status = match scene_to_json(&self.sim).and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string())) {
                Ok(()) => format!("Scène enregistrée : {}", path.display()),
                Err(e) => format!("Erreur d'enregistrement : {e}"),
            };
        }
        self.after_blocking_dialog();
    }

    fn load_scene(&mut self) {
        let path = rfd::FileDialog::new().add_filter("Scène multiphasique", &["mps"]).pick_file();
        if let Some(path) = path {
            self.sync_host();
            self.status = match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| scene_from_json(&mut self.sim, &s)) {
                Ok(w) if w.is_empty() => "Scène chargée".into(),
                Ok(w) => format!("Scène chargée ({} avertissement(s) : {})", w.len(), w[0]),
                Err(e) => format!("Erreur de chargement : {e}"),
            };
            self.upload_if_gpu();
            self.enforce_backend();
        }
        self.after_blocking_dialog();
    }

    fn top_bar(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        egui::TopBottomPanel::top("mp_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Sauver la scène (S)").clicked() {
                    self.save_scene();
                }
                if ui.button("Charger la scène (L)").clicked() {
                    self.load_scene();
                }
                ui.separator();
                let label = if self.paused { "Reprendre (Espace)" } else { "Pause (Espace)" };
                if ui.button(label).clicked() {
                    self.paused = !self.paused;
                }
                if ui.add_enabled(self.paused, egui::Button::new("Pas (→)")).clicked() {
                    self.step_frame = 1;
                }
                if ui.button("Reset flow (R)").clicked() {
                    self.do_reset_flow();
                }
                if ui.button("Recommencer (Maj+R)").clicked() {
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
        egui::SidePanel::left("mp_controls").default_width(300.0).show(ctx, |ui| {
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
                    ui.add(egui::Slider::new(&mut self.t_scale, 1.0..=1000.0).logarithmic(true).text("Échelle ΔT (K)"));
                }
                if self.display == MpDisplay::Pressure {
                    ui.add(egui::Slider::new(&mut self.p_scale, 0.0..=1e6).text("Échelle p (Pa, 0 = auto)"));
                }
                ui.checkbox(&mut self.solids_white, "Solides blancs (I)");

                ui.separator();
                ui.heading("Pinceau");
                for t in sim_tools(&self.sim) {
                    let name = tool_label(&self.sim, t);
                    ui.selectable_value(&mut self.tool, t, name);
                }
                ui.selectable_value(&mut self.tool, MpTool::ClearCell, tool_name(&self.sim.reg, MpTool::ClearCell));
                ui.horizontal(|ui| {
                    ui.label("Forme (M)");
                    ui.selectable_value(&mut self.circle, true, "Rond");
                    ui.selectable_value(&mut self.circle, false, "Carré");
                });
                ui.add(egui::Slider::new(&mut self.radius, 1..=60).text("Rayon (C)"));
                ui.add(egui::Slider::new(&mut self.brush_temp, 100.0..=3000.0).text("T pinceau (K)"));
                ui.weak("T pinceau : température des phases et mélanges ajoutés, des solides, et cible du chauffage");
                ui.add(egui::Slider::new(&mut self.sim.heat_power_max, 1e6..=1e10).logarithmic(true).text("Puissance max. du chauffage (W/m³)"));
                ui.weak("Le chauffage (Maj + clic) apporte sa chaleur pendant le temps SIMULÉ, au plus à cette puissance. Un liquide posé \
                         au-dessus de son ébullition est posé en vapeur. Faire bouillir de l'eau froide au-delà de ~5e8 W/m³ est une \
                         explosion de vapeur : la simulation la suit au ralenti (pas de temps de quelques µs).");
                if self.tool == MpTool::Ignite {
                    ui.add(egui::Slider::new(&mut self.ignite_temp, 1200.0..=3000.0).text("T du noyau de flamme (K)"));
                    ui.weak("Allumer : le mélange local est remplacé par ses gaz brûlés chauds (à pression constante) ; un noyau trop petit \
                             (< ~10 mailles de diamètre) peut s'éteindre, comme une étincelle trop faible.");
                }

                ui.separator();
                self.cell_effects_ui(ui);

                ui.separator();
                self.materials_ui(ui);

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
                if let Some(why) = gpu_unsupported_reason(&self.sim) {
                    ui.weak(format!("GPU indisponible pour cette scène : {why}"));
                }
                if self.backend == Backend::Gpu {
                    if let Some(g) = &self.gpu {
                        let l = g.last;
                        ui.weak(format!(
                            "GPU résident ({}) : dt {:.2e} s, {} Newton, {} PCG (max {} par résolution), sous-pas : {} conduction, {} espèces, {} noyaux par pas{}{}",
                            g.adapter_name,
                            l.dt,
                            l.newton,
                            l.pcg,
                            l.pcg_max,
                            l.cond_sub,
                            l.species_sub,
                            l.dispatches,
                            if l.retries > 0 { format!(", {} retour(s) arrière", l.retries) } else { String::new() },
                            if l.redo > 0 { format!(", {} pas refait(s) (budget)", l.redo) } else { String::new() }
                        ));
                    }
                }
                ui.add_enabled(
                    self.backend == Backend::Cpu,
                    egui::Checkbox::new(&mut self.sim.multigrid, "Pression : préconditionneur multigrille (sinon IC(0), ~10× plus d'itérations ; GPU : toujours multigrille)"),
                );
                ui.add(egui::Slider::new(&mut self.sim.cfl, 0.1..=0.9).text("CFL"));
                ui.add(egui::Slider::new(&mut self.sim.dt_max, 1e-5..=2e-2).logarithmic(true).text("dt max (s)"));
                let mut g = -self.sim.gravity[1];
                if ui.add(egui::Slider::new(&mut g, 0.0..=100.0).text("Gravité (m/s²)")).changed() {
                    self.sim.gravity = [0.0, -g];
                }
                ui.checkbox(&mut self.sim.surface_tension, "Tension de surface");
                ui.checkbox(&mut self.sim.viscous, "Viscosité");
                ui.checkbox(&mut self.sim.conduction, "Conduction thermique");
                ui.checkbox(&mut self.sim.phase_change, "Changement de phase (ébullition / cavitation / évaporation)");
                if !self.sim.reg.reactions.is_empty() {
                    ui.checkbox(&mut self.sim.chemistry, "Réactions chimiques");
                    ui.checkbox(&mut self.sim.radiation, "Rayonnement des gaz chauds (H2O, CO2)");
                    ui.add(egui::Slider::new(&mut self.sim.flame_resolution, 4.0..=12.0).text("Épaisseur des flammes (mailles)"));
                    ui.weak("Flamme épaissie : la vitesse de flamme est conservée quelle que soit la maille ; plus de mailles = plus précis, plus lent.");
                }
                ui.checkbox(&mut self.sim.conservative_momentum, "Quantité de mouvement conservative (sinon semi-lagrangien)");
                if ui.button("Reset temperature").clicked() {
                    self.sync_host();
                    let t_amb = self.sim.t_ambient;
                    reset_temperature(&mut self.sim, t_amb);
                    self.upload_if_gpu();
                    self.status = "Température remise à l'ambiante".into();
                }

                ui.separator();
                ui.heading("Bords");
                let dx = self.settings.dx;
                let names = ["Gauche", "Droite", "Bas", "Haut"];
                for k in 0..4 {
                    boundary_ui(ui, names[k], &mut self.boundaries[k], dx);
                }
                ui.weak("Inlet : vitesse imposée sur la face normale, fluide d'entrée choisi à l'écran de démarrage ; Outlet : ouvert à 1 atm.");
                if self.apply_boundaries() {
                    self.sync_host();
                    self.upload_if_gpu();
                    self.enforce_backend();
                }
            });
        });
    }

    /// « Cell Effects » + « Solid thermal » du mono-fluide
    fn cell_effects_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Cellules spéciales (Ctrl + clic gauche)");
        ui.checkbox(&mut self.show_effects, "Afficher les effets (couleurs des cellules spéciales)");
        for k in CellKind::ALL {
            if ui.selectable_value(&mut self.cell_kind, k, k.name()).changed() {
                self.picking_destination = false;
            }
        }
        ui.separator();
        match self.cell_kind {
            CellKind::Wall => {
                if self.sim.reg.solids.is_empty() {
                    ui.weak("Aucun solide défini (voir Matériaux).");
                } else {
                    self.sel_solid = self.sel_solid.min(self.sim.reg.solids.len() - 1);
                    egui::ComboBox::from_label("Matériau").selected_text(self.sim.reg.solids[self.sel_solid].name.clone()).show_ui(ui, |ui| {
                        for (i, s) in self.sim.reg.solids.iter().enumerate() {
                            ui.selectable_value(&mut self.sel_solid, i, &s.name);
                        }
                    });
                }
                ui.label("Comportement thermique du solide posé");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Passive, "Solide passif");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Isothermal, "Isotherme (T imposée)");
                ui.selectable_value(&mut self.thermal_mode, ThermalMode::Heater, "Chauffage (W/m³)");
                match self.thermal_mode {
                    ThermalMode::Passive => {}
                    ThermalMode::Isothermal => {
                        ui.add(egui::Slider::new(&mut self.thermal_temp, 100.0..=2000.0).text("T (K)"));
                    }
                    ThermalMode::Heater => {
                        ui.add(egui::Slider::new(&mut self.thermal_power, 1e2..=1e9).logarithmic(true).text("Puissance (W/m³)"));
                        let dx = self.sim.dx;
                        ui.weak(format!("= {:.2} W par mètre de profondeur et par cellule", self.thermal_power * dx * dx));
                    }
                }
                ui.weak("Le même comportement s'applique aux solides posés au clic gauche avec l'outil « Solide ».");
            }
            CellKind::Sink => {
                ui.add(egui::Slider::new(&mut self.sink_rate, 0.0..=10.0).text("Aspiration (m/s)"));
                ui.weak("Aspire le fluide voisin sur chaque face ouverte et le fait disparaître (la masse retirée est comptée dans le bilan).");
            }
            CellKind::Emitter => {
                let names: Vec<String> = self.sim.reg.phases.iter().map(|p| p.name.clone()).collect();
                self.emit_phase = self.emit_phase.min(names.len().saturating_sub(1));
                egui::ComboBox::from_label("Fluide émis").selected_text(names.get(self.emit_phase).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut self.emit_phase, i, n);
                    }
                });
                ui.add(egui::Slider::new(&mut self.emit_temp, 100.0..=2000.0).text("T émise (K)"));
                ui.add(egui::Slider::new(&mut self.emit_vx, -10.0..=10.0).text("Vitesse X (m/s)"));
                ui.add(egui::Slider::new(&mut self.emit_vy, -10.0..=10.0).text("Vitesse Y (m/s)"));
                ui.weak("Buse : crée en continu le fluide choisi et l'expulse à cette vitesse (autres faces fermées). Vitesse nulle : réservoir à la pression locale.");
            }
            CellKind::Accelerator => {
                ui.add(egui::Slider::new(&mut self.acc_ax, -50.0..=50.0).text("Accélération X (m/s²)"));
                ui.add(egui::Slider::new(&mut self.acc_ay, -50.0..=50.0).text("Accélération Y (m/s²)"));
            }
            CellKind::Teleporter => {
                ui.label("1. Choisir une destination\n2. Ctrl + clic gauche pour poser des téléporteurs");
                let label = if self.picking_destination { "Cliquez sur la grille…" } else { "Choisir la destination" };
                if ui.button(label).clicked() {
                    self.picking_destination = !self.picking_destination;
                }
                match self.tp_dest {
                    Some(d) => {
                        ui.label(format!("Destination : {}, {}", d % self.sim.nx, d / self.sim.nx));
                    }
                    None => {
                        ui.weak("Pas de destination");
                    }
                }
                ui.add(egui::Slider::new(&mut self.tp_speed, 0.0..=10.0).text("Aspiration (m/s)"));
                ui.weak("Le fluide absorbé (composition, température) réapparaît à la destination avec le même débit volumique.");
            }
        }
    }

    /// Éditeur de matériaux de la fenêtre : agit sur une copie, appliquée par « Appliquer et recommencer ».
    fn materials_ui(&mut self, ui: &mut egui::Ui) {
        let mut apply = false;
        ui.collapsing("Matériaux (fluides, solides, miscibilité…)", |ui| {
            if self.editor.ui(ui, &mut self.lib) {
                self.lib_dirty = true;
            }
            ui.separator();
            let label = if self.lib_dirty { "Appliquer et recommencer (modifié)" } else { "Appliquer et recommencer" };
            if ui.button(label).clicked() {
                apply = true;
            }
            ui.weak("Ajouter, supprimer ou modifier un matériau change les indices : la simulation est reconstruite depuis le scénario courant.");
        });
        if apply {
            self.settings.materials = self.lib.clone();
            save_settings(&self.settings);
            self.lib_dirty = false;
            self.reset();
        }
    }

    fn set_backend(&mut self, b: Backend) {
        match b {
            Backend::Cpu => {
                self.sync_host(); // l'état le plus récent est sur le GPU
                self.gpu = None;
                self.backend = Backend::Cpu;
                self.status = "Calcul CPU (f64)".into();
            }
            Backend::Gpu => {
                if let Some(why) = gpu_unsupported_reason(&self.sim) {
                    self.backend = Backend::Cpu;
                    self.status = format!("GPU indisponible : {why}");
                    return;
                }
                match MultiphaseGpu::try_new(&self.sim) {
                    Ok(g) => {
                        self.gpu = Some(g);
                        self.backend = Backend::Gpu;
                        self.status = "Calcul GPU (f32, état résident sur le GPU)".into();
                    }
                    Err(e) => {
                        self.gpu = None;
                        self.backend = Backend::Cpu;
                        self.status = format!("GPU indisponible : {e}");
                    }
                }
            }
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
                    row("Chaleur injectée (J/m)", format!("{:.4e}", st.heat_input));
                    row("Cellules spéciales", format!("{}", st.effects));
                    row("Pas annulés (filet)", format!("{}", st.rollbacks));
                    if !st.reactions.is_empty() {
                        row("Puissance chimique (W/m)", format!("{:.4e}", st.chem_power));
                        row("Chaleur de réaction (J/m)", format!("{:.4e}", st.chem_heat));
                        row("Énergie rayonnée (J/m)", format!("{:.4e}", st.radiated));
                        row("Épaississement des flammes F", format!("{:.0}", st.flame_factor));
                    }
                });
                if !st.reactions.is_empty() {
                    ui.separator();
                    ui.label("Avancement des réactions (mol/m)");
                    egui::Grid::new("mp_rx_grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                        for (n, x) in &st.reactions {
                            ui.label(n);
                            ui.label(format!("{:.4e}", x));
                            ui.end_row();
                        }
                    });
                }
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
                if !self.build_warnings.is_empty() {
                    ui.separator();
                    for w in &self.build_warnings {
                        ui.colored_label(Color32::from_rgb(230, 170, 40), format!("⚠ {w}"));
                    }
                }
                ui.separator();
                let mut reset_prof = false;
                egui::CollapsingHeader::new(format!(
                    "Profil du calcul : {:.1} ms par pas, {} fil(s)",
                    1e3 * self.sim.profile.total() / self.sim.profile.steps.max(1) as f64,
                    rayon::current_num_threads()
                ))
                .id_source("mp_prof")
                .show(ui, |ui| {
                    egui::Grid::new("mp_prof_grid").num_columns(3).spacing([10.0, 2.0]).show(ui, |ui| {
                        for (name, ms, frac) in self.sim.profile.report() {
                            ui.label(name);
                            ui.label(format!("{ms:.2} ms"));
                            ui.label(format!("{:.0} %", 100.0 * frac));
                            ui.end_row();
                        }
                    });
                    ui.label(format!("Itérations : {} PCG, {} Newton (dernier pas)", self.sim.last_pressure_iters, self.sim.last_newton_iters));
                    if ui.button("Remettre le profil à zéro").clicked() {
                        reset_prof = true;
                    }
                });
                if reset_prof {
                    self.sim.profile = Default::default();
                }
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
                    ("S / L", "Sauver / charger la scène (solides, thermique, cellules spéciales)"),
                    ("I", "Solides blancs"),
                    ("V", "Mode d'affichage suivant (dont divergence)"),
                    ("M", "Pinceau rond / carré"),
                    ("C / Maj+C", "Rayon de pinceau suivant / précédent"),
                    ("R", "Reset flow (vitesses à zéro)"),
                    ("Maj + R", "Recommencer le scénario"),
                    ("Échap", "Retour aux réglages"),
                    ("Clic gauche", "Applique l'outil (phase, mélange, solide, chauffage, poussée, allumage…)"),
                    ("Ctrl + clic gauche", "Pose le type de cellule choisi (solide + thermique, puits, émetteur, accélérateur, téléporteur)"),
                    ("Clic droit", "Pousse le fluide avec le curseur"),
                    ("Clic milieu / Ctrl + milieu", "Efface solide, effet et comportement thermique"),
                    ("Maj + clic gauche / milieu", "Chauffer / refroidir vers T pinceau"),
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
            // les cibles du pinceau « Chauffer » ne valent que pour l'image en cours (elles sont reposées tant que le bouton est enfoncé)
            self.sim.clear_heat_targets();
            if let Some(g) = &mut self.gpu {
                g.sync_heat_targets(&self.sim);
            }
            self.stats_age += 1;
            // GPU résident : l'état complet n'est relu que pour les statistiques (~2 fois par seconde) et les outils
            let every = if self.backend == Backend::Gpu { 30 } else { 5 };
            if self.stats_age >= every || self.paused {
                self.sync_host();
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

    /// outil posé par Ctrl + clic gauche d'après le type de cellule choisi
    fn ctrl_tool(&self) -> Option<(MpTool, MpThermal)> {
        match self.cell_kind {
            CellKind::Wall => {
                if self.sim.reg.solids.is_empty() {
                    return None;
                }
                let th = match self.thermal_mode {
                    ThermalMode::Passive => MpThermal::Passive,
                    ThermalMode::Isothermal => MpThermal::Isothermal(self.thermal_temp),
                    ThermalMode::Heater => MpThermal::Heater(self.thermal_power),
                };
                Some((MpTool::Solid(self.sel_solid.min(self.sim.reg.solids.len() - 1)), th))
            }
            CellKind::Sink => Some((MpTool::Effect(MpEffect::Sink { rate: self.sink_rate }), MpThermal::Passive)),
            CellKind::Emitter => Some((
                MpTool::Effect(MpEffect::Emitter { vx: self.emit_vx, vy: self.emit_vy, phase: self.emit_phase, temp: self.emit_temp }),
                MpThermal::Passive,
            )),
            CellKind::Accelerator => Some((MpTool::Effect(MpEffect::Accelerator { ax: self.acc_ax, ay: self.acc_ay }), MpThermal::Passive)),
            CellKind::Teleporter => self.tp_dest.map(|d| (MpTool::Effect(MpEffect::Teleporter { dest: d, speed: self.tp_speed }), MpThermal::Passive)),
        }
    }

    fn mouse(&mut self, ctx: &egui::Context, response: &egui::Response, rect: egui::Rect, dt: f64) {
        let (pos, left, right, middle, shift, ctrl) = ctx.input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.button_down(PointerButton::Primary),
                i.pointer.button_down(PointerButton::Secondary),
                i.pointer.button_down(PointerButton::Middle),
                i.modifiers.shift,
                i.modifiers.command, // Ctrl (Win/Linux) ou Cmd (macOS)
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

        // choix de la destination d'un téléporteur
        if self.picking_destination {
            if response.clicked() {
                let c = self.sim.idx(ci, cj);
                if self.sim.is_solid(c) {
                    self.status = "La destination ne peut pas être un solide".into();
                } else if self.sim.special[c].is_some() {
                    self.status = "La destination ne peut pas être une cellule spéciale".into();
                } else {
                    self.tp_dest = Some(c);
                    self.picking_destination = false;
                    self.status = format!("Destination du téléporteur : {ci}, {cj}");
                }
            }
            return;
        }

        if !(left || right || middle) {
            return;
        }
        if left && ctrl && self.cell_kind == CellKind::Teleporter && self.tp_dest.is_none() {
            self.status = "Téléporteur : choisissez d'abord une destination".into();
            return;
        }
        let cells = brush_cells(nx, ny, ci, cj, self.radius, self.circle);
        // GPU résident : un pinceau « Chauffer » sur du fluide ne pose que des cibles (envoyées seules) ; tout autre outil modifie l'état :
        // il est d'abord relu, puis renvoyé
        let heat_only = !right && ((left && shift && !ctrl && !middle) || (middle && shift && !ctrl && !left));
        let light = heat_only && cells.iter().all(|&(i, j)| i < nx && j < ny && !self.sim.is_solid(self.sim.idx(i, j)));
        if !light {
            self.sync_host();
        }
        let t_tool = if self.tool == MpTool::Ignite { self.ignite_temp } else { self.brush_temp };
        let base = ToolOpts { temp: t_tool, cursor, dt, thermal: self.thermal_for_tool() };
        let mut edited = false;
        if left {
            if shift && !ctrl {
                apply_tool_with(&mut self.sim, MpTool::Heat, &cells, &ToolOpts { temp: self.brush_temp, ..base });
            } else if ctrl {
                if let Some((tool, th)) = self.ctrl_tool() {
                    let temp = if matches!(self.cell_kind, CellKind::Wall) && self.thermal_mode == ThermalMode::Isothermal { self.thermal_temp } else { self.brush_temp };
                    apply_tool_with(&mut self.sim, tool, &cells, &ToolOpts { thermal: th, temp, ..base });
                }
            } else {
                apply_tool_with(&mut self.sim, self.tool, &cells, &base);
            }
            edited = true;
        }
        if right {
            apply_tool_with(&mut self.sim, MpTool::Push, &cells, &base);
            edited = true;
        }
        if middle {
            if shift && !ctrl {
                apply_tool_with(&mut self.sim, MpTool::Heat, &cells, &ToolOpts { temp: self.brush_temp, ..base });
            } else {
                apply_tool_with(&mut self.sim, MpTool::ClearCell, &cells, &base);
            }
            edited = true;
        }
        if edited {
            if light {
                if let Some(g) = &mut self.gpu {
                    g.sync_heat_targets(&self.sim);
                }
            } else {
                self.upload_if_gpu();
                self.enforce_backend();
            }
        }
    }

    /// comportement thermique donné aux solides posés par l'outil « Solide » du clic gauche (le même que celui du type de cellule « mur »)
    fn thermal_for_tool(&self) -> MpThermal {
        match self.thermal_mode {
            ThermalMode::Passive => MpThermal::Passive,
            ThermalMode::Isothermal => MpThermal::Isothermal(self.thermal_temp),
            ThermalMode::Heater => MpThermal::Heater(self.thermal_power),
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
                    // pas protégés sur le GPU (instantané, retour arrière + dt/4 comme le CPU) ; seuls quelques scalaires sont relus,
                    // une fois par série de pas (jusqu'à 8 pas soumis d'un bloc, selon le temps simulé restant pour cette image)
                    let remaining = target - (self.sim.time - t0);
                    let k = if self.paused { 1 } else { ((remaining / self.sim.last_dt.max(1e-9)).floor() as usize).clamp(1, 8) };
                    g.step_batch(&mut self.sim, k);
                }
                _ => {
                    self.sim.step(); // pas protégé : retour arrière + sous-pas si l'état devient invalide
                }
            }
            if self.sim.diverged {
                self.paused = true;
                self.step_frame = 0;
                self.status = "Divergence numérique : simulation mise en pause (état invalide détecté, dernier état sain conservé en CPU). Réduire CFL / dt max, ou recommencer".into();
                break;
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
        let o = RenderOpts { display: self.display, t_scale: self.t_scale, p_scale: self.p_scale, show_effects: self.show_effects, solids_white: self.solids_white };
        match (&mut self.gpu, self.backend) {
            // GPU résident : image rendue sur le GPU, seuls nx·ny·4 octets sont relus
            (Some(g), Backend::Gpu) => g.render(&self.sim, &o, &mut self.rgba),
            _ => render_rgba(&self.sim, o, &mut self.rgba),
        }
        // destination du téléporteur en cours de réglage
        if self.show_effects && self.cell_kind == CellKind::Teleporter {
            if let Some(d) = self.tp_dest {
                let (i, j) = (d % self.sim.nx, d / self.sim.nx);
                let p = ((self.sim.ny - 1 - j) * self.sim.nx + i) * 4;
                if p + 2 < self.rgba.len() && !self.sim.is_solid(d) {
                    self.rgba[p] = 0x00;
                    self.rgba[p + 1] = 0xff;
                    self.rgba[p + 2] = 0xff;
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2, Event, Modifiers, RawInput, Rect};

    fn frame(ctx: &egui::Context, w: &mut MultiphaseWindow, events: Vec<Event>, modifiers: Modifiers) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1500.0, 950.0))),
            events,
            modifiers,
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            w.ui(ctx);
        });
    }

    fn settings() -> SimulationSettings {
        SimulationSettings { multiphase: true, width: 320, height: 240, precision: 10, gravity: 9.81, mp_scenario: 5, ..Default::default() }
    }

    fn click_at(ctx: &egui::Context, w: &mut MultiphaseWindow, p: egui::Pos2, mods: Modifiers, button: egui::PointerButton) {
        frame(ctx, w, vec![Event::PointerMoved(p)], mods);
        frame(ctx, w, vec![Event::PointerButton { pos: p, button, pressed: true, modifiers: mods }], mods);
        frame(ctx, w, vec![Event::PointerButton { pos: p, button, pressed: false, modifiers: mods }], mods);
    }

    #[test]
    fn window_renders_and_every_feature_responds() {
        let ctx = egui::Context::default();
        let mut w = MultiphaseWindow::new(&settings());
        for _ in 0..5 {
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        assert!(w.sim.time > 0.0, "la simulation n'avance pas");
        let ctrl = Modifiers { ctrl: true, command: true, ..Default::default() };
        let centre = pos2(760.0, 475.0);

        // chaque type de cellule posable
        w.paused = true;
        for kind in [CellKind::Sink, CellKind::Emitter, CellKind::Accelerator, CellKind::Wall] {
            w.cell_kind = kind;
            if kind == CellKind::Wall {
                w.thermal_mode = ThermalMode::Heater;
            }
            click_at(&ctx, &mut w, centre, ctrl, egui::PointerButton::Primary);
            match kind {
                CellKind::Wall => assert!(w.sim.solid.iter().any(|&s| s >= 0), "solide non posé"),
                _ => assert!(w.sim.special_count() > 0, "{} non posé", kind.name()),
            }
            if kind == CellKind::Wall {
                assert!(w.sim.thermal.iter().any(|t| matches!(t, MpThermal::Heater(_))), "chauffage non posé");
            }
        }
        // téléporteur : sans destination rien n'est posé, puis choix de la destination par clic
        let before = w.sim.special_count();
        w.cell_kind = CellKind::Teleporter;
        click_at(&ctx, &mut w, pos2(900.0, 475.0), ctrl, egui::PointerButton::Primary);
        assert_eq!(w.sim.special_count(), before);
        w.picking_destination = true;
        click_at(&ctx, &mut w, pos2(1000.0, 600.0), Modifiers::default(), egui::PointerButton::Primary);
        assert!(w.tp_dest.is_some() && !w.picking_destination);
        click_at(&ctx, &mut w, pos2(900.0, 475.0), ctrl, egui::PointerButton::Primary);
        assert!(w.sim.special.iter().any(|s| matches!(s, Some(MpEffect::Teleporter { .. }))));

        // clic milieu : efface solide + effets
        w.radius = 60;
        click_at(&ctx, &mut w, centre, Modifiers::default(), egui::PointerButton::Middle);
        assert!(w.sim.solid.iter().all(|&s| s < 0) && w.sim.special_count() == 0, "MMB n'a pas tout effacé");

        // la simulation continue sans NaN avec des extras
        w.radius = 2;
        w.cell_kind = CellKind::Emitter;
        click_at(&ctx, &mut w, pos2(700.0, 800.0), ctrl, egui::PointerButton::Primary);
        w.paused = false;
        for _ in 0..30 {
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        assert!(w.sim.state_is_sane(), "état invalide après 30 trames avec émetteur");

        // bords en direct : Inlet à gauche (eau), Outlet à droite
        w.boundaries[0].boundary_type = BoundaryType::Inlet;
        w.boundaries[0].velocity_x_ms = 0.5;
        w.boundaries[1].boundary_type = BoundaryType::Outlet;
        frame(&ctx, &mut w, vec![], Modifiers::default());
        assert!(w.sim.inlet[0].is_some() && matches!(w.sim.bc[1], Bc::Open { .. }) && w.sim.bc[0] == Bc::Wall);
        for _ in 0..20 {
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        assert!(w.sim.state_is_sane());

        // GPU résident : accepte les extras (cellules spéciales, entrées, solides thermiques) ; sans adaptateur, repli CPU expliqué
        w.set_backend(Backend::Gpu);
        assert!(w.status.contains("GPU"), "{}", w.status);
        if w.backend == Backend::Gpu {
            for _ in 0..20 {
                frame(&ctx, &mut w, vec![], Modifiers::default());
            }
            w.sync_host();
            assert!(w.sim.state_is_sane() && !w.sim.diverged, "GPU : état invalide avec émetteur et entrée de bord");
        }

        // raccourcis : C change le rayon, I les solides, V l'affichage, R remet l'écoulement à zéro
        let r0 = w.radius;
        let key = |k: egui::Key| Event::Key { key: k, physical_key: Some(k), pressed: true, repeat: false, modifiers: Modifiers::default() };
        frame(&ctx, &mut w, vec![key(egui::Key::C)], Modifiers::default());
        assert_ne!(w.radius, r0);
        frame(&ctx, &mut w, vec![key(egui::Key::I)], Modifiers::default());
        assert!(w.solids_white);
        let d0 = w.display;
        frame(&ctx, &mut w, vec![key(egui::Key::V)], Modifiers::default());
        assert_ne!(w.display, d0);
        frame(&ctx, &mut w, vec![key(egui::Key::R)], Modifiers::default());
        assert!(w.status.contains("remis à zéro"), "{}", w.status);

        // tous les modes d'affichage rendent
        for d in MpDisplay::ALL {
            w.display = d;
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
    }

    #[test]
    fn combustion_scenario_ignites_from_the_ui_and_renders_flames() {
        let ctx = egui::Context::default();
        let mut st = settings();
        st.mp_scenario = MpScenario::ConfinedExplosion.index();
        let mut w = MultiphaseWindow::new(&st);
        assert!(w.sim.reg.reactions.len() >= 3 && w.status.starts_with("Scénario"), "{}", w.status);
        // chimie sur le GPU résident (si un adaptateur est disponible)
        w.set_backend(Backend::Gpu);
        assert!(w.status.contains("GPU"), "{}", w.status);
        // allumage supplémentaire au pinceau, dans un coin
        w.tool = MpTool::Ignite;
        w.radius = 4;
        w.paused = true;
        click_at(&ctx, &mut w, pos2(500.0, 200.0), Modifiers::default(), egui::PointerButton::Primary);
        w.paused = false;
        for _ in 0..25 {
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        w.sync_host(); // GPU résident : l'état n'est relu que sur demande
        assert!(w.sim.state_is_sane() && w.sim.chem_heat > 0.0, "rien n'a brûlé ({})", w.status);
        for d in [MpDisplay::Flame, MpDisplay::HeatRelease, MpDisplay::Temperature] {
            w.display = d;
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        assert!(sim_tools(&w.sim).iter().any(|t| matches!(t, MpTool::Mixture(_))));
    }

    #[test]
    fn applying_new_materials_rebuilds_the_simulation() {
        let ctx = egui::Context::default();
        let mut w = MultiphaseWindow::new(&settings());
        let n0 = w.sim.reg.phases.len();
        // l'utilisateur ajoute un liquide miscible à l'eau, supprime l'huile
        let mut lib = w.lib.clone();
        let mut alc = lib.template_fluid(crate::simulation::phases::PhaseKind::Liquid);
        alc.name = "alcool".into();
        lib.fluids.push(alc);
        lib.set_miscible("eau", "alcool", true);
        let huile = lib.fluid_index("huile").unwrap();
        lib.remove_fluid(huile);
        w.settings.materials = lib;
        w.scenario = MpScenario::DamBreak;
        w.reset();
        assert_eq!(w.sim.reg.phases.len(), n0, "4 - huile + alcool");
        assert!(w.sim.reg.miscible_groups().iter().any(|g| g.len() == 2));
        w.fix_selection();
        for _ in 0..10 {
            frame(&ctx, &mut w, vec![], Modifiers::default());
        }
        assert!(w.sim.state_is_sane());
        // l'outil « huile » n'existe plus : un outil valide est sélectionné
        assert!(matches!(w.tool, MpTool::Phase(k) if k < w.sim.reg.phases.len()));
    }
}
