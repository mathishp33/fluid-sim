//! Fenêtre d'EXPÉRIENCE (format des protocoles : EXPERIENCES.md). Un protocole écrit en français — départ, mesures, actions en temps
//! simulé — qu'on exécute pas à pas avec ses mesures et ses résultats (même moteur que la ligne de commande : `mp_experiment`), et
//! des conditions initiales faciles à poser :
//!
//!  * pinceaux et rectangles sur l'image, comme dans la fenêtre multiphasique : chaque geste devient une ligne du protocole
//!    (« remplir trait … avec "eau" à 293.15 K »), insérée avant le premier « simuler » ; l'aperçu est reconstruit depuis le texte,
//!    ce qu'on voit est donc exactement ce que la ligne de commande rejouera ;
//!  * champs de valeurs (température, vitesses, fraction d'un fluide, solides) lus dans un fichier CSV ou une image PNG, ou donnés
//!    par une formule en x, y.
//!
//! Clic droit sur l'image : copie les coordonnées (en mètres) de la maille, à coller dans le protocole.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use egui::{Color32, ColorImage, PointerButton, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};

use crate::simulation::mp_experiment::{apply_action, fmt_num, fmt_point, initial_state, results_dir, Action, Measure, Protocol, RunState, Runner, Step, Zone};
use crate::simulation::mp_script::{insert_initial_line, parse_all, zone_text};
use crate::simulation::mp_view::{cell_info, compute_stats, render_rgba, MpDisplay, MpScenario, MpStats, RenderOpts};
use crate::simulation::multiphase::MultiphaseSim;
use crate::window::main_window::WindowAction;
use crate::window::starting_window::SimulationSettings;

const FRAME_BUDGET: Duration = Duration::from_millis(14);
/// délai après la dernière frappe avant de reconstruire l'aperçu des conditions initiales
const REBUILD_DELAY: Duration = Duration::from_millis(600);

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Protocol,
    Initial,
}

/// Outil des conditions initiales (pinceau ou rectangle)
#[derive(Clone, Copy, PartialEq, Debug)]
enum CiTool {
    Fill,
    Solid,
    Body,
    Temperature,
    Clear,
    Ignite,
    Sink,
    Emitter,
    Grille,
}

impl CiTool {
    const ALL: [CiTool; 9] =
        [CiTool::Fill, CiTool::Solid, CiTool::Body, CiTool::Temperature, CiTool::Clear, CiTool::Ignite, CiTool::Sink, CiTool::Emitter, CiTool::Grille];
    fn name(self) -> &'static str {
        match self {
            CiTool::Fill => "Remplir d'un fluide ou d'un mélange",
            CiTool::Solid => "Solide",
            CiTool::Body => "Corps rigide fixe (disque au clic, rectangle tiré)",
            CiTool::Temperature => "Température (composition gardée)",
            CiTool::Clear => "Effacer (solides, cellules spéciales)",
            CiTool::Ignite => "Allumer (noyau de gaz brûlés)",
            CiTool::Sink => "Puits",
            CiTool::Emitter => "Émetteur",
            CiTool::Grille => "Grille (arrête les grains)",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Shape {
    Brush,
    Rect,
}

#[derive(Clone, Copy, PartialEq)]
enum FieldKind {
    T,
    Vx,
    Vy,
    Fraction,
    Solid,
}

#[derive(Clone, Copy, PartialEq)]
enum SolidThermal {
    Passive,
    Isothermal,
    Heater,
}

const SYNTAX_HELP: &str = "\
Une commande par ligne ; # commentaire ; noms entre guillemets ; point décimal ; unités facultatives (m, s, K par défaut).
Coordonnées en mètres, origine en bas à gauche, y vers le haut.

DÉPART (première ligne)
  départ scénario \"Rupture de barrage\" grille 160 x 120 maille 5 mm [gravité 9.81]
  départ état \"etat.mps\"                      (état complet enregistré)
  matériaux \"bibliotheque.json\"                (sinon matériaux par défaut)
  calcul gpu                                    (ou cpu)

ZONES
  partout · disque (x, y) rayon r · rect (x1, y1) (x2, y2) · point (x, y) · trait (x1, y1) (x2, y2) … rayon r

CONDITIONS INITIALES ET ACTIONS (instantanées)
  remplir <zone> avec \"eau\" [à 20 °C]           température <zone> 350 K
  solide <zone> \"acier\" [isotherme 400 K | chauffant 1e6 W/m³] [à T]
  corps disque (x, y) rayon r \"acier\" | corps rect (x1, y1) (x2, y2) \"verre\" [angle 30] | corps polygone (x1, y1) (x2, y2) (x3, y3) \"acier\"
        (corps rigide aux bords lissés ; angle en degrés) — fixe, ou mouvement imposé :
        vitesse (vx, vy) · rotation 90 (°/s, rad/s, tr/min) · oscillation (vx, vy) période 1 s · oscillation rotation 30 période 2 s · pivot (x, y)
  effacer <zone>     allumer <zone> [à 2200 K]    vitesse <zone> (vx, vy) m/s
  puits <zone> [0.2 m/s]   émetteur <zone> \"eau\" (0, 1) m/s [à T]   grille <zone> [trous 1 cm] [ouverte 0.6]
  bord gauche|droite|bas|haut  mur | ouvert [p] | entrée (vx, vy) [à T] [\"fluide\"]
  champ T|vx|vy|fraction \"eau\"|solide \"acier\"  fichier \"f.csv|f.png\" [origine haut|bas] [valeurs a b]
        | formule \"300 + 50*exp(-((x-0.2)^2+(y-0.1)^2)/0.02^2)\"     [dans <zone>] [à T]
  gravité (0, -9.81) · pas max 1 ms

ACTIONS À DURÉE
  chauffer <zone> à 600 K [pendant 2 s]          accélérer <zone> de (5, 0) m/s² [pendant 0.5 s]

TEMPS, MESURES, SORTIES
  simuler 1 s · simuler jusqu'à 3 s · pause · marque \"texte\" · sauver état \"nom\"
  mesurer toutes les 10 ms : stats, T en (x, y), moyenne|max|min|somme <grandeur> dans <zone>, masse \"eau\" [dans <zone>] [comme \"nom\"]
     grandeurs : T, p, vx, vy, vitesse, rho, fraction \"fluide\", chaleur
  balayage T0 = 300 K, 350 K, 400 K               ($T0 dans le texte : une expérience par valeur)

Résultats : résultats/<nom du protocole>/ à côté du protocole (mesures.csv, journal.txt, états .mps, copie du protocole).
Ligne de commande : fluid-sim experience protocole.exp [--gpu] [--sortie dossier]";

/// Fenêtre d'expérience (écran à part, lancé depuis l'écran de démarrage).
pub struct ExperimentWindow {
    /// réglages de l'écran de démarrage (scénario, grille, gravité du protocole « Nouveau »)
    settings: SimulationSettings,
    text: String,
    path: Option<PathBuf>,
    saved_text: String,
    errors: Vec<String>,
    /// aperçu des conditions initiales et texte dont il vient
    preview: Option<MultiphaseSim>,
    preview_src: String,
    edited_at: Instant,
    /// première variante du protocole lu (zones affichées, résumé)
    parsed: Option<Protocol>,
    runner: Option<Runner>,
    /// variantes d'un balayage qui restent à exécuter
    queue: Vec<Protocol>,
    n_variants: usize,
    use_gpu: bool,
    /// racine des résultats (None : « résultats » à côté du protocole)
    pub out_root: Option<PathBuf>,
    last_out: Option<PathBuf>,

    texture: Option<TextureHandle>,
    rgba: Vec<u8>,
    display: MpDisplay,
    t_scale: f64,
    p_scale: f64,
    show_zones: bool,
    show_effects: bool,
    canvas: Rect,
    hovered: Option<(usize, usize)>,
    stats: MpStats,
    stats_age: usize,

    tab: Tab,
    tool: CiTool,
    shape: Shape,
    radius: usize,
    fluid: String,
    solid: String,
    temp: f64,
    thermal: SolidThermal,
    thermal_temp: f64,
    thermal_power: f64,
    sink_rate: f64,
    emit_v: [f64; 2],
    grille_hole_mm: f64,
    grille_open: f64,
    stroke: Vec<[f64; 2]>,
    drag_start: Option<(usize, usize)>,
    drag_now: Option<(usize, usize)>,
    /// dernière ligne ajoutée par un geste (numéro, texte) : « Annuler »
    last_insert: Option<(usize, String)>,

    field_kind: FieldKind,
    field_name: String,
    field_file: Option<PathBuf>,
    field_formula: String,
    field_use_formula: bool,
    field_top: bool,
    field_range: bool,
    field_a: f64,
    field_b: f64,
    field_in_rect: bool,
    last_rect: Option<Zone>,

    plot_col: usize,
    status: String,
    show_help: bool,
}

/// taille de maille des réglages (f32) arrondie au micromètre
fn settings_dx(s: &SimulationSettings) -> f64 {
    ((s.dx as f64) * 1e6).round() / 1e6
}

impl ExperimentWindow {
    pub fn new(settings: &SimulationSettings) -> Self {
        let text = Self::template(settings);
        let mut w = ExperimentWindow {
            settings: settings.clone(),
            saved_text: String::new(),
            text,
            path: None,
            errors: vec![],
            preview: None,
            preview_src: String::new(),
            edited_at: Instant::now(),
            parsed: None,
            runner: None,
            queue: vec![],
            n_variants: 0,
            use_gpu: false,
            out_root: None,
            last_out: None,
            texture: None,
            rgba: vec![],
            display: MpDisplay::Phases,
            t_scale: 100.0,
            p_scale: 0.0,
            show_zones: true,
            show_effects: true,
            canvas: Rect::NOTHING,
            hovered: None,
            stats: MpStats::default(),
            stats_age: 0,
            tab: Tab::Protocol,
            tool: CiTool::Fill,
            shape: Shape::Brush,
            radius: 3,
            fluid: "eau".into(),
            solid: "acier".into(),
            temp: 293.15,
            thermal: SolidThermal::Passive,
            thermal_temp: 400.0,
            thermal_power: 1e6,
            sink_rate: 0.2,
            emit_v: [0.0, 1.0],
            grille_hole_mm: 10.0,
            grille_open: 0.6,
            stroke: vec![],
            drag_start: None,
            drag_now: None,
            last_insert: None,
            field_kind: FieldKind::T,
            field_name: "eau".into(),
            field_file: None,
            field_formula: "293.15 + 50*exp(-((x - lx/2)^2 + (y - ly/2)^2)/0.03^2)".into(),
            field_use_formula: true,
            field_top: true,
            field_range: false,
            field_a: 0.0,
            field_b: 1.0,
            field_in_rect: false,
            last_rect: None,
            plot_col: 2,
            status: "Écrivez ou chargez un protocole ; l'onglet « Conditions initiales » le complète au pinceau.".into(),
            show_help: false,
        };
        w.rebuild_preview();
        w
    }

    /// protocole de départ : le scénario et la grille de l'écran de démarrage
    fn template(s: &SimulationSettings) -> String {
        let p = s.precision.max(1);
        let (nx, ny) = ((s.width / p).max(16), (s.height / p).max(16));
        let g = if s.gravity > 0.0 { s.gravity as f64 } else { 9.81 };
        let sc = MpScenario::from_index(s.mp_scenario);
        format!(
            "# Expérience — format : voir l'aide (bouton « Aide du format ») ou EXPERIENCES.md\n\
             départ scénario \"{}\" grille {nx} x {ny} maille {} gravité {}\n\
             mesurer toutes les 10 ms : stats\n\
             # conditions initiales : onglet « Conditions initiales » (pinceaux, champs de valeurs), ou lignes remplir / champ…\n\
             simuler 1 s\n",
            sc.name(),
            fmt_num(settings_dx(s)),
            fmt_num(g)
        )
    }

    fn base_dir(&self) -> PathBuf {
        self.path.as_ref().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
    }

    fn stem(&self) -> String {
        self.path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "sans_titre".into())
    }

    /// Relit le protocole et reconstruit l'aperçu des conditions initiales (départ + actions avant le premier « simuler »).
    fn rebuild_preview(&mut self) {
        self.preview_src = self.text.clone();
        self.errors.clear();
        let protos = match parse_all(&self.text) {
            Ok(p) => p,
            Err(errs) => {
                self.errors = errs.iter().map(|e| e.to_string()).collect();
                self.parsed = None;
                return;
            }
        };
        let p = protos.into_iter().next().expect("au moins une variante");
        match initial_state(&p, &self.base_dir()) {
            Ok((sim, _, _)) => {
                let errs = crate::simulation::mp_experiment::check_names(&p, &sim);
                self.errors = errs.iter().map(|(l, e)| format!("ligne {l} : {e}")).collect();
                self.stats = compute_stats(&sim);
                self.preview = Some(sim);
            }
            Err(e) => self.errors.push(e),
        }
        self.parsed = Some(p);
    }

    fn running(&self) -> bool {
        self.runner.as_ref().is_some_and(|r| matches!(r.state, RunState::Running | RunState::Paused(_)))
    }

    /// simulation affichée : celle de l'expérience en cours (ou finie), sinon l'aperçu
    fn shown(&self) -> Option<&MultiphaseSim> {
        match &self.runner {
            Some(r) => Some(&r.sim),
            None => self.preview.as_ref(),
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> WindowAction {
        let mut action = WindowAction::None;
        // aperçu reconstruit un instant après la dernière frappe (pas pendant une expérience)
        if self.text != self.preview_src && self.runner.is_none() && self.edited_at.elapsed() >= REBUILD_DELAY {
            self.rebuild_preview();
        }
        self.shortcuts(ctx, &mut action);
        self.tick_run();
        self.top_bar(ctx, &mut action);
        egui::TopBottomPanel::bottom("exp_status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });
        self.left_panel(ctx);
        self.right_panel(ctx);
        self.central(ctx);
        egui::Window::new("Format des protocoles").open(&mut self.show_help).default_width(760.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().max_height(560.0).show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(SYNTAX_HELP).monospace()).wrap());
            });
        });
        ctx.request_repaint();
        action
    }

    fn shortcuts(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let pressed = |k: egui::Key| ctx.input(|i| i.key_pressed(k));
        if pressed(egui::Key::Space) {
            self.toggle_pause();
        }
        if pressed(egui::Key::Escape) {
            *action = WindowAction::BackToSettings;
        }
    }

    // ------------------------------------------------------------------------------------------------ exécution

    fn start_run(&mut self) {
        if self.running() {
            return;
        }
        let protos = match parse_all(&self.text) {
            Ok(p) => p,
            Err(errs) => {
                self.errors = errs.iter().map(|e| e.to_string()).collect();
                self.status = format!("Protocole à corriger : {}", self.errors[0]);
                return;
            }
        };
        self.n_variants = protos.len();
        self.queue = protos;
        self.queue.reverse();
        self.next_variant();
    }

    fn next_variant(&mut self) {
        let Some(p) = self.queue.pop() else { return };
        let base = self.base_dir();
        let root = self.out_root.clone().unwrap_or_else(|| base.join("résultats"));
        let dir = results_dir(&root, &self.stem(), &p.variant);
        let k = self.n_variants - self.queue.len();
        let label = p.variant.as_ref().map(|(n, v)| format!(" — variante {k}/{} : {n} = {v}", self.n_variants)).unwrap_or_default();
        match Runner::new(p, &base, Some(dir.clone()), Some(self.use_gpu)) {
            Ok(r) => {
                self.status = format!("Expérience lancée{label} ; résultats : {}", dir.display());
                self.last_out = Some(dir);
                self.plot_col = self.plot_col.min(r.columns.len().saturating_sub(1)).max(2.min(r.columns.len().saturating_sub(1)));
                self.runner = Some(r);
            }
            Err(e) => {
                self.queue.clear();
                self.errors = e.lines().map(String::from).collect();
                self.status = format!("Impossible de lancer : {}", self.errors.first().cloned().unwrap_or_default());
            }
        }
    }

    fn tick_run(&mut self) {
        let Some(r) = &mut self.runner else { return };
        if r.state == RunState::Running {
            r.advance(FRAME_BUDGET);
            match r.state.clone() {
                RunState::Finished => {
                    if self.queue.is_empty() {
                        self.status = format!(
                            "Terminé : t = {:.6} s, {} pas — résultats : {}",
                            r.sim.time,
                            r.sim.steps_done(),
                            self.last_out.as_ref().map(|d| d.display().to_string()).unwrap_or_default()
                        );
                    } else {
                        self.next_variant();
                    }
                }
                RunState::Failed(e) => self.status = format!("Échec : {e}"),
                RunState::Paused(why) => self.status = format!("En pause : {why} (Espace ou « Reprendre »)"),
                RunState::Running => {}
            }
        }
    }

    fn toggle_pause(&mut self) {
        if let Some(r) = &mut self.runner {
            match r.state {
                RunState::Running => {
                    r.pause("pause demandée");
                    self.status = "En pause".into();
                }
                RunState::Paused(_) => {
                    r.resume();
                    self.status = "Reprise".into();
                }
                _ => {}
            }
        }
    }

    fn stop_run(&mut self) {
        if let Some(r) = &mut self.runner {
            r.stop();
        }
        self.queue.clear();
        self.status = "Expérience arrêtée ; mesures déjà relevées et journal écrits".into();
    }

    /// quitte l'affichage de l'expérience : retour à l'aperçu des conditions initiales
    fn back_to_initial(&mut self) {
        if self.running() {
            self.stop_run();
        }
        self.runner = None;
        self.rebuild_preview();
    }

    // ------------------------------------------------------------------------------------------------ fichiers

    fn open(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("Protocole d'expérience", &["exp", "txt"]).pick_file() {
            match std::fs::read_to_string(&p) {
                Ok(t) => {
                    self.text = t.clone();
                    self.saved_text = t;
                    self.path = Some(p.clone());
                    self.runner = None;
                    self.last_insert = None;
                    self.rebuild_preview();
                    self.status = format!("Protocole ouvert : {}", p.display());
                }
                Err(e) => self.status = format!("Erreur de lecture : {e}"),
            }
        }
    }

    fn save(&mut self, ask: bool) {
        let path = match (&self.path, ask) {
            (Some(p), false) => Some(p.clone()),
            _ => rfd::FileDialog::new().add_filter("Protocole d'expérience", &["exp"]).set_file_name(format!("{}.exp", self.stem())).save_file(),
        };
        if let Some(p) = path {
            match std::fs::write(&p, &self.text) {
                Ok(()) => {
                    self.saved_text = self.text.clone();
                    self.path = Some(p.clone());
                    self.status = format!("Protocole enregistré : {}", p.display());
                    // les fichiers nommés par le protocole se lisent maintenant par rapport à son dossier
                    self.rebuild_preview();
                }
                Err(e) => self.status = format!("Erreur d'enregistrement : {e}"),
            }
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context, action: &mut WindowAction) {
        egui::TopBottomPanel::top("exp_top").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                let name = self.path.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "sans titre".into());
                let dirty = if self.text != self.saved_text { " •" } else { "" };
                ui.strong(format!("Expérience : {name}{dirty}"));
                ui.separator();
                ui.menu_button("Fichier", |ui| {
                    if ui.button("Nouveau protocole").clicked() {
                        ui.close_menu();
                        self.text = Self::template(&self.settings);
                        self.path = None;
                        self.saved_text.clear();
                        self.runner = None;
                        self.rebuild_preview();
                    }
                    if ui.button("Ouvrir…").clicked() {
                        ui.close_menu();
                        self.open();
                    }
                    if ui.button("Enregistrer").clicked() {
                        ui.close_menu();
                        self.save(false);
                    }
                    if ui.button("Enregistrer sous…").clicked() {
                        ui.close_menu();
                        self.save(true);
                    }
                    ui.separator();
                    if ui.add_enabled(self.last_out.is_some(), egui::Button::new("Ouvrir le dossier des résultats")).clicked() {
                        ui.close_menu();
                        if let Some(d) = &self.last_out {
                            open_folder(d);
                        }
                    }
                });
                ui.separator();
                let running = self.running();
                if ui.add_enabled(!running, egui::Button::new("▶ Lancer")).on_hover_text("Exécute le protocole depuis son départ (toutes les variantes d'un balayage)").clicked() {
                    self.start_run();
                }
                let paused = matches!(self.runner.as_ref().map(|r| &r.state), Some(RunState::Paused(_)));
                if ui.add_enabled(running, egui::Button::new(if paused { "Reprendre" } else { "Pause" })).on_hover_text("Espace").clicked() {
                    self.toggle_pause();
                }
                if ui.add_enabled(running, egui::Button::new("⏹ Arrêter")).clicked() {
                    self.stop_run();
                }
                if ui.add_enabled(self.runner.is_some(), egui::Button::new("Conditions initiales")).on_hover_text("Revenir à l'aperçu des conditions initiales").clicked() {
                    self.back_to_initial();
                }
                ui.add_enabled(!running, egui::Checkbox::new(&mut self.use_gpu, "GPU"))
                    .on_hover_text("Calcul GPU (f32) : plus rapide, sans garantie de résultats identiques au bit près (le CPU f64 la donne)");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Retour aux réglages").on_hover_text("Échap").clicked() {
                        *action = WindowAction::BackToSettings;
                    }
                    ui.toggle_value(&mut self.show_help, "Aide du format");
                });
            });
            ui.add_space(2.0);
        });
    }

    // ------------------------------------------------------------------------------------------------ panneau de gauche

    fn left_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("exp_left").default_width(470.0).min_width(330.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Protocol, egui::RichText::new("Protocole").strong());
                ui.selectable_value(&mut self.tab, Tab::Initial, egui::RichText::new("Conditions initiales").strong());
            });
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.tab {
                Tab::Protocol => self.protocol_tab(ui),
                Tab::Initial => self.initial_tab(ui),
            });
        });
    }

    fn protocol_tab(&mut self, ui: &mut egui::Ui) {
        let running = self.running();
        let edit = egui::TextEdit::multiline(&mut self.text).code_editor().desired_rows(22).desired_width(f32::INFINITY).interactive(!running);
        if ui.add(edit).changed() {
            self.edited_at = Instant::now();
            self.last_insert = None;
        }
        ui.horizontal(|ui| {
            if ui.button("Vérifier").on_hover_text("Relit le protocole et reconstruit les conditions initiales").clicked() {
                self.rebuild_preview();
                self.status = if self.errors.is_empty() { "Protocole correct".into() } else { format!("{} erreur(s)", self.errors.len()) };
            }
            ui.menu_button("Insérer…", |ui| {
                for (label, line) in [
                    ("mesurer", "mesurer toutes les 10 ms : stats, T en (0.1, 0.1)"),
                    ("simuler", "simuler 1 s"),
                    ("chauffer", "chauffer disque (0.1, 0.1) rayon 1 cm à 400 K pendant 1 s"),
                    ("accélérer", "accélérer rect (0, 0) (0.1, 0.05) de (5, 0) m/s² pendant 0.5 s"),
                    ("vitesse", "vitesse rect (0, 0) (0.1, 0.05) (1, 0) m/s"),
                    ("bord", "bord haut ouvert"),
                    ("sauver état", "sauver état \"etat\""),
                    ("marque", "marque \"repère\""),
                    ("pause", "pause"),
                    ("balayage", "balayage T0 = 300 K, 350 K, 400 K"),
                ] {
                    if ui.button(label).clicked() {
                        ui.close_menu();
                        if !self.text.ends_with('\n') && !self.text.is_empty() {
                            self.text.push('\n');
                        }
                        self.text.push_str(line);
                        self.text.push('\n');
                        self.edited_at = Instant::now();
                    }
                }
            });
            if ui.button("Aide du format").clicked() {
                self.show_help = true;
            }
        });
        if !self.errors.is_empty() {
            ui.add_space(4.0);
            for e in &self.errors {
                ui.colored_label(Color32::from_rgb(235, 90, 80), format!("✖ {e}"));
            }
        } else if let Some(p) = &self.parsed {
            let acts = p.steps.iter().filter(|(s, _)| matches!(s, Step::Do(_))).count();
            let ci = p.first_advance();
            ui.weak(format!(
                "{} conditions initiales, {} actions ensuite, {} mesure(s){}, fin prévue à t = {} s",
                ci,
                acts.saturating_sub(ci),
                p.measures.len(),
                p.period.map_or(String::new(), |t| format!(" toutes les {} s", fmt_num(t))),
                fmt_num(p.planned_end(0.0))
            ));
        }
        ui.add_space(6.0);
        let out = self.out_root.clone().unwrap_or_else(|| self.base_dir().join("résultats")).join(self.stem());
        ui.weak(format!("Résultats : {}", out.display()));
        ui.add_space(6.0);
        self.display_ui(ui);
    }

    fn display_ui(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(egui::RichText::new("Affichage").strong()).default_open(true).show(ui, |ui| {
            egui::ComboBox::from_label("Champ").selected_text(self.display.name()).show_ui(ui, |ui| {
                for m in MpDisplay::ALL {
                    ui.selectable_value(&mut self.display, m, m.name());
                }
            });
            if self.display == MpDisplay::Temperature {
                ui.add(egui::Slider::new(&mut self.t_scale, 1.0..=1000.0).logarithmic(true).text("Échelle ΔT (K)"));
            }
            ui.checkbox(&mut self.show_zones, "Zones du protocole sur l'image").on_hover_text("Actions (blanc), actions à durée (orange), sondes de mesure (cyan)");
            ui.checkbox(&mut self.show_effects, "Couleurs des cellules spéciales");
        });
    }

    fn names(&self) -> (Vec<String>, Vec<String>) {
        match self.shown() {
            Some(s) => {
                let mut f: Vec<String> = s.reg.phases.iter().map(|p| p.name.clone()).collect();
                f.extend(s.mixtures.iter().map(|m| m.0.clone()));
                (f, s.reg.solids.iter().map(|s| s.name.clone()).collect())
            }
            None => (vec![], vec![]),
        }
    }

    fn initial_tab(&mut self, ui: &mut egui::Ui) {
        if self.runner.is_some() {
            ui.weak("Une expérience est affichée : « Conditions initiales » (en haut) revient à l'aperçu pour le modifier.");
            ui.add_space(6.0);
        }
        let (fluids, solids) = self.names();
        ui.label("Chaque geste sur l'image ajoute une ligne au protocole, avant le premier « simuler ».");
        ui.add_space(4.0);
        egui::ComboBox::from_label("Outil").width(260.0).selected_text(self.tool.name()).show_ui(ui, |ui| {
            for t in CiTool::ALL {
                ui.selectable_value(&mut self.tool, t, t.name());
            }
        });
        ui.horizontal(|ui| {
            ui.label("Forme");
            ui.selectable_value(&mut self.shape, Shape::Brush, "Pinceau");
            ui.selectable_value(&mut self.shape, Shape::Rect, "Rectangle (glisser)");
        });
        let dx = self.shown().map_or(0.01, |s| s.dx);
        if self.shape == Shape::Brush {
            let r_m = fmt_num(self.radius as f64 * dx);
            ui.add(egui::Slider::new(&mut self.radius, 0..=40).text(format!("Rayon (mailles) = {r_m} m")));
        }
        let pick = |ui: &mut egui::Ui, label: &str, cur: &mut String, list: &[String]| {
            egui::ComboBox::from_label(label).width(220.0).selected_text(cur.clone()).show_ui(ui, |ui| {
                for n in list {
                    ui.selectable_value(cur, n.clone(), n);
                }
            });
        };
        match self.tool {
            CiTool::Fill | CiTool::Emitter => pick(ui, "Fluide", &mut self.fluid, &fluids),
            CiTool::Solid | CiTool::Body => pick(ui, "Matériau", &mut self.solid, &solids),
            _ => {}
        }
        if self.tool == CiTool::Body {
            ui.weak("Corps net aux bords lissés (fraction exacte de chaque maille) ; fixe pour l'instant — le mouvement viendra (SOLID.md).");
        }
        if matches!(self.tool, CiTool::Fill | CiTool::Temperature | CiTool::Emitter | CiTool::Solid) {
            ui.add(egui::Slider::new(&mut self.temp, 100.0..=3000.0).text("T (K)"));
        }
        match self.tool {
            CiTool::Solid => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.thermal, SolidThermal::Passive, "Passif");
                    ui.selectable_value(&mut self.thermal, SolidThermal::Isothermal, "Isotherme");
                    ui.selectable_value(&mut self.thermal, SolidThermal::Heater, "Chauffant");
                });
                match self.thermal {
                    SolidThermal::Isothermal => {
                        ui.add(egui::Slider::new(&mut self.thermal_temp, 100.0..=2000.0).text("T isotherme (K)"));
                    }
                    SolidThermal::Heater => {
                        ui.add(egui::Slider::new(&mut self.thermal_power, 1e2..=1e9).logarithmic(true).text("Puissance (W/m³)"));
                    }
                    SolidThermal::Passive => {}
                }
            }
            CiTool::Ignite => {
                ui.add(egui::Slider::new(&mut self.temp, 1200.0..=3000.0).text("T du noyau (K)"));
            }
            CiTool::Sink => {
                ui.add(egui::Slider::new(&mut self.sink_rate, 0.0..=10.0).text("Aspiration (m/s)"));
            }
            CiTool::Emitter => {
                ui.add(egui::Slider::new(&mut self.emit_v[0], -10.0..=10.0).text("Vitesse x (m/s)"));
                ui.add(egui::Slider::new(&mut self.emit_v[1], -10.0..=10.0).text("Vitesse y (m/s)"));
            }
            CiTool::Grille => {
                ui.add(egui::Slider::new(&mut self.grille_hole_mm, 0.5..=50.0).logarithmic(true).text("Trous (mm)"));
                ui.add(egui::Slider::new(&mut self.grille_open, 0.1..=0.9).text("Fraction ouverte"));
            }
            _ => {}
        }
        ui.horizontal(|ui| {
            if ui.add_enabled(self.last_insert.is_some(), egui::Button::new("Annuler le dernier geste")).clicked() {
                self.undo_last();
            }
        });

        ui.add_space(8.0);
        egui::CollapsingHeader::new(egui::RichText::new("Charger un champ de valeurs").strong()).default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Grandeur");
                ui.selectable_value(&mut self.field_kind, FieldKind::T, "T");
                ui.selectable_value(&mut self.field_kind, FieldKind::Vx, "vx");
                ui.selectable_value(&mut self.field_kind, FieldKind::Vy, "vy");
                ui.selectable_value(&mut self.field_kind, FieldKind::Fraction, "fraction");
                ui.selectable_value(&mut self.field_kind, FieldKind::Solid, "solide");
            });
            match self.field_kind {
                FieldKind::Fraction => pick(ui, "Fluide", &mut self.field_name, &fluids),
                FieldKind::Solid => pick(ui, "Matériau", &mut self.field_name, &solids),
                _ => {}
            }
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.field_use_formula, true, "Formule");
                ui.selectable_value(&mut self.field_use_formula, false, "Fichier CSV / PNG");
            });
            if self.field_use_formula {
                ui.add(egui::TextEdit::singleline(&mut self.field_formula).code_editor().desired_width(f32::INFINITY));
                ui.weak("En x, y (m), lx, ly ; exp, sqrt, sin, min, max, step, si(c, a, b)… Températures en K, vitesses en m/s, fractions de 0 à 1.");
            } else {
                ui.horizontal(|ui| {
                    if ui.button("Choisir…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().add_filter("Champ de valeurs", &["csv", "txt", "png"]).pick_file() {
                            self.field_file = Some(p);
                        }
                    }
                    ui.label(self.field_file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "aucun fichier".into()));
                });
                ui.horizontal(|ui| {
                    ui.label("Première rangée du fichier :");
                    ui.selectable_value(&mut self.field_top, true, "en haut");
                    ui.selectable_value(&mut self.field_top, false, "en bas");
                });
                ui.checkbox(&mut self.field_range, "Image : niveaux 0 (noir) et 1 (blanc) → valeurs a et b");
                if self.field_range {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.field_a).speed(0.1).prefix("a = "));
                        ui.add(egui::DragValue::new(&mut self.field_b).speed(0.1).prefix("b = "));
                    });
                }
                ui.weak("CSV : une rangée de valeurs par ligne, case vide = inchangé ; PNG : transparent = inchangé ; solide là où la valeur ≥ 0.5.");
            }
            ui.add_enabled(self.last_rect.is_some(), egui::Checkbox::new(&mut self.field_in_rect, "Dans le dernier rectangle tracé (sinon tout le domaine)"));
            if ui.add_enabled(self.runner.is_none(), egui::Button::new("Ajouter au protocole")).clicked() {
                self.add_field_line();
            }
        });
        ui.add_space(6.0);
        self.display_ui(ui);
    }

    // ------------------------------------------------------------------------------------------------ lignes des gestes

    fn temp_text(t: f64) -> String {
        format!("{} K", fmt_num(t))
    }

    /// ligne du protocole pour l'outil courant sur une zone
    fn tool_line(&self, zone: &Zone) -> String {
        let z = zone_text(zone);
        let t = Self::temp_text(self.temp);
        match self.tool {
            CiTool::Fill => format!("remplir {z} avec \"{}\" à {t}", self.fluid),
            CiTool::Temperature => format!("température {z} {t}"),
            CiTool::Solid => match self.thermal {
                SolidThermal::Passive => format!("solide {z} \"{}\" à {t}", self.solid),
                SolidThermal::Isothermal => format!("solide {z} \"{}\" isotherme {}", self.solid, Self::temp_text(self.thermal_temp)),
                SolidThermal::Heater => format!("solide {z} \"{}\" chauffant {} W/m³ à {t}", self.solid, fmt_num(self.thermal_power)),
            },
            CiTool::Body => match zone {
                Zone::Rect { a, b } => format!("corps rect {} {} \"{}\"", fmt_point(*a), fmt_point(*b), self.solid),
                Zone::Disque { c, r } => format!("corps disque {} rayon {} \"{}\"", fmt_point(*c), fmt_num(*r), self.solid),
                Zone::Trait { pts, r } if !pts.is_empty() => format!("corps disque {} rayon {} \"{}\"", fmt_point(pts[0]), fmt_num(*r), self.solid),
                _ => format!("# corps : forme non prise en charge ({z})"),
            },
            CiTool::Clear => format!("effacer {z}"),
            CiTool::Ignite => format!("allumer {z} à {t}"),
            CiTool::Sink => format!("puits {z} {} m/s", fmt_num(self.sink_rate)),
            CiTool::Emitter => format!("émetteur {z} \"{}\" {} m/s à {t}", self.fluid, fmt_point(self.emit_v)),
            CiTool::Grille => format!("grille {z} trous {} m ouverte {}", fmt_num(self.grille_hole_mm * 1e-3), fmt_num(self.grille_open)),
        }
    }

    /// action d'une ligne, lue par le même analyseur que le protocole
    fn line_action(line: &str) -> Result<Action, String> {
        let probe = format!("départ scénario \"0\"\n{line}\n");
        match parse_all(&probe) {
            Ok(mut v) => match v.remove(0).steps.into_iter().next() {
                Some((Step::Do(a), _)) => Ok(a),
                _ => Err("ligne sans action".into()),
            },
            Err(e) => Err(e.iter().map(|e| e.msg.clone()).collect::<Vec<_>>().join(" ; ")),
        }
    }

    /// ajoute une ligne de conditions initiales et reconstruit l'aperçu depuis le texte
    fn commit_line(&mut self, line: String) {
        let (t, n) = insert_initial_line(&self.text, &line);
        self.text = t;
        self.last_insert = Some((n, line.clone()));
        self.rebuild_preview();
        self.status = if self.errors.is_empty() { format!("Ligne {n} ajoutée : {line}") } else { format!("Ligne {n} ajoutée, mais : {}", self.errors[0]) };
    }

    fn undo_last(&mut self) {
        let Some((n, line)) = self.last_insert.take() else { return };
        let mut lines: Vec<&str> = self.text.lines().collect();
        if lines.get(n - 1) == Some(&line.as_str()) {
            lines.remove(n - 1);
            let mut t = lines.join("\n");
            t.push('\n');
            self.text = t;
            self.rebuild_preview();
            self.status = format!("Ligne {n} retirée");
        } else {
            self.status = "Le texte a changé depuis : rien à annuler".into();
        }
    }

    fn add_field_line(&mut self) {
        let q = match self.field_kind {
            FieldKind::T => "T".to_string(),
            FieldKind::Vx => "vx".into(),
            FieldKind::Vy => "vy".into(),
            FieldKind::Fraction => format!("fraction \"{}\"", self.field_name),
            FieldKind::Solid => format!("solide \"{}\"", self.field_name),
        };
        let src = if self.field_use_formula {
            if self.field_formula.contains('"') {
                self.status = "La formule ne doit pas contenir de guillemets".into();
                return;
            }
            format!("formule \"{}\"", self.field_formula.trim())
        } else {
            let Some(f) = &self.field_file else {
                self.status = "Choisissez d'abord un fichier".into();
                return;
            };
            // chemin relatif si le fichier est dans le dossier du protocole
            let base = self.base_dir();
            let shown = match f.strip_prefix(&base) {
                Ok(rel) => rel.display().to_string(),
                Err(_) => f.display().to_string(),
            };
            let mut s = format!("fichier \"{shown}\"");
            if !self.field_top {
                s.push_str(" origine bas");
            }
            if self.field_range {
                s.push_str(&format!(" valeurs {} {}", fmt_num(self.field_a), fmt_num(self.field_b)));
            }
            s
        };
        let zone = match (&self.last_rect, self.field_in_rect) {
            (Some(z), true) => format!(" dans {}", zone_text(z)),
            _ => String::new(),
        };
        self.commit_line(format!("champ {q} {src}{zone}"));
    }

    // ------------------------------------------------------------------------------------------------ panneau de droite

    fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("exp_right").default_width(330.0).min_width(250.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.heading("Expérience");
                match &self.runner {
                    None => {
                        ui.weak("Aperçu des conditions initiales (rien ne tourne).");
                    }
                    Some(r) => {
                        let state = match &r.state {
                            RunState::Running => "en cours".to_string(),
                            RunState::Paused(w) => format!("en pause ({w})"),
                            RunState::Finished => "terminée".into(),
                            RunState::Failed(e) => format!("échec : {e}"),
                        };
                        ui.label(format!("État : {state}"));
                        ui.add(egui::ProgressBar::new(r.progress() as f32).text(format!("t = {:.4} s / {} s", r.sim.time, fmt_num(r.t_end))));
                        let line = self.text.lines().nth(r.line.saturating_sub(1)).unwrap_or("").trim();
                        ui.label(format!("Ligne {} : {line}", r.line));
                        ui.weak(format!("{} pas · dt {:.2e} s · {}", r.sim.steps_done(), r.sim.last_dt, if r.gpu.is_some() { "GPU f32" } else { "CPU f64" }));
                    }
                }
                ui.separator();
                self.measures_ui(ui);
                ui.separator();
                egui::CollapsingHeader::new(egui::RichText::new("Maille sous le curseur").strong()).default_open(true).show(ui, |ui| {
                    match (self.hovered, self.shown()) {
                        (Some((i, j)), Some(s)) => {
                            let (x, y) = ((i as f64 + 0.5) * s.dx, (j as f64 + 0.5) * s.dx);
                            ui.label(format!("({}, {}) m — clic droit : copier", fmt_num(x), fmt_num(y)));
                            egui::Grid::new("exp_cell").num_columns(2).striped(true).show(ui, |ui| {
                                for (a, b) in cell_info(s, i, j) {
                                    ui.label(a);
                                    ui.label(b);
                                    ui.end_row();
                                }
                            });
                        }
                        _ => {
                            ui.weak("Survolez l'image");
                        }
                    }
                });
                if let Some(r) = &self.runner {
                    egui::CollapsingHeader::new(egui::RichText::new("Journal").strong()).default_open(false).show(ui, |ui| {
                        for l in r.log.iter().rev().take(40) {
                            ui.label(egui::RichText::new(l).monospace().size(11.0));
                        }
                    });
                }
                egui::CollapsingHeader::new(egui::RichText::new("Bilan").strong()).default_open(false).show(ui, |ui| {
                    let st = &self.stats;
                    egui::Grid::new("exp_stats").num_columns(2).striped(true).show(ui, |ui| {
                        for (a, b) in [
                            ("T min / max (K)", format!("{:.2} / {:.2}", st.t_min, st.t_max)),
                            ("p min / max (Pa)", format!("{:.0} / {:.0}", st.p_min, st.p_max)),
                            ("Vitesse max (m/s)", format!("{:.3}", st.speed_max)),
                            ("Énergie interne (J/m)", format!("{:.5e}", st.internal)),
                        ] {
                            ui.label(a);
                            ui.label(b);
                            ui.end_row();
                        }
                        for (n, m, _) in &st.phases {
                            ui.label(format!("masse {n} (kg/m)"));
                            ui.label(format!("{m:.4e}"));
                            ui.end_row();
                        }
                    });
                });
            });
        });
    }

    fn measures_ui(&mut self, ui: &mut egui::Ui) {
        let Some(r) = &self.runner else {
            if let Some(p) = &self.parsed {
                if !p.measures.is_empty() {
                    ui.weak(format!("{} mesure(s) au lancement.", p.measures.len()));
                }
            }
            return;
        };
        if r.rows.is_empty() || r.columns.len() <= 2 {
            ui.weak("Pas de mesure (ligne « mesurer toutes les … : … »).");
            return;
        }
        ui.strong("Mesures");
        let last = r.rows.last().unwrap();
        egui::Grid::new("exp_meas").num_columns(2).striped(true).show(ui, |ui| {
            for (k, c) in r.columns.iter().enumerate().take(12) {
                ui.label(c);
                ui.label(if k == 1 { format!("{}", last[k]) } else { format!("{:.6e}", last[k]) });
                ui.end_row();
            }
        });
        if r.columns.len() > 12 {
            ui.weak(format!("… et {} autres colonnes (mesures.csv)", r.columns.len() - 12));
        }
        let mut col = self.plot_col.min(r.columns.len() - 1).max(2);
        egui::ComboBox::from_label("Courbe").width(220.0).selected_text(r.columns[col].clone()).show_ui(ui, |ui| {
            for (k, c) in r.columns.iter().enumerate().skip(2) {
                ui.selectable_value(&mut col, k, c);
            }
        });
        self.plot_col = col;
        plot(ui, &r.rows, col);
    }

    // ------------------------------------------------------------------------------------------------ image

    fn central(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (outer, response) = ui.allocate_exact_size(avail, Sense::click_and_drag());
            let Some((nx, ny)) = self.shown().map(|s| (s.nx, s.ny)) else {
                ui.painter().rect_filled(outer, 0.0, Color32::from_gray(20));
                ui.painter().text(outer.center(), egui::Align2::CENTER_CENTER, "Pas d'aperçu : corrigez le protocole", egui::FontId::proportional(18.0), Color32::LIGHT_GRAY);
                return;
            };
            if outer.width() < 1.0 || outer.height() < 1.0 {
                return;
            }
            let scale = (outer.width() / nx as f32).min(outer.height() / ny as f32);
            let rect = Rect::from_center_size(outer.center(), egui::vec2(nx as f32 * scale, ny as f32 * scale));
            self.canvas = rect;
            self.mouse(ctx, &response, rect);
            self.stats_age += 1;
            if self.stats_age >= 20 {
                self.stats_age = 0;
                if let Some(r) = &mut self.runner {
                    r.sync_host();
                }
                if let Some(s) = self.shown() {
                    self.stats = compute_stats(s);
                }
            }
            self.update_texture(ctx);
            let painter = ui.painter_at(outer);
            painter.rect_filled(outer, 0.0, Color32::BLACK);
            if let Some(tex) = &self.texture {
                painter.image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, egui::pos2(1.0, 1.0)), Color32::WHITE);
            }
            if self.show_zones {
                self.draw_zones(&painter, rect);
            }
            self.draw_gesture(&painter, rect);
        });
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        let o = RenderOpts { display: self.display, t_scale: self.t_scale, p_scale: self.p_scale, show_effects: self.show_effects, solids_white: false };
        let (nx, ny) = match self.shown() {
            Some(s) => (s.nx, s.ny),
            None => return,
        };
        match &mut self.runner {
            Some(r) => match &mut r.gpu {
                Some(g) => g.render(&r.sim, &o, &mut self.rgba),
                None => render_rgba(&r.sim, o, &mut self.rgba),
            },
            None => {
                if let Some(s) = &self.preview {
                    render_rgba(s, o, &mut self.rgba);
                }
            }
        }
        if self.rgba.len() != nx * ny * 4 {
            return;
        }
        let image = ColorImage::from_rgba_unmultiplied([nx, ny], &self.rgba);
        match &mut self.texture {
            Some(t) => t.set(image, TextureOptions::NEAREST),
            None => self.texture = Some(ctx.load_texture("experience", image, TextureOptions::NEAREST)),
        }
    }

    /// maille sous un point de l'écran
    fn cell_at(&self, p: Pos2, rect: Rect) -> Option<(usize, usize)> {
        let s = self.shown()?;
        if !rect.contains(p) {
            return None;
        }
        let u = ((p.x - rect.min.x) / rect.width()).clamp(0.0, 0.9999);
        let v = ((p.y - rect.min.y) / rect.height()).clamp(0.0, 0.9999);
        Some(((u * s.nx as f32) as usize, s.ny - 1 - (v * s.ny as f32) as usize))
    }

    fn center_of(&self, c: (usize, usize)) -> [f64; 2] {
        let dx = self.shown().map_or(0.01, |s| s.dx);
        [(c.0 as f64 + 0.5) * dx, (c.1 as f64 + 0.5) * dx]
    }

    fn mouse(&mut self, ctx: &egui::Context, response: &egui::Response, rect: Rect) {
        let (pos, left_down, left_released) = ctx.input(|i| (i.pointer.latest_pos(), i.pointer.button_down(PointerButton::Primary), i.pointer.button_released(PointerButton::Primary)));
        let over = pos.filter(|&p| response.contains_pointer() && rect.contains(p));
        self.hovered = over.and_then(|p| self.cell_at(p, rect));
        // clic droit : coordonnées dans le presse-papiers
        if response.secondary_clicked() {
            if let Some(c) = self.hovered {
                let t = fmt_point(self.center_of(c));
                ctx.copy_text(t.clone());
                self.status = format!("Coordonnées copiées : {t}");
            }
        }
        let editable = self.tab == Tab::Initial && self.runner.is_none() && self.preview.is_some();
        if !editable {
            self.stroke.clear();
            self.drag_start = None;
            return;
        }
        if left_down {
            if let Some(c) = self.hovered {
                match self.shape {
                    Shape::Brush if self.tool == CiTool::Body => {
                        // corps : un disque au point d'appui (pas de trait)
                        if self.stroke.is_empty() {
                            let p = self.center_of(c);
                            self.stroke.push(p);
                        }
                    }
                    Shape::Brush => {
                        let p = self.center_of(c);
                        if self.stroke.last() != Some(&p) {
                            // retour immédiat : le segment depuis le point précédent est posé sur l'aperçu
                            let seg = match self.stroke.last() {
                                Some(&q) => vec![q, p],
                                None => vec![p],
                            };
                            self.stroke.push(p);
                            let r = self.radius as f64 * self.preview.as_ref().map_or(0.01, |s| s.dx);
                            let line = self.tool_line(&Zone::Trait { pts: seg, r });
                            let base = self.base_dir();
                            if let (Ok(a), Some(sim)) = (Self::line_action(&line), self.preview.as_mut()) {
                                let _ = apply_action(sim, &a, &base);
                            }
                        }
                    }
                    Shape::Rect => {
                        if self.drag_start.is_none() {
                            self.drag_start = Some(c);
                        }
                        self.drag_now = Some(c);
                    }
                }
            }
        }
        if left_released {
            match self.shape {
                Shape::Brush => {
                    if !self.stroke.is_empty() {
                        let dx = self.preview.as_ref().map_or(0.01, |s| s.dx);
                        let body = self.tool == CiTool::Body;
                        let r = if body { self.radius.max(1) } else { self.radius } as f64 * dx;
                        let pts = std::mem::take(&mut self.stroke);
                        let zone = if pts.len() == 1 || body { Zone::Disque { c: pts[0], r } } else { Zone::Trait { pts, r } };
                        let line = self.tool_line(&zone);
                        self.commit_line(line);
                    }
                }
                Shape::Rect => {
                    if let (Some(a), Some(b)) = (self.drag_start.take(), self.drag_now.take()) {
                        let dx = self.preview.as_ref().map_or(0.01, |s| s.dx);
                        let (i0, i1) = (a.0.min(b.0), a.0.max(b.0));
                        let (j0, j1) = (a.1.min(b.1), a.1.max(b.1));
                        let zone = Zone::Rect { a: [i0 as f64 * dx, j0 as f64 * dx], b: [(i1 + 1) as f64 * dx, (j1 + 1) as f64 * dx] };
                        self.last_rect = Some(zone.clone());
                        let line = self.tool_line(&zone);
                        self.commit_line(line);
                    }
                }
            }
        }
    }

    fn to_screen(&self, rect: Rect, p: [f64; 2]) -> Pos2 {
        let (lx, ly) = self.shown().map_or((1.0, 1.0), |s| (s.nx as f64 * s.dx, s.ny as f64 * s.dx));
        egui::pos2(rect.min.x + (p[0] / lx) as f32 * rect.width(), rect.max.y - (p[1] / ly) as f32 * rect.height())
    }

    fn draw_zone(&self, painter: &egui::Painter, rect: Rect, z: &Zone, color: Color32) {
        let px = rect.width() / self.shown().map_or(1.0, |s| (s.nx as f64 * s.dx) as f32);
        let st = Stroke::new(1.5_f32, color);
        match z {
            Zone::Partout => {}
            Zone::Point(p) => {
                let c = self.to_screen(rect, *p);
                painter.line_segment([c - egui::vec2(5.0, 0.0), c + egui::vec2(5.0, 0.0)], st);
                painter.line_segment([c - egui::vec2(0.0, 5.0), c + egui::vec2(0.0, 5.0)], st);
            }
            Zone::Disque { c, r } => {
                painter.circle_stroke(self.to_screen(rect, *c), (*r as f32 * px).max(2.0), st);
            }
            Zone::Rect { a, b } => {
                painter.rect_stroke(Rect::from_two_pos(self.to_screen(rect, *a), self.to_screen(rect, *b)), 0.0, st, egui::StrokeKind::Middle);
            }
            Zone::Trait { pts, r } => {
                let line: Vec<Pos2> = pts.iter().map(|p| self.to_screen(rect, *p)).collect();
                painter.add(egui::Shape::line(line, Stroke::new((2.0 * *r as f32 * px).max(1.5), color.gamma_multiply(0.35))));
            }
        }
    }

    fn draw_zones(&self, painter: &egui::Painter, rect: Rect) {
        let Some(p) = &self.parsed else { return };
        // corps rigides : leur contour à leur position courante (ils peuvent bouger pendant l'expérience)
        if let Some(s) = self.shown() {
            for b in &s.bodies {
                let mut pts: Vec<Pos2> = b.world_vertices().iter().map(|p| self.to_screen(rect, *p)).collect();
                if let Some(&f) = pts.first() {
                    pts.push(f);
                }
                painter.add(egui::Shape::line(pts, Stroke::new(1.5_f32, Color32::from_rgb(150, 190, 255))));
            }
        }
        for (s, _) in &p.steps {
            if let Step::Do(a) = s {
                if let Some(z) = a.zone() {
                    let col = if matches!(a, Action::Heat { .. } | Action::Accelerate { .. }) { Color32::from_rgb(255, 160, 40) } else { Color32::from_gray(220) };
                    self.draw_zone(painter, rect, z, col);
                }
            }
        }
        for (m, _) in &p.measures {
            match m {
                Measure::Probe { zone, .. } | Measure::Mass { zone, .. } => self.draw_zone(painter, rect, zone, Color32::from_rgb(60, 220, 230)),
                Measure::Stats => {}
            }
        }
    }

    fn draw_gesture(&self, painter: &egui::Painter, rect: Rect) {
        let dx = self.shown().map_or(0.01, |s| s.dx);
        if let (Shape::Rect, Some(a), Some(b)) = (self.shape, self.drag_start, self.drag_now) {
            let z = Zone::Rect {
                a: [a.0.min(b.0) as f64 * dx, a.1.min(b.1) as f64 * dx],
                b: [(a.0.max(b.0) + 1) as f64 * dx, (a.1.max(b.1) + 1) as f64 * dx],
            };
            self.draw_zone(painter, rect, &z, Color32::YELLOW);
        }
        if self.tab == Tab::Initial && self.runner.is_none() && self.shape == Shape::Brush {
            if let Some(c) = self.hovered {
                self.draw_zone(painter, rect, &Zone::Disque { c: self.center_of(c), r: (self.radius as f64 + 0.5) * dx }, Color32::YELLOW);
            }
        }
    }
}

/// petite courbe valeur(t) d'une colonne des mesures
fn plot(ui: &mut egui::Ui, rows: &[Vec<f64>], col: usize) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 150.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_gray(25));
    let pts: Vec<(f64, f64)> = rows.iter().filter(|r| r[col].is_finite()).map(|r| (r[0], r[col])).collect();
    if pts.len() < 2 {
        return;
    }
    let (t0, t1) = (pts[0].0, pts[pts.len() - 1].0);
    let (mut v0, mut v1) = pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
    if v1 - v0 < 1e-300 {
        v0 -= 0.5;
        v1 += 0.5;
    }
    let inner = rect.shrink2(egui::vec2(6.0, 16.0));
    let map = |t: f64, v: f64| {
        egui::pos2(
            inner.min.x + ((t - t0) / (t1 - t0).max(1e-300)) as f32 * inner.width(),
            inner.max.y - ((v - v0) / (v1 - v0)) as f32 * inner.height(),
        )
    };
    let line: Vec<Pos2> = pts.iter().map(|&(t, v)| map(t, v)).collect();
    painter.add(egui::Shape::line(line, Stroke::new(1.5_f32, Color32::from_rgb(90, 200, 255))));
    let font = egui::FontId::monospace(10.0);
    painter.text(rect.left_top() + egui::vec2(4.0, 2.0), egui::Align2::LEFT_TOP, format!("{v1:.4e}"), font.clone(), Color32::GRAY);
    painter.text(rect.left_bottom() + egui::vec2(4.0, -2.0), egui::Align2::LEFT_BOTTOM, format!("{v0:.4e}"), font.clone(), Color32::GRAY);
    painter.text(rect.right_bottom() + egui::vec2(-4.0, -2.0), egui::Align2::RIGHT_BOTTOM, format!("t = {} s", fmt_num(t1)), font, Color32::GRAY);
}

fn open_folder(d: &Path) {
    #[cfg(target_os = "windows")]
    let cmd = "explorer";
    #[cfg(target_os = "macos")]
    let cmd = "open";
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let cmd = "xdg-open";
    let _ = std::process::Command::new(cmd).arg(d).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2, Event, Modifiers, RawInput};

    fn frame(ctx: &egui::Context, w: &mut ExperimentWindow, events: Vec<Event>) {
        let input = RawInput { screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(1600.0, 950.0))), events, ..Default::default() };
        let _ = ctx.run(input, |ctx| {
            w.ui(ctx);
        });
    }

    fn window(text: &str) -> ExperimentWindow {
        let mut w = ExperimentWindow::new(&SimulationSettings { multiphase: true, ..Default::default() });
        w.text = text.into();
        w.rebuild_preview();
        assert!(w.errors.is_empty(), "{:?}", w.errors);
        w
    }

    /// centre écran de la maille (i, j)
    fn at(w: &ExperimentWindow, i: usize, j: usize) -> Pos2 {
        let s = w.preview.as_ref().unwrap();
        let r = w.canvas;
        pos2(r.min.x + (i as f32 + 0.5) / s.nx as f32 * r.width(), r.max.y - (j as f32 + 0.5) / s.ny as f32 * r.height())
    }

    fn drag(ctx: &egui::Context, w: &mut ExperimentWindow, path: &[Pos2]) {
        let b = |p: Pos2, pressed: bool| Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::default() };
        frame(ctx, w, vec![Event::PointerMoved(path[0])]);
        frame(ctx, w, vec![b(path[0], true)]);
        for &p in &path[1..] {
            frame(ctx, w, vec![Event::PointerMoved(p)]);
        }
        let last = *path.last().unwrap();
        frame(ctx, w, vec![b(last, false)]);
        frame(ctx, w, vec![]);
    }

    #[test]
    fn a_brush_stroke_becomes_a_protocol_line_and_the_preview_is_what_the_line_rebuilds() {
        let ctx = egui::Context::default();
        let mut w = window("départ scénario \"cuve vide\" grille 40 x 30 maille 1 cm\nsimuler 10 ms\n");
        w.tab = Tab::Initial;
        w.tool = CiTool::Fill;
        w.fluid = "eau".into();
        w.radius = 2;
        frame(&ctx, &mut w, vec![]);
        frame(&ctx, &mut w, vec![]);
        // trait rapide (des mailles sautées entre deux images) : la ligne couvre tout le trait
        let path = [at(&w, 5, 5), at(&w, 15, 8), at(&w, 30, 8)];
        drag(&ctx, &mut w, &path);
        let line = w.text.lines().nth(1).unwrap().to_string();
        assert!(line.starts_with("remplir trait (0.055, 0.055) (0.155, 0.085) (0.305, 0.085) rayon 0.02 avec \"eau\""), "{}", w.text);
        assert_eq!(w.text.lines().nth(2), Some("simuler 10 ms"));
        let eau = w.preview.as_ref().unwrap().reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let shown = w.preview.as_ref().unwrap().mass.clone();
        assert!(shown[eau][w.preview.as_ref().unwrap().idx(22, 8)] > 900.0, "milieu du trait sans eau");
        // aperçu = protocole relu
        let p = parse_all(&w.text).unwrap().remove(0);
        let (s, _, _) = initial_state(&p, Path::new(".")).unwrap();
        assert_eq!(s.mass, shown);
        // annuler le geste
        w.undo_last();
        assert_eq!(w.text.lines().count(), 2);
        assert!(w.preview.as_ref().unwrap().mass[eau].iter().all(|&m| m == 0.0));
    }

    #[test]
    fn a_rectangle_and_a_value_field_are_added_as_lines() {
        let ctx = egui::Context::default();
        let mut w = window("départ scénario \"cuve vide\" grille 40 x 30 maille 1 cm\nsimuler 10 ms\n");
        w.tab = Tab::Initial;
        w.shape = Shape::Rect;
        w.tool = CiTool::Solid;
        w.solid = "acier".into();
        frame(&ctx, &mut w, vec![]);
        frame(&ctx, &mut w, vec![]);
        let path = [at(&w, 2, 3), at(&w, 9, 4)];
        drag(&ctx, &mut w, &path);
        assert!(w.text.contains("solide rect (0.02, 0.03) (0.1, 0.05) \"acier\""), "{}", w.text);
        let s = w.preview.as_ref().unwrap();
        assert_eq!(s.solid.iter().filter(|&&k| k >= 0).count(), 8 * 2);
        // champ de température dans ce rectangle (formule)
        w.field_kind = FieldKind::T;
        w.field_use_formula = true;
        w.field_formula = "500".into();
        w.field_in_rect = true;
        w.add_field_line();
        assert!(w.text.contains("champ T formule \"500\" dans rect (0.02, 0.03) (0.1, 0.05)"), "{}", w.text);
        let s = w.preview.as_ref().unwrap();
        assert_eq!(s.temp[s.idx(5, 3)], 500.0);
        assert!(s.temp[s.idx(20, 20)] < 400.0);
        assert!(w.errors.is_empty(), "{:?}", w.errors);
    }

    #[test]
    fn the_body_tool_places_a_fixed_rigid_body_by_click_or_rectangle() {
        let ctx = egui::Context::default();
        let mut w = window("départ scénario \"cuve vide\" grille 40 x 30 maille 1 cm\nsimuler 10 ms\n");
        w.tab = Tab::Initial;
        w.tool = CiTool::Body;
        w.solid = "acier".into();
        w.radius = 3;
        frame(&ctx, &mut w, vec![]);
        frame(&ctx, &mut w, vec![]);
        // clic (même glissé) : un disque au point d'appui
        let path = [at(&w, 10, 10), at(&w, 14, 12)];
        drag(&ctx, &mut w, &path);
        assert!(w.text.contains("corps disque (0.105, 0.105) rayon 0.03 \"acier\""), "{}", w.text);
        assert_eq!(w.preview.as_ref().unwrap().bodies.len(), 1);
        // rectangle tiré
        w.shape = Shape::Rect;
        w.solid = "verre".into();
        let path = [at(&w, 25, 5), at(&w, 30, 8)];
        drag(&ctx, &mut w, &path);
        assert!(w.text.contains("corps rect (0.25, 0.05) (0.31, 0.09) \"verre\""), "{}", w.text);
        let s = w.preview.as_ref().unwrap();
        assert_eq!(s.bodies.len(), 2);
        let area: f64 = s.body_frac.iter().sum::<f64>() * s.dx * s.dx;
        assert!((area - (std::f64::consts::PI * 9e-4 + 0.06 * 0.04)).abs() < 1e-9, "aire {area}");
        assert!(w.errors.is_empty(), "{:?}", w.errors);
        // l'image se rend avec les corps
        for d in [MpDisplay::Phases, MpDisplay::Temperature] {
            w.display = d;
            frame(&ctx, &mut w, vec![]);
        }
    }

    #[test]
    fn the_window_runs_a_protocol_to_the_end_with_its_measures() {
        let ctx = egui::Context::default();
        let dir = std::env::temp_dir().join(format!("fluid-sim-expwin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut w = window(
            "départ scénario \"barrage\" grille 32 x 24 maille 1 cm\nmesurer toutes les 5 ms : stats, T en (0.05, 0.05)\nsimuler 20 ms\npause\nchauffer partout à 320 K pendant 10 ms\nsimuler 10 ms\n",
        );
        w.out_root = Some(dir.clone());
        frame(&ctx, &mut w, vec![]);
        w.start_run();
        assert!(w.running(), "{}", w.status);
        let mut paused = false;
        for _ in 0..400 {
            frame(&ctx, &mut w, vec![]);
            let st = w.runner.as_ref().unwrap().state.clone();
            if let RunState::Paused(_) = st {
                paused = true;
                // tous les modes d'affichage rendent pendant la pause
                for d in MpDisplay::ALL {
                    w.display = d;
                    frame(&ctx, &mut w, vec![]);
                }
                w.toggle_pause();
            }
            if st == RunState::Finished {
                break;
            }
        }
        let r = w.runner.as_ref().unwrap();
        assert_eq!(r.state, RunState::Finished, "{}", w.status);
        assert!(paused, "la ligne « pause » n'a pas mis l'expérience en pause");
        assert!((r.sim.time - 0.03).abs() < 1e-8);
        let t: Vec<f64> = r.rows.iter().map(|row| row[0]).collect();
        assert!(t.len() >= 4 && t[0] == 0.0 && (t.last().unwrap() - 0.03).abs() < 1e-8, "relevés : {t:?}");
        assert!(dir.join("sans_titre").join("mesures.csv").exists());
        assert!(w.status.starts_with("Terminé"), "{}", w.status);
        // retour aux conditions initiales
        w.back_to_initial();
        assert!(w.runner.is_none() && w.preview.as_ref().unwrap().time == 0.0);
    }
}
