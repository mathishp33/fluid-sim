use egui::ComboBox;
use serde::{Deserialize, Serialize};

use crate::simulation::mono::fluid::{Boundary, BoundaryType};
use crate::simulation::multi::mp_materials::MaterialLibrary;
use crate::simulation::multi::mp_view::{build_scenario_with, MpScenario};
use crate::window::material_editor::MaterialEditor;

const CELL_SIZES: [(f32, &str); 4] = [(1e-3, "1 mm"), (1e-2, "1 cm"), (1e-1, "10 cm"), (1.0, "1 m")];

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)] // un config.json ancien ou incomplet ne casse plus le chargement
pub struct SimulationSettings {
    pub width: usize,
    pub height: usize,
    pub dx: f32, // m / cellule
    pub gravity: f32, // m/s²
    pub precision: usize,
    pub left_boundary: Boundary,
    pub right_boundary: Boundary,
    pub top_boundary: Boundary,
    pub bottom_boundary: Boundary,
    /// indice du scénario multiphasique (voir `MpScenario::ALL`)
    pub mp_scenario: usize,
    /// matériaux du mode multiphasique (fluides, solides, miscibilité, changements de phase, réactions) : modifiables, persistés ici
    pub materials: MaterialLibrary,
    /// nom du fluide qui entre par un bord réglé sur « Inlet » en mode multiphasique
    pub mp_inlet_fluid: String,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            dx: 1e-2,
            gravity: 0.0,
            precision: 10,
            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),
            mp_scenario: 0,
            materials: MaterialLibrary::default(),
            mp_inlet_fluid: "eau".into(),
        }
    }
}

pub struct StartingGui {
    pub settings: SimulationSettings,
    editor: MaterialEditor,
}

/// Ce que lance l'écran de démarrage : la fenêtre de simulation, ou la fenêtre d'expérience (protocoles, mesures)
pub enum Launch {
    Simulation(SimulationSettings),
    Experiment(SimulationSettings),
}

impl StartingGui {
    pub fn new(settings: SimulationSettings) -> Self {
        Self { settings, editor: MaterialEditor::default() }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> Option<Launch> {
        let mut launch = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Fluid Simulation Settings");
                ui.add_space(8.0);

                let sc = MpScenario::from_index(self.settings.mp_scenario);
                ComboBox::from_label("Scenario").selected_text(sc.name()).show_ui(ui, |ui| {
                    for (i, s) in MpScenario::ALL.iter().enumerate() {
                        ui.selectable_value(&mut self.settings.mp_scenario, i, s.name());
                    }
                });
                ui.weak("Uses width, height, precision, cell size and gravity below (gravity 0 → 9.81 m/s²). A boundary set to Outlet opens to 1 atm; Inlet injects the fluid chosen below at the given velocity. Typical: cell size 1 cm, precision 10.");
                let names: Vec<String> = self.settings.materials.fluids.iter().map(|p| p.name.clone()).collect();
                ComboBox::from_label("Inlet fluid").selected_text(self.settings.mp_inlet_fluid.clone()).show_ui(ui, |ui| {
                    for n in &names {
                        ui.selectable_value(&mut self.settings.mp_inlet_fluid, n.clone(), n);
                    }
                });
                ui.collapsing("Materials (fluids, solids, miscibility, phase change, reactions)", |ui| {
                    self.editor.ui(ui, &mut self.settings.materials);
                });
                ui.add_space(4.0);

                ui.add(egui::Slider::new(&mut self.settings.width, 100..=1920).text("Width"));
                ui.add(egui::Slider::new(&mut self.settings.height, 100..=1080).text("Height"));

                ui.add(egui::Slider::new(&mut self.settings.gravity, 0.0..=30.0).text("Gravity (m/s²)"));

                let current = CELL_SIZES.iter()
                    .find(|(v, _)| (*v - self.settings.dx).abs() < 1e-9)
                    .map(|(_, n)| n.to_string())
                    .unwrap_or_else(|| format!("{} m", self.settings.dx));
                ComboBox::from_label("Cell size").selected_text(current)
                    .show_ui(ui, |ui| {
                        for (v, n) in CELL_SIZES {
                            ui.selectable_value(&mut self.settings.dx, v, n);
                        }
                    });
                let p = self.settings.precision.max(1) as f32;
                ui.label(format!("Domaine : {:.2} × {:.2} m",
                                 self.settings.width as f32 / p * self.settings.dx,
                                 self.settings.height as f32 / p * self.settings.dx));

                ComboBox::from_label("Precision (pixels)")
                    .selected_text(format!("{}", self.settings.precision))
                    .show_ui(ui, |ui| {
                        for level in [1, 2, 5, 10, 20] {
                            ui.selectable_value(&mut self.settings.precision, level, format!("{level}"));
                        }
                    });

                let dx = self.settings.dx;
                boundary_ui(ui, "Left Boundary", &mut self.settings.left_boundary, dx);
                boundary_ui(ui, "Right Boundary", &mut self.settings.right_boundary, dx);
                boundary_ui(ui, "Top Boundary", &mut self.settings.top_boundary, dx);
                boundary_ui(ui, "Bottom Boundary", &mut self.settings.bottom_boundary, dx);

                ui.add_space(8.0);
                // le scénario choisi doit pouvoir être construit avec les matériaux courants (sinon on dit lequel manque)
                let problem = build_scenario_with(&self.settings.materials, MpScenario::from_index(self.settings.mp_scenario), 8, 8, 0.01, 9.81).err();
                if let Some(e) = &problem {
                    ui.colored_label(egui::Color32::from_rgb(230, 80, 80), format!("✖ {e}"));
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(problem.is_none(), egui::Button::new("Launch Simulation")).clicked() {
                        save_settings(&self.settings);
                        launch = Some(Launch::Simulation(self.settings.clone()));
                    }
                    if ui
                        .button("Fenêtre d'expérience")
                        .on_hover_text(
                            "Protocole écrit (conditions initiales, actions en temps simulé, mesures) exécuté pas à pas, résultats en CSV ; \
                             part du scénario, de la grille et de la gravité choisis ici (solveur multiphasique)",
                        )
                        .clicked()
                    {
                        save_settings(&self.settings);
                        launch = Some(Launch::Experiment(self.settings.clone()));
                    }
                });
            });
        });

        launch
    }
}

/// Vitesses en m/s (SI). `dx` (m/cellule) sert à afficher l'offset en mm.
pub fn boundary_ui(ui: &mut egui::Ui, name: &str, boundary: &mut Boundary, dx: f32) {
    ui.collapsing(name, |ui| {
        ComboBox::from_label("Type")
            .selected_text(format!("{:?}", boundary.boundary_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut boundary.boundary_type, BoundaryType::Wall, "Wall");
                ui.selectable_value(&mut boundary.boundary_type, BoundaryType::Inlet, "Inlet");
                ui.selectable_value(&mut boundary.boundary_type, BoundaryType::Outlet, "Outlet");
            });

        match boundary.boundary_type {
            BoundaryType::Inlet => {
                ui.add(egui::Slider::new(&mut boundary.velocity_x_ms, -10.0..=10.0).text("Velocity X (m/s)"));
                ui.add(egui::Slider::new(&mut boundary.velocity_y_ms, -10.0..=10.0).text("Velocity Y (m/s)"));
                ui.add(egui::Slider::new(&mut boundary.offset, 0..=500).text("Offset (cells)"));
                ui.weak(format!("offset = {:.1} mm", boundary.offset as f32 * dx * 1000.0));
                ui.add(egui::Slider::new(&mut boundary.temperature, 100.0..=2000.0).text("Temperature (K)"));
            }
            BoundaryType::Wall => {}
            BoundaryType::Outlet => {}
        }
    });
}

pub fn load_settings() -> SimulationSettings {
    let path = "config.json";

    match std::fs::read_to_string(path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_else(|e| {
            eprintln!("Unable to read config.json : {e}");
            SimulationSettings::default()
        }),

        Err(_) => {
            println!("No config found, using default values ...");
            SimulationSettings::default()
        }
    }
}

pub fn save_settings(settings: &SimulationSettings) {
    let path = "config.json";

    match serde_json::to_string_pretty(settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("unable to save config.json : {e}");
            }
        }

        Err(e) => {
            eprintln!("impossible to serialize config : {e}");
        }
    }
}
