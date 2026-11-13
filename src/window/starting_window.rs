use egui::ComboBox;
use serde::{Deserialize, Serialize};

use crate::simulation::fluid::{Boundary, BoundaryType};
use crate::simulation::materials::{FluidKind, FluidMaterial};

const CELL_SIZES: [(f32, &str); 4] = [(1e-3, "1 mm"), (1e-2, "1 cm"), (1e-1, "10 cm"), (1.0, "1 m")];

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)] // un config.json ancien ou incomplet ne casse plus le chargement
pub struct SimulationSettings {
    pub width: usize,
    pub height: usize,
    pub material: FluidMaterial,
    pub dx: f32, // m / cellule
    pub gravity: f32, // m/s^2
    pub precision: usize,
    #[serde(alias = "max_density_color")]
    pub max_tracer_color: [u8; 4],
    pub pressure_iters: usize,
    pub left_boundary: Boundary,
    pub right_boundary: Boundary,
    pub top_boundary: Boundary,
    pub bottom_boundary: Boundary,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            material: FluidMaterial::water(),
            dx: 1e-2,
            gravity: 0.0,
            precision: 10,
            max_tracer_color: [255, 255, 255, 255],
            pressure_iters: 30,
            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),
        }
    }
}

impl SimulationSettings {
    pub fn max_color(&self) -> u32 {
        let c = self.max_tracer_color;
        ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
    }
}

pub struct StartingGui {
    pub settings: SimulationSettings,
}

impl StartingGui {
    pub fn new(settings: SimulationSettings) -> Self {
        Self { settings }
    }

    pub fn ui(&mut self, ctx: &egui::Context) -> Option<SimulationSettings> {
        let mut launch = None;

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Fluid Simulation Settings");
                ui.add_space(8.0);

                ui.add(egui::Slider::new(&mut self.settings.width, 100..=1920).text("Width"));
                ui.add(egui::Slider::new(&mut self.settings.height, 100..=1080).text("Height"));

                let kind_name = match self.settings.material.kind {
                    FluidKind::Gas => "Air (gas, low-Mach)",
                    FluidKind::Liquid => "Water (liquid)",
                };
                ComboBox::from_label("Fluid preset").selected_text(kind_name)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(false, "Air (gas, low-Mach)").clicked() { self.settings.material = FluidMaterial::air(); }
                        if ui.selectable_label(false, "Water (liquid)").clicked() { self.settings.material = FluidMaterial::water(); }
                    });
                ui.add(egui::Slider::new(&mut self.settings.material.viscosity, 1e-7..=1e-3).logarithmic(true).text("ν (m²/s)"));
                ui.add(egui::Slider::new(&mut self.settings.material.conductivity, 0.01..=400.0).logarithmic(true).text("k fluid (W/m/K)"));
                ui.add(egui::Slider::new(&mut self.settings.material.solid_conductivity, 0.01..=400.0).logarithmic(true).text("k solid (W/m/K)"));
                ui.add(egui::Slider::new(&mut self.settings.material.t_ref, 200.0..=500.0).text("T ambient (K)"));
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

                ui.add(egui::Slider::new(&mut self.settings.pressure_iters, 0..=200).text("Pressure Iterations"));

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

                ui.label("Max Tracer Color");
                let c = self.settings.max_tracer_color;
                let mut rgb = [c[0], c[1], c[2]];
                ui.color_edit_button_srgb(&mut rgb);
                self.settings.max_tracer_color = [rgb[0], rgb[1], rgb[2], 255];

                ui.add_space(8.0);
                if ui.button("Launch Simulation").clicked() {
                    save_settings(&self.settings);
                    launch = Some(self.settings.clone());
                }
            });
        });

        launch
    }
}

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
                ui.add(egui::Slider::new(&mut boundary.velocity_x, -200.0..=200.0).text("Velocity X (cells/s)"));
                ui.add(egui::Slider::new(&mut boundary.velocity_y, -200.0..=200.0).text("Velocity Y (cells/s)"));
                ui.weak(format!("= {:.3} m/s, {:.3} m/s", boundary.velocity_x * dx, boundary.velocity_y * dx));
                ui.add(egui::Slider::new(&mut boundary.tracer, 0.0..=1.0).text("Tracer"));
                ui.add(egui::Slider::new(&mut boundary.offset, 0..=500).text("Offset"));
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

fn save_settings(settings: &SimulationSettings) {
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