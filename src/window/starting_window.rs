use egui::ComboBox;
use serde::{Deserialize, Serialize};

use crate::simulation::fluid::{Boundary, BoundaryType};

#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationSettings {
    pub width: usize,
    pub height: usize,
    pub precision: usize,
    pub start_density: f32,
    pub max_density_color: [u8; 4],
    pub diffusion_rate: f32,
    pub pressure_iters: usize,
    pub diffusion_iters: usize,
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
            precision: 10,
            start_density: 0.2,
            max_density_color: [255, 255, 255, 255],
            diffusion_rate: 0.1,
            pressure_iters: 3,
            diffusion_iters: 3,
            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),
        }
    }
}

impl SimulationSettings {
    pub fn max_color(&self) -> u32 {
        let c = self.max_density_color;
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

                ui.add(egui::Slider::new(&mut self.settings.diffusion_rate, 0.0..=5.0).text("Diffusion Rate"));

                ui.add(egui::Slider::new(&mut self.settings.pressure_iters, 0..=200).text("Pressure Iterations"));
                ui.add(egui::Slider::new(&mut self.settings.diffusion_iters, 0..=50).text("Diffusion Iterations"));

                ComboBox::from_label("Precision (pixels)")
                    .selected_text(format!("{}", self.settings.precision))
                    .show_ui(ui, |ui| {
                        for level in [1, 2, 5, 10, 20] {
                            ui.selectable_value(&mut self.settings.precision, level, format!("{level}"));
                        }
                    });

                ui.add(egui::Slider::new(&mut self.settings.start_density, 0.0..=1.0).text("Default Density"));

                boundary_ui(ui, "Left Boundary", &mut self.settings.left_boundary);
                boundary_ui(ui, "Right Boundary", &mut self.settings.right_boundary);
                boundary_ui(ui, "Top Boundary", &mut self.settings.top_boundary);
                boundary_ui(ui, "Bottom Boundary", &mut self.settings.bottom_boundary);

                ui.label("Max Density Color");
                let c = self.settings.max_density_color;
                let mut rgb = [c[0], c[1], c[2]];
                ui.color_edit_button_srgb(&mut rgb);
                self.settings.max_density_color = [rgb[0], rgb[1], rgb[2], 255];

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

pub fn boundary_ui(ui: &mut egui::Ui, name: &str, boundary: &mut Boundary) {
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
                ui.add(egui::Slider::new(&mut boundary.velocity_x, -200.0..=200.0).text("Velocity X"));
                ui.add(egui::Slider::new(&mut boundary.velocity_y, -200.0..=200.0).text("Velocity Y"));
                ui.add(egui::Slider::new(&mut boundary.density, 0.0..=1.0).text("Density"));
                ui.add(egui::Slider::new(&mut boundary.offset, 0..=500).text("Offset"));
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