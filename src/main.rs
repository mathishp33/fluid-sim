use eframe::egui::{self, Color32, ComboBox};
use serde::{Deserialize, Serialize};
use simulation::fluid_sim::Boundary;

mod window;
mod simulation;

fn launch_simulation(width: usize, height: usize,
    particle_radius: usize, precision: usize,
    start_density: f32, diffusion_rate: f32,
    max_color: u32, randomize: bool, random_smoothing: usize,
    pressure_iters: usize, diffusion_iters: usize,
    left_boundary: Boundary, right_boundary: Boundary,
    top_boundary: Boundary, bottom_boundary: Boundary) {

    let mut window = window::FluidWindow::new(width, height, particle_radius, precision, start_density, diffusion_rate,
         max_color, randomize, random_smoothing, pressure_iters, diffusion_iters, left_boundary, right_boundary, top_boundary, bottom_boundary);
    window.run();
}

#[derive(Serialize, Deserialize, Clone)]
struct SimulationSettings {
    width: usize,
    height: usize,
    particle_radius: usize,
    precision: usize,
    start_density: f32,
    max_density_color: [u8; 4],
    diffusion_rate: f32,
    randomize: bool,
    random_smoothing: usize,
    pressure_iters: usize,
    diffusion_iters: usize,
    left_boundary: Boundary,
    right_boundary: Boundary,
    top_boundary: Boundary,
    bottom_boundary: Boundary,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            particle_radius: 10,
            precision: 10,
            start_density: 0.2,
            max_density_color: [255, 255, 255, 255],
            diffusion_rate: 0.1,
            randomize: false,
            random_smoothing: 100,
            pressure_iters: 3,
            diffusion_iters: 3,
            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),
        }
    }
}

struct MyApp {
    settings: SimulationSettings,
    dark_theme_set: bool,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            settings: SimulationSettings::default(),
            dark_theme_set: false,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        
        if !self.dark_theme_set {
            ctx.set_visuals(egui::Visuals::dark());
            self.dark_theme_set = true;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Fluid Simulation Settings");

            ui.add(egui::Slider::new(&mut self.settings.width, 100..=1920).text("Width"));
            ui.add(egui::Slider::new(&mut self.settings.height, 100..=1080).text("Height"));

            ui.add(egui::Slider::new(&mut self.settings.diffusion_rate, 0.0..=5.0).text("Diffusion Rate"));

            ui.add(egui::Slider::new(&mut self.settings.pressure_iters, 0..=30).text("Pressure Iterations"));
            ui.add(egui::Slider::new(&mut self.settings.diffusion_iters, 0..=30).text("Diffusion Iterations"));

            ui.add(egui::Slider::new(&mut self.settings.particle_radius, 1..=100).text("Mouse Radius (pixels)"));

            ComboBox::from_label("Precision (pixels)")
                .selected_text(format!("{}", self.settings.precision))
                .show_ui(ui, |ui| {
                    for level in [1, 2, 5, 10, 20] {
                        ui.selectable_value(&mut self.settings.precision, level, format!("{level}"));
                    }
                });

            ui.add(egui::Slider::new(&mut self.settings.start_density, 0.0..=1.0).text("Default Density"));
            ui.checkbox(&mut self.settings.randomize, "Randomize Initial Density (it overrides Default Density)");
            ui.add(egui::Slider::new(&mut self.settings.random_smoothing, 1..=10000).text("Random Smoothing"));

            boundary_ui(ui, "Left Boundary", &mut self.settings.left_boundary);
            boundary_ui(ui, "Right Boundary", &mut self.settings.right_boundary);
            boundary_ui(ui, "Top Boundary", &mut self.settings.top_boundary);
            boundary_ui(ui, "Bottom Boundary", &mut self.settings.bottom_boundary);

            //ui.label("Max Density Color");
            //ui.color_edit_button_srgba(&mut self.settings.max_density_color);

            if ui.button("Launch Simulation").clicked() {
                save_settings(&self.settings);

                let color = Color32::from_rgba_unmultiplied(
                    self.settings.max_density_color[0],
                    self.settings.max_density_color[1],
                    self.settings.max_density_color[2],
                    self.settings.max_density_color[3],
                );
                let max_color = ((color.r() as u32) << 16) | ((color.g() as u32) << 8) | ((color.b() as u32) << 0);
                launch_simulation(
                    self.settings.width, self.settings.height,
                    self.settings.particle_radius, self.settings.precision,
                    self.settings.start_density, self.settings.diffusion_rate,
                    max_color, self.settings.randomize, self.settings.random_smoothing,
                    self.settings.pressure_iters, self.settings.diffusion_iters,
                    self.settings.left_boundary, self.settings.right_boundary,
                    self.settings.top_boundary, self.settings.bottom_boundary,
                );
            }
        });
    }
}
fn boundary_ui(ui: &mut egui::Ui, name: &str, boundary: &mut Boundary) {
    ui.collapsing(name, |ui| {
        ComboBox::from_label("Type")
            .selected_text(format!("{:?}", boundary.boundary_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut boundary.boundary_type, simulation::fluid_sim::BoundaryType::Wall, "Wall");
                ui.selectable_value(&mut boundary.boundary_type, simulation::fluid_sim::BoundaryType::Inlet, "Inlet");
                ui.selectable_value(&mut boundary.boundary_type, simulation::fluid_sim::BoundaryType::Outlet, "Outlet");
            });

        match boundary.boundary_type {
            simulation::fluid_sim::BoundaryType::Inlet => {
                ui.add(egui::Slider::new(&mut boundary.velocity_x, -20.0..=20.0).text("Velocity X"));
                ui.add(egui::Slider::new(&mut boundary.velocity_y, -20.0..=20.0).text("Velocity Y"));
                ui.add(egui::Slider::new(&mut boundary.density, 0.0..=1.0).text("Density"));
            }

            simulation::fluid_sim::BoundaryType::Wall => {}

            simulation::fluid_sim::BoundaryType::Outlet => {}
        }
    });
}

fn load_settings() -> SimulationSettings {
    let path = "config.json";

    match std::fs::read_to_string(path) {
        Ok(data) => {
            serde_json::from_str(&data).unwrap_or_else(|e| {
                eprintln!("Unable to read config.json : {e}");
                SimulationSettings::default()
            })
        }

        Err(_) => {
            println!("nNo config found, using default values ...");
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

fn main() -> eframe::Result<()> {
    println!("Rayon threads: {}", rayon::current_num_threads());

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Fluid Simulation Config",
        options,
        Box::new(|_cc| {
            Ok(Box::new(MyApp {
                settings: load_settings(),
                dark_theme_set: false,
            }))
        }),
    )
}