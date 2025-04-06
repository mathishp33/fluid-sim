use eframe::egui::{self, Color32, ComboBox};

mod window;

fn launch_simulation(width: usize, height: usize, fps: i32, particle_radius: usize, precision: usize, start_density: f64, diffusion_rate: f64, friction_rate: f64) {
    let mut window = window::FluidWindow::new(width, height, fps, particle_radius, precision, start_density, diffusion_rate, friction_rate);
    window.run();
}

struct SimulationSettings {
    width: usize,
    height: usize,
    max_fps: i32,
    particle_radius: usize,
    precision: usize,
    start_density: f64,
    create_density_on_advection: bool,
    max_density_color: Color32,
    inverse_density_color: bool,
    diffusion_rate: f64,
    friction_rate: f64,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            max_fps: 60,
            particle_radius: 10,
            precision: 5,
            start_density: 0.2,
            create_density_on_advection: false,
            max_density_color: Color32::WHITE,
            inverse_density_color: false,
            diffusion_rate: 0.1,
            friction_rate: 0.5,
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

            ui.add(egui::Slider::new(&mut self.settings.diffusion_rate, 0.0..=0.20).text("Diffusion Rate"));
            ui.add(egui::Slider::new(&mut self.settings.friction_rate, 0.0..=1.0).text("Restitution"));

            ComboBox::from_label("Max FPS")
                .selected_text(format!("{}", self.settings.max_fps))
                .show_ui(ui, |ui| {
                    for fps in [30, 60, 120, 144, 180, 240] {
                        ui.selectable_value(&mut self.settings.max_fps, fps, format!("{fps}"));
                    }
                });

            ui.add(egui::Slider::new(&mut self.settings.particle_radius, 1..=50).text("Mouse Radius (pixels)"));

            ComboBox::from_label("Precision (pixels)")
                .selected_text(format!("{}", self.settings.precision))
                .show_ui(ui, |ui| {
                    for level in [1, 2, 5, 10] {
                        ui.selectable_value(&mut self.settings.precision, level, format!("{level}"));
                    }
                });

            ui.add(egui::Slider::new(&mut self.settings.start_density, 0.0..=1.0).text("Default Density"));

            ui.checkbox(&mut self.settings.create_density_on_advection, "Create Density on Advection");

            ui.label("Max Density Color");
            ui.color_edit_button_srgba(&mut self.settings.max_density_color);

            ui.checkbox(&mut self.settings.inverse_density_color, "Inverse Color Density");

            if ui.button("Launch Simulation").clicked() {
                launch_simulation(self.settings.width, self.settings.height, self.settings.max_fps, self.settings.particle_radius, self.settings.precision, self.settings.start_density, self.settings.diffusion_rate, self.settings.friction_rate);
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Fluid Simulation Config",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}