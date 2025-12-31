use eframe::egui::{self, Color32, ComboBox};

mod window;

fn launch_simulation(width: usize, height: usize, particle_radius: usize, precision: usize, start_density: f64, diffusion_rate: f64, max_color: u32) {
    let mut window = window::FluidWindow::new(width, height, particle_radius, precision, start_density, diffusion_rate, max_color);
    window.run();
}

struct SimulationSettings {
    width: usize,
    height: usize,
    particle_radius: usize,
    precision: usize,
    start_density: f64,
    max_density_color: Color32,
    diffusion_rate: f64,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            particle_radius: 10,
            precision: 5,
            start_density: 0.2,
            max_density_color: Color32::WHITE,
            diffusion_rate: 0.1,
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

            ui.add(egui::Slider::new(&mut self.settings.particle_radius, 1..=50).text("Mouse Radius (pixels)"));

            ComboBox::from_label("Precision (pixels)")
                .selected_text(format!("{}", self.settings.precision))
                .show_ui(ui, |ui| {
                    for level in [1, 2, 5, 10] {
                        ui.selectable_value(&mut self.settings.precision, level, format!("{level}"));
                    }
                });

            ui.add(egui::Slider::new(&mut self.settings.start_density, 0.0..=1.0).text("Default Density"));

            ui.label("Max Density Color");
            ui.color_edit_button_srgba(&mut self.settings.max_density_color);

            if ui.button("Launch Simulation").clicked() {
                let color = self.settings.max_density_color;
                let max_color = ((color.r() as u32) << 16) | ((color.g() as u32) << 8) | ((color.b() as u32) << 0);
                launch_simulation(self.settings.width, self.settings.height, self.settings.particle_radius, self.settings.precision, self.settings.start_density, self.settings.diffusion_rate, max_color);
            
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