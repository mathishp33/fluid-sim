use crate::window::main_window::{FluidWindow, WindowAction};
use crate::window::starting_window::{load_settings, StartingGui};

/// Application racine : une seule fenêtre eframe (wgpu) qui affiche
/// soit l'écran de configuration, soit la simulation.
pub struct App {
    starting: StartingGui,
    simulation: Option<FluidWindow>,
    theme_set: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            starting: StartingGui::new(load_settings()),
            simulation: None,
            theme_set: false,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_set {
            ctx.set_visuals(egui::Visuals::dark());
            self.theme_set = true;
        }

        if let Some(sim) = &mut self.simulation {
            if matches!(sim.ui(ctx), WindowAction::BackToSettings) {
                self.simulation = None;
                ctx.request_repaint();
            }
        } else if let Some(settings) = self.starting.ui(ctx) {
            self.simulation = Some(FluidWindow::new(&settings));
            ctx.request_repaint();
        }
    }
}