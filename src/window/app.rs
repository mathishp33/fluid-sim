use crate::window::experiment_window::ExperimentWindow;
use crate::window::WindowAction;
use crate::window::multiphase_window::MultiphaseWindow;
use crate::window::starting_window::{load_settings, Launch, StartingGui};


pub struct App {
    starting: StartingGui,
    simulation: Option<MultiphaseWindow>,
    experiment: Option<ExperimentWindow>,
    theme_set: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            starting: StartingGui::new(load_settings()),
            simulation: None,
            experiment: None,
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
        } else if let Some(exp) = &mut self.experiment {
            if matches!(exp.ui(ctx), WindowAction::BackToSettings) {
                self.experiment = None;
                ctx.request_repaint();
            }
        } else if let Some(launch) = self.starting.ui(ctx) {
            match launch {
                Launch::Simulation(settings) => self.simulation = Some(MultiphaseWindow::new(&settings)),
                Launch::Experiment(settings) => self.experiment = Some(ExperimentWindow::new(&settings)),
            }
            ctx.request_repaint();
        }
    }
}
