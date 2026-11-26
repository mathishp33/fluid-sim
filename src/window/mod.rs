pub mod starting_window;
pub mod app;
pub mod multiphase_window;
pub mod experiment_window;
pub mod material_editor;

/// Ce qu'une fenêtre demande à l'application à la fin d'une image
pub enum WindowAction {
    None,
    BackToSettings,
}
