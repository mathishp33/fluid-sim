use crate::window::starting_window::StartingGui;
use crate::window::starting_window::load_settings;

mod window;
mod simulation;


fn main() -> eframe::Result<()> {
    println!("Rayon threads: {}", rayon::current_num_threads());

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Fluid Simulation Config",
        options,
        Box::new(|_cc| {
            Ok(Box::new(StartingGui {
                settings: load_settings(),
                dark_theme_set: false,
            }))
        }),
    )
}