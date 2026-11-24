use eframe::egui_wgpu;

mod simulation;
mod window;

fn main() -> eframe::Result<()> {
    // `fluid-sim experience protocole.exp [--gpu] [--sortie dossier]` : expérience sans fenêtre (voir EXPERIENCES.md)
    let args: Vec<String> = std::env::args().skip(1).collect();
    if matches!(args.first().map(String::as_str), Some("experience" | "expérience" | "exp")) {
        std::process::exit(simulation::mp_experiment::run_cli(&args[1..]));
    }
    println!("Rayon threads: {}", rayon::current_num_threads());

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 500.0]),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            present_mode: egui_wgpu::wgpu::PresentMode::AutoNoVsync,
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "SFSDM : Simulation Fluide/Solide Dynamique de Mathis",
        options,
        Box::new(|_cc| Ok(Box::new(window::app::App::new()))),
    )
}