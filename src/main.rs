mod window;

fn main() {
    let mut window = window::FluidWindow::new(800, 600, 120, 10, 5);
    window.run();
}

