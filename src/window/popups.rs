

#[derive(Clone)]
pub struct SpecialCellPopupResult {
    rate: f32,
    vx: f32,
    vy: f32,
    ax: f32,
    ay: f32,
    density: f32,
    dest_idx: usize,
}

impl SpecialCellPopupResult {
    pub fn new() -> Self {
        SpecialCellPopupResult {
            rate: 50.0,
            vx: 0.0,
            vy: 0.0,
            ax: 10.0,
            ay: 10.0,
            density: 0.5,
            dest_idx: 0,
        }
    }
}
pub struct SpecialCellPopup {
    pub show_popup: bool,
    pub popup_result: Option<SpecialCellPopupResult>,

    data: SpecialCellPopupResult,
}

impl SpecialCellPopup {
    pub fn new() -> Self {
        Self {
            show_popup: false,
            popup_result: None,
            data: SpecialCellPopupResult::new(),
        }
    }

    pub fn open(&mut self) {
        self.show_popup = true;
        self.popup_result = None;
    }

    pub fn show_special_cell_popup(&mut self, ctx: &egui::Context) {
        if !self.show_popup {
            return;
        }

        let mut close_popup = false;

        egui::Window::new("Special Cell")
            .collapsible(false)
            .resizable(false)
            .open(&mut self.show_popup)
            .show(ctx, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.data.ax, -200.0..=200.0)
                        .text("X acceleration")
                );

                ui.add(
                    egui::Slider::new(&mut self.data.ay, -200.0..=200.0)
                        .text("Y acceleration")
                );

                if ui.button("Valider").clicked() {
                    self.popup_result = Some(self.data.clone());
                    close_popup = true;
                }

                if ui.button("Annuler").clicked() {
                    close_popup = true;
                }
            });

        if close_popup {
            self.show_popup = false;
        }
    }

    pub fn take_result(&mut self) -> Option<SpecialCellPopupResult> {
        self.popup_result.take()
    }
}