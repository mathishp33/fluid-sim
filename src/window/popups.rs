struct MyApp {
    show_popup: bool,
    filename: String,
    popup_result: Option<String>,
}

impl MyApp {
    fn show_filename_popup(&mut self, ctx: &egui::Context) {
        if !self.show_popup {
            return;
        }

        let mut close_popup = false;

        egui::Window::new("Nom du fichier")
            .collapsible(false)
            .resizable(false)
            .open(&mut self.show_popup)
            .show(ctx, |ui| {
                ui.text_edit_singleline(&mut self.filename);

                if ui.button("Valider").clicked() {
                    self.popup_result = Some(self.filename.clone());
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
}