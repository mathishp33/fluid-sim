//! Éditeur de la bibliothèque de matériaux (fluides, solides, miscibilité, changements de phase, réactions).
//! Utilisé par l'écran de démarrage et par la fenêtre multiphasique. Ne touche qu'à `MaterialLibrary` ; l'appelant décide quand
//! reconstruire la simulation (les indices de matériaux changent quand on ajoute / supprime).

use egui::{ComboBox, DragValue, Ui};

use crate::simulation::materials::PropertyLaw;
use crate::simulation::mp_materials::{ChangeDef, MaterialLibrary, ReactionDef};
use crate::simulation::phases::{Eos, Phase, PhaseKind, Thermal};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EditorTab {
    #[default]
    Fluids,
    Solids,
    Miscibility,
    PhaseChange,
    Reactions,
}

#[derive(Default)]
pub struct MaterialEditor {
    tab: EditorTab,
    sel_fluid: usize,
    sel_solid: usize,
    sel_reaction: usize,
    name_buf: String,
    name_for: Option<(EditorTab, usize)>,
    status: String,
}

impl MaterialEditor {
    /// Dessine l'éditeur. Retourne true si la bibliothèque a été modifiée pendant cette trame.
    pub fn ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for (t, n) in [
                (EditorTab::Fluids, "Fluides"),
                (EditorTab::Solids, "Solides"),
                (EditorTab::Miscibility, "Miscibilité"),
                (EditorTab::PhaseChange, "Changement de phase"),
                (EditorTab::Reactions, "Réactions"),
            ] {
                ui.selectable_value(&mut self.tab, t, n);
            }
        });
        ui.separator();
        match self.tab {
            EditorTab::Fluids => changed |= self.fluids_ui(ui, lib),
            EditorTab::Solids => changed |= self.solids_ui(ui, lib),
            EditorTab::Miscibility => changed |= self.miscibility_ui(ui, lib),
            EditorTab::PhaseChange => changed |= self.phase_change_ui(ui, lib),
            EditorTab::Reactions => changed |= self.reactions_ui(ui, lib),
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui.button("Restaurer tous les matériaux par défaut").on_hover_text("Remplace TOUTE la bibliothèque (vos matériaux sont perdus)").clicked() {
                *lib = MaterialLibrary::default();
                self.sel_fluid = 0;
                self.sel_solid = 0;
                self.name_for = None;
                changed = true;
            }
            let (mf, ms) = lib.missing_defaults();
            if !mf.is_empty() || !ms.is_empty() {
                ui.menu_button("Réajouter un défaut supprimé", |ui| {
                    let d = MaterialLibrary::default();
                    for n in &mf {
                        if ui.button(format!("fluide : {n}")).clicked() {
                            if let Some(p) = d.fluids.iter().find(|p| &p.name == n) {
                                lib.fluids.push(p.clone());
                                // restaure aussi ses liens par défaut vers des fluides présents
                                let back: Vec<ChangeDef> = d
                                    .changes
                                    .iter()
                                    .filter(|c| (&c.liquid == n || &c.vapor == n) && lib.fluid_index(&c.liquid).is_some() && lib.fluid_index(&c.vapor).is_some())
                                    .cloned()
                                    .collect();
                                for c in back {
                                    if !lib.changes.contains(&c) {
                                        lib.changes.push(c);
                                    }
                                }
                                changed = true;
                            }
                            ui.close_menu();
                        }
                    }
                    for n in &ms {
                        if ui.button(format!("solide : {n}")).clicked() {
                            if let Some(s) = d.solids.iter().find(|s| &s.name == n) {
                                lib.solids.push(s.clone());
                                changed = true;
                            }
                            ui.close_menu();
                        }
                    }
                });
            }
            if ui.button("Exporter…").clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("Matériaux (JSON)", &["json"]).set_file_name("materiaux.json").save_file() {
                    self.status = match serde_json::to_string_pretty(lib).map_err(|e| e.to_string()).and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string())) {
                        Ok(()) => format!("Exporté : {}", path.display()),
                        Err(e) => format!("Erreur d'export : {e}"),
                    };
                }
            }
            if ui.button("Importer…").clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("Matériaux (JSON)", &["json"]).pick_file() {
                    match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str::<MaterialLibrary>(&s).map_err(|e| e.to_string())) {
                        Ok(l) => {
                            *lib = l;
                            self.sel_fluid = 0;
                            self.sel_solid = 0;
                            self.name_for = None;
                            self.status = format!("Importé : {}", path.display());
                            changed = true;
                        }
                        Err(e) => self.status = format!("Erreur d'import : {e}"),
                    }
                }
            }
        });
        if !self.status.is_empty() {
            ui.weak(&self.status);
        }
        // validation en direct
        match lib.build() {
            Ok(b) => {
                for w in b.warnings {
                    ui.colored_label(egui::Color32::from_rgb(230, 170, 40), format!("⚠ {w}"));
                }
            }
            Err(e) => {
                ui.colored_label(egui::Color32::from_rgb(230, 80, 80), format!("✖ {e}"));
            }
        }
        changed
    }

    /// champ « nom » avec tampon : n'applique le renommage qu'à la validation (perte de focus / Entrée)
    fn name_field(&mut self, ui: &mut Ui, tab: EditorTab, idx: usize, current: &str) -> Option<String> {
        if self.name_for != Some((tab, idx)) {
            self.name_buf = current.to_string();
            self.name_for = Some((tab, idx));
        }
        let r = ui.text_edit_singleline(&mut self.name_buf);
        if r.lost_focus() && self.name_buf.trim() != current {
            let n = self.name_buf.trim().to_string();
            if n.is_empty() {
                self.name_buf = current.to_string();
                return None;
            }
            return Some(n);
        }
        None
    }

    // ------------------------------------------------------------------ fluides
    fn fluids_ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui.button("+ Gaz").clicked() {
                let p = lib.template_fluid(PhaseKind::Gas);
                lib.fluids.push(p);
                self.sel_fluid = lib.fluids.len() - 1;
                changed = true;
            }
            if ui.button("+ Liquide").clicked() {
                let p = lib.template_fluid(PhaseKind::Liquid);
                lib.fluids.push(p);
                self.sel_fluid = lib.fluids.len() - 1;
                changed = true;
            }
            if ui.add_enabled(!lib.fluids.is_empty(), egui::Button::new("Supprimer")).clicked() {
                let i = self.sel_fluid.min(lib.fluids.len().saturating_sub(1));
                lib.remove_fluid(i);
                self.sel_fluid = self.sel_fluid.saturating_sub(1);
                self.name_for = None;
                changed = true;
            }
            if ui.add_enabled(!lib.fluids.is_empty(), egui::Button::new("Dupliquer")).clicked() {
                let mut p = lib.fluids[self.sel_fluid.min(lib.fluids.len() - 1)].clone();
                p.name = lib.unique_name(&format!("{} copie", p.name));
                lib.fluids.push(p);
                self.sel_fluid = lib.fluids.len() - 1;
                changed = true;
            }
        });
        if lib.fluids.is_empty() {
            ui.weak("Aucun fluide : ajoutez au moins un gaz.");
            return changed;
        }
        self.sel_fluid = self.sel_fluid.min(lib.fluids.len() - 1);
        ui.horizontal_wrapped(|ui| {
            for (i, p) in lib.fluids.iter().enumerate() {
                let tag = if p.is_gas() { "gaz" } else { "liquide" };
                ui.selectable_value(&mut self.sel_fluid, i, format!("{} ({tag})", p.name));
            }
        });
        ui.separator();
        let i = self.sel_fluid;
        let cur = lib.fluids[i].name.clone();
        ui.horizontal(|ui| {
            ui.label("Nom");
            if let Some(n) = self.name_field(ui, EditorTab::Fluids, i, &cur) {
                lib.rename_fluid(i, &n);
                changed = true;
            }
        });
        let amb_name = lib.fluids[i].name.clone();
        if lib.fluids[i].is_gas() {
            let mut is_amb = lib.ambient == amb_name;
            if ui.checkbox(&mut is_amb, "Gaz ambiant (entre par les bords ouverts, remplit les cellules vidées)").changed() && is_amb {
                lib.ambient = amb_name;
                changed = true;
            }
        }
        let p = &mut lib.fluids[i];
        changed |= fluid_props_ui(ui, p);
        changed
    }

    // ------------------------------------------------------------------ solides
    fn solids_ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            if ui.button("+ Solide").clicked() {
                let s = lib.template_solid();
                lib.solids.push(s);
                self.sel_solid = lib.solids.len() - 1;
                changed = true;
            }
            if ui.add_enabled(!lib.solids.is_empty(), egui::Button::new("Supprimer")).clicked() {
                let i = self.sel_solid.min(lib.solids.len().saturating_sub(1));
                lib.remove_solid(i);
                self.sel_solid = self.sel_solid.saturating_sub(1);
                self.name_for = None;
                changed = true;
            }
            if ui.add_enabled(!lib.solids.is_empty(), egui::Button::new("Dupliquer")).clicked() {
                let mut s = lib.solids[self.sel_solid.min(lib.solids.len() - 1)].clone();
                s.name = lib.unique_name(&format!("{} copie", s.name));
                lib.solids.push(s);
                self.sel_solid = lib.solids.len() - 1;
                changed = true;
            }
        });
        if lib.solids.is_empty() {
            ui.weak("Aucun solide.");
            return changed;
        }
        self.sel_solid = self.sel_solid.min(lib.solids.len() - 1);
        ui.horizontal_wrapped(|ui| {
            for (i, s) in lib.solids.iter().enumerate() {
                ui.selectable_value(&mut self.sel_solid, i, &s.name);
            }
        });
        ui.separator();
        let i = self.sel_solid;
        let cur = lib.solids[i].name.clone();
        ui.horizontal(|ui| {
            ui.label("Nom");
            if let Some(n) = self.name_field(ui, EditorTab::Solids, i, &cur) {
                lib.rename_solid(i, &n);
                changed = true;
            }
        });
        let s = &mut lib.solids[i];
        egui::Grid::new("solid_props").num_columns(2).show(ui, |ui| {
            ui.label("ρ (kg/m³)");
            changed |= ui.add(DragValue::new(&mut s.rho).speed(10.0).clamp_range(1.0..=30000.0)).changed();
            ui.end_row();
            ui.label("cp (J/(kg·K))");
            changed |= ui.add(DragValue::new(&mut s.cp).speed(5.0).clamp_range(10.0..=20000.0)).changed();
            ui.end_row();
            ui.label("k (W/(m·K))");
            changed |= ui.add(DragValue::new(&mut s.conductivity).speed(0.1).clamp_range(1e-3..=1000.0)).changed();
            ui.end_row();
            ui.label("Couleur");
            changed |= ui.color_edit_button_rgb(&mut s.color).changed();
            ui.end_row();
        });
        changed
    }

    // ------------------------------------------------------------------ miscibilité
    fn miscibility_ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.label("Cochez les couples de fluides miscibles (ils se mélangent à l'échelle moléculaire : pas d'interface entre eux, diffusion de Fick). Seuls les couples gaz↔gaz et liquide↔liquide existent. La relation est transitive : si A~B et B~C, A, B et C forment un seul mélange.");
        for kind in [PhaseKind::Liquid, PhaseKind::Gas] {
            let names: Vec<String> = lib.fluids.iter().filter(|p| p.kind == kind).map(|p| p.name.clone()).collect();
            ui.add_space(4.0);
            ui.strong(if kind == PhaseKind::Liquid { "Liquides" } else { "Gaz" });
            if names.len() < 2 {
                ui.weak("moins de deux fluides de ce type : rien à régler");
                continue;
            }
            egui::Grid::new(format!("misc_{kind:?}")).striped(true).show(ui, |ui| {
                ui.label("");
                for n in names.iter().skip(1) {
                    ui.label(n);
                }
                ui.end_row();
                for (a, na) in names.iter().enumerate().take(names.len() - 1) {
                    ui.label(na);
                    for (b, nb) in names.iter().enumerate().skip(1) {
                        if b <= a {
                            ui.label("");
                            continue;
                        }
                        let mut on = lib.is_miscible(na, nb);
                        if ui.checkbox(&mut on, "").changed() {
                            lib.set_miscible(na, nb, on);
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
        }
        ui.add_space(4.0);
        ui.weak("La miscibilité n'est gérée que par le calcul CPU (le GPU est désactivé quand un couple est coché).");
        changed
    }

    // ------------------------------------------------------------------ changements de phase
    fn phase_change_ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.label("Paire liquide ↔ vapeur d'une même substance (ébullition, condensation, cavitation). La vapeur doit être un gaz parfait.");
        let liquids: Vec<String> = lib.fluids.iter().filter(|p| p.kind == PhaseKind::Liquid).map(|p| p.name.clone()).collect();
        let gases: Vec<String> = lib.fluids.iter().filter(|p| matches!(p.eos, Eos::IdealGas { .. })).map(|p| p.name.clone()).collect();
        if ui.add_enabled(!liquids.is_empty() && !gases.is_empty(), egui::Button::new("+ Paire liquide ↔ vapeur")).clicked() {
            lib.changes.push(ChangeDef { liquid: liquids[0].clone(), vapor: gases[0].clone(), l_ref: 2.257e6, t_ref: 373.15, p_ref: 101325.0 });
            changed = true;
        }
        let mut remove = None;
        for (ci, c) in lib.changes.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Liquide");
                    ComboBox::from_id_source(("pc_liq", ci)).selected_text(&c.liquid).show_ui(ui, |ui| {
                        for n in &liquids {
                            changed |= ui.selectable_value(&mut c.liquid, n.clone(), n).changed();
                        }
                    });
                    ui.label("Vapeur");
                    ComboBox::from_id_source(("pc_vap", ci)).selected_text(&c.vapor).show_ui(ui, |ui| {
                        for n in &gases {
                            changed |= ui.selectable_value(&mut c.vapor, n.clone(), n).changed();
                        }
                    });
                    if ui.button("Supprimer").clicked() {
                        remove = Some(ci);
                    }
                });
                egui::Grid::new(("pc_grid", ci)).num_columns(2).show(ui, |ui| {
                    ui.label("Chaleur latente L_ref (J/kg)");
                    changed |= ui.add(DragValue::new(&mut c.l_ref).speed(1e3).clamp_range(1e4..=1e8)).changed();
                    ui.end_row();
                    ui.label("T_ref (K)");
                    changed |= ui.add(DragValue::new(&mut c.t_ref).speed(0.5).clamp_range(100.0..=1500.0)).changed();
                    ui.end_row();
                    ui.label("p_sat(T_ref) (Pa)");
                    changed |= ui.add(DragValue::new(&mut c.p_ref).speed(100.0).clamp_range(1.0..=1e8)).changed();
                    ui.end_row();
                });
            });
        }
        if let Some(i) = remove {
            lib.changes.remove(i);
            changed = true;
        }
        changed
    }

    // ------------------------------------------------------------------ réactions
    fn reactions_ui(&mut self, ui: &mut Ui, lib: &mut MaterialLibrary) -> bool {
        let mut changed = false;
        ui.colored_label(egui::Color32::from_rgb(230, 170, 40), "Les réactions sont enregistrées et vérifiées (bilan de masse) mais le moteur cinétique n'est pas encore implémenté : elles n'ont AUCUN effet sur la simulation pour l'instant.");
        let names: Vec<String> = lib.fluids.iter().map(|p| p.name.clone()).collect();
        if ui.add_enabled(!names.is_empty(), egui::Button::new("+ Réaction")).clicked() {
            lib.reactions.push(ReactionDef {
                name: format!("réaction {}", lib.reactions.len() + 1),
                reactants: vec![(names[0].clone(), 1.0)],
                products: vec![(names[0].clone(), 1.0)],
                pre_exp: 1.0e6,
                activation_energy: 5.0e4,
                orders: vec![],
            });
            self.sel_reaction = lib.reactions.len() - 1;
            changed = true;
        }
        if lib.reactions.is_empty() {
            return changed;
        }
        self.sel_reaction = self.sel_reaction.min(lib.reactions.len() - 1);
        ui.horizontal_wrapped(|ui| {
            for (i, r) in lib.reactions.iter().enumerate() {
                ui.selectable_value(&mut self.sel_reaction, i, &r.name);
            }
        });
        let mut remove = false;
        let r = &mut lib.reactions[self.sel_reaction];
        ui.horizontal(|ui| {
            ui.label("Nom");
            changed |= ui.text_edit_singleline(&mut r.name).changed();
            if ui.button("Supprimer").clicked() {
                remove = true;
            }
        });
        for (title, list) in [("Réactifs", &mut r.reactants), ("Produits", &mut r.products)] {
            ui.strong(title);
            let mut del = None;
            for (k, (n, nu)) in list.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    changed |= ui.add(DragValue::new(nu).speed(0.1).clamp_range(0.001..=100.0)).changed();
                    ComboBox::from_id_source((title, k)).selected_text(n.as_str()).show_ui(ui, |ui| {
                        for m in &names {
                            changed |= ui.selectable_value(n, m.clone(), m).changed();
                        }
                    });
                    if ui.small_button("✖").clicked() {
                        del = Some(k);
                    }
                });
            }
            if let Some(k) = del {
                list.remove(k);
                changed = true;
            }
            if ui.small_button(format!("+ {title}")).clicked() && !names.is_empty() {
                list.push((names[0].clone(), 1.0));
                changed = true;
            }
        }
        egui::Grid::new("react_grid").num_columns(2).show(ui, |ui| {
            ui.label("Facteur pré-exponentiel A");
            changed |= ui.add(DragValue::new(&mut r.pre_exp).speed(1e3).clamp_range(0.0..=1e20)).changed();
            ui.end_row();
            ui.label("Énergie d'activation Ea (J/mol)");
            changed |= ui.add(DragValue::new(&mut r.activation_energy).speed(500.0).clamp_range(0.0..=1e7)).changed();
            ui.end_row();
        });
        if remove {
            lib.reactions.remove(self.sel_reaction);
            changed = true;
        }
        changed
    }
}

/// Propriétés d'un fluide. Retourne true si modifié.
fn fluid_props_ui(ui: &mut Ui, p: &mut Phase) -> bool {
    let mut ch = false;
    egui::Grid::new("fluid_props").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        ui.label("Couleur");
        ch |= ui.color_edit_button_rgb(&mut p.color).changed();
        ui.end_row();
        match &mut p.eos {
            Eos::IdealGas { r_spec } => {
                ui.label("R spécifique (J/(kg·K))");
                ch |= ui.add(DragValue::new(r_spec).speed(1.0).clamp_range(10.0..=5000.0)).changed();
                ui.end_row();
            }
            Eos::Tait { p0, b, n, thermal } => {
                ui.label("Tait p0 (Pa)");
                ch |= ui.add(DragValue::new(p0).speed(1000.0).clamp_range(1e3..=1e7)).changed();
                ui.end_row();
                ui.label("Tait B (Pa)");
                ch |= ui.add(DragValue::new(b).speed(1e6).clamp_range(1e6..=1e10)).changed();
                ui.end_row();
                ui.label("Tait n");
                ch |= ui.add(DragValue::new(n).speed(0.05).clamp_range(1.5..=20.0)).changed();
                ui.end_row();
                ui.label("Dilatation thermique");
                let is_kell = matches!(thermal, Thermal::Kell);
                ui.horizontal(|ui| {
                    if ui.selectable_label(!is_kell, "β constant").clicked() && is_kell {
                        *thermal = Thermal::Beta { rho0: 1000.0, t0: 293.15, beta: 2.0e-4 };
                        ch = true;
                    }
                    if ui.selectable_label(is_kell, "Kell (eau pure)").clicked() && !is_kell {
                        *thermal = Thermal::Kell;
                        ch = true;
                    }
                });
                ui.end_row();
                if let Thermal::Beta { rho0, t0, beta } = thermal {
                    ui.label("ρ0 à T0 (kg/m³)");
                    ch |= ui.add(DragValue::new(rho0).speed(1.0).clamp_range(10.0..=20000.0)).changed();
                    ui.end_row();
                    ui.label("T0 (K)");
                    ch |= ui.add(DragValue::new(t0).speed(0.5).clamp_range(100.0..=1000.0)).changed();
                    ui.end_row();
                    ui.label("β (1/K)");
                    ch |= ui.add(DragValue::new(beta).speed(1e-5).clamp_range(0.0..=5e-3)).changed();
                    ui.end_row();
                }
                ui.label("Tension superficielle σ (N/m)");
                ch |= ui.add(DragValue::new(&mut p.sigma).speed(0.001).clamp_range(0.0..=1.0)).changed();
                ui.end_row();
            }
        }
        ui.label("cp (J/(kg·K))");
        ch |= ui.add(DragValue::new(&mut p.cp).speed(5.0).clamp_range(100.0..=20000.0)).changed();
        ui.end_row();
        ui.label("k à T_ref (W/(m·K))");
        ch |= ui.add(DragValue::new(&mut p.conductivity).speed(0.001).clamp_range(1e-3..=100.0)).changed();
        ui.end_row();
        ui.label("μ à T_ref (Pa·s)");
        ch |= ui.add(DragValue::new(&mut p.mu_ref).speed(1e-5).clamp_range(1e-7..=100.0)).changed();
        ui.end_row();
        ui.label("T_ref (K)");
        ch |= ui.add(DragValue::new(&mut p.t_ref).speed(0.5).clamp_range(100.0..=1500.0)).changed();
        ui.end_row();
        ui.label("Masse molaire (g/mol)");
        let mut mm = p.molar_mass * 1000.0;
        if ui.add(DragValue::new(&mut mm).speed(0.1).clamp_range(1.0..=2000.0)).changed() {
            p.molar_mass = mm / 1000.0;
            ch = true;
        }
        ui.end_row();
        ui.label("Diffusion D (m²/s) dans un mélange miscible");
        ch |= ui.add(DragValue::new(&mut p.diffusivity).speed(1e-7).clamp_range(0.0..=1e-2)).changed();
        ui.end_row();
    });
    ch |= law_ui(ui, "Loi μ(T)", "mu_law", &mut p.mu_law);
    ch |= law_ui(ui, "Loi k(T)", "k_law", &mut p.k_law);
    ui.weak("L'énergie interne de référence (e0) d'une paire liquide↔vapeur est calculée automatiquement à partir de la chaleur latente.");
    ch
}

fn law_ui(ui: &mut Ui, label: &str, id: &str, law: &mut PropertyLaw) -> bool {
    let mut ch = false;
    let name = |l: &PropertyLaw| match l {
        PropertyLaw::Constant => "constante",
        PropertyLaw::Sutherland { .. } => "Sutherland (gaz)",
        PropertyLaw::Vogel { .. } => "Vogel (liquide)",
        PropertyLaw::Arrhenius { .. } => "Arrhenius",
        PropertyLaw::Linear { .. } => "linéaire",
    };
    ui.horizontal(|ui| {
        ui.label(label);
        ComboBox::from_id_source(id).selected_text(name(law)).show_ui(ui, |ui| {
            for cand in [
                PropertyLaw::Constant,
                PropertyLaw::Sutherland { s: 110.4 },
                PropertyLaw::Vogel { b: 247.8, c: 140.0 },
                PropertyLaw::Arrhenius { e_r: 3000.0 },
                PropertyLaw::Linear { slope: 0.003 },
            ] {
                let same = std::mem::discriminant(law) == std::mem::discriminant(&cand);
                if ui.selectable_label(same, name(&cand)).clicked() && !same {
                    *law = cand;
                    ch = true;
                }
            }
        });
        match law {
            PropertyLaw::Constant => {}
            PropertyLaw::Sutherland { s } => {
                ch |= ui.add(DragValue::new(s).speed(1.0).clamp_range(1.0..=2000.0).prefix("S = ")).changed();
            }
            PropertyLaw::Vogel { b, c } => {
                ch |= ui.add(DragValue::new(b).speed(1.0).clamp_range(1.0..=5000.0).prefix("b = ")).changed();
                ch |= ui.add(DragValue::new(c).speed(1.0).clamp_range(1.0..=400.0).prefix("c = ")).changed();
            }
            PropertyLaw::Arrhenius { e_r } => {
                ch |= ui.add(DragValue::new(e_r).speed(10.0).clamp_range(0.0..=20000.0).prefix("E/R = ")).changed();
            }
            PropertyLaw::Linear { slope } => {
                ch |= ui.add(DragValue::new(slope).speed(1e-4).clamp_range(-0.05..=0.05).prefix("pente = ")).changed();
            }
        }
    });
    ch
}
