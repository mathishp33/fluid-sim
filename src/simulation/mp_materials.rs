//! Bibliothèque de matériaux de l'utilisateur (fluides, solides, miscibilité, changements de phase, réactions) :
//! sérialisable (config.json), éditable, et convertie en `Registry` (indices) pour le solveur.
//!
//! Tout est référencé PAR NOM dans la bibliothèque (donc stable quand on ajoute / supprime / réordonne) ; `build()` fait la
//! correspondance nom → indice. Les matériaux par défaut (air, eau, vapeur, huile / acier, verre) sont des entrées ordinaires :
//! l'utilisateur peut les modifier ou les supprimer, et `MaterialLibrary::default()` les restitue.
//!
//! Réactions chimiques (à venir) : `ReactionDef` est déjà stocké, validé (bilan de masse) et converti en `Reaction` dans le registre ;
//! seul le moteur cinétique (`MultiphaseSim::react_step`) reste à écrire.

use serde::{Deserialize, Serialize};

use super::phases::{Eos, Phase, PhaseKind, Reaction, Registry, Solid, Thermal, MAX_PHASES};

/// paire liquide ↔ vapeur (ébullition / condensation / cavitation) définie par noms
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChangeDef {
    pub liquid: String,
    pub vapor: String,
    /// chaleur latente à t_ref (J/kg)
    pub l_ref: f64,
    pub t_ref: f64,
    /// pression de saturation à t_ref (Pa)
    pub p_ref: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReactionDef {
    pub name: String,
    pub reactants: Vec<(String, f64)>,
    pub products: Vec<(String, f64)>,
    pub pre_exp: f64,
    pub activation_energy: f64,
    #[serde(default)]
    pub orders: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialLibrary {
    pub fluids: Vec<Phase>,
    pub solids: Vec<Solid>,
    /// paires de fluides miscibles (par nom)
    pub miscible: Vec<(String, String)>,
    pub changes: Vec<ChangeDef>,
    pub reactions: Vec<ReactionDef>,
    /// fluide qui entre par les bords ouverts / remplit les cellules vidées (par nom ; si absent : premier gaz)
    pub ambient: String,
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        MaterialLibrary {
            fluids: vec![Phase::air(), Phase::water(), Phase::steam(), Phase::oil()],
            solids: vec![Solid::steel(), Solid::glass()],
            miscible: vec![],
            changes: vec![ChangeDef { liquid: "eau".into(), vapor: "vapeur".into(), l_ref: 2.257e6, t_ref: 373.15, p_ref: 101325.0 }],
            reactions: vec![],
            ambient: "air".into(),
        }
    }
}

/// Résultat de `MaterialLibrary::build`
pub struct Built {
    pub registry: Registry,
    /// indice du fluide ambiant
    pub ambient: usize,
    /// messages non bloquants (références orphelines ignorées, groupes miscibles douteux, …)
    pub warnings: Vec<String>,
}

/// Rôles utilisés par les scénarios prédéfinis, retrouvés par nom puis, à défaut, par propriété.
#[derive(Clone, Copy, Debug)]
pub struct Roles {
    pub air: Option<usize>,
    pub water: Option<usize>,
    pub vapor: Option<usize>,
    pub oil: Option<usize>,
    pub plate: Option<usize>,
}

fn lower(s: &str) -> String {
    s.trim().to_lowercase()
}

impl Roles {
    pub fn resolve(reg: &Registry, ambient: usize) -> Roles {
        let by_name = |names: &[&str], kind: Option<PhaseKind>| -> Option<usize> {
            reg.phases.iter().position(|p| names.contains(&lower(&p.name).as_str()) && kind.map_or(true, |k| p.kind == k))
        };
        let is_vapor = |k: usize| reg.changes.iter().any(|c| c.vapor == k);
        let is_liq_with_change = |k: usize| reg.changes.iter().any(|c| c.liquid == k);
        let air = by_name(&["air"], Some(PhaseKind::Gas))
            .or_else(|| (reg.phases.get(ambient).map_or(false, |p| p.is_gas())).then_some(ambient))
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas() && !is_vapor(k)))
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas()));
        let water = by_name(&["eau", "water"], Some(PhaseKind::Liquid))
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].kind == PhaseKind::Liquid && is_liq_with_change(k)))
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].kind == PhaseKind::Liquid));
        let vapor = water
            .and_then(|w| reg.change_for_liquid(w).map(|c| c.vapor))
            .or_else(|| by_name(&["vapeur", "vapor", "vapour", "steam"], Some(PhaseKind::Gas)));
        let oil = by_name(&["huile", "oil"], Some(PhaseKind::Liquid)).or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].kind == PhaseKind::Liquid && Some(k) != water));
        let plate = reg.solids.iter().position(|s| ["acier", "steel"].contains(&lower(&s.name).as_str())).or(if reg.solids.is_empty() { None } else { Some(0) });
        Roles { air, water, vapor, oil, plate }
    }
}

impl MaterialLibrary {
    pub fn fluid_index(&self, name: &str) -> Option<usize> {
        self.fluids.iter().position(|p| p.name == name)
    }
    pub fn solid_index(&self, name: &str) -> Option<usize> {
        self.solids.iter().position(|s| s.name == name)
    }

    // ---------------------------------------------------------------- édition cohérente
    /// Nom libre de la forme « base », « base 2 », « base 3 », …
    pub fn unique_name(&self, base: &str) -> String {
        let taken = |n: &str| self.fluids.iter().any(|p| p.name == n) || self.solids.iter().any(|s| s.name == n);
        if !taken(base) {
            return base.to_string();
        }
        (2..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap()
    }

    /// Fluide neuf avec des propriétés génériques (à personnaliser par l'utilisateur).
    pub fn template_fluid(&self, kind: PhaseKind) -> Phase {
        match kind {
            PhaseKind::Gas => Phase {
                name: self.unique_name("gaz"),
                kind,
                eos: Eos::IdealGas { r_spec: 287.0 },
                cp: 1000.0,
                conductivity: 0.026,
                mu_ref: 1.8e-5,
                mu_law: super::materials::PropertyLaw::Sutherland { s: 110.4 },
                k_law: super::materials::PropertyLaw::Constant,
                t_ref: 293.15,
                e0: 0.0,
                sigma: 0.0,
                color: [0.4, 0.8, 0.4],
                molar_mass: 0.029,
                diffusivity: 2.0e-5,
            },
            PhaseKind::Liquid => Phase {
                name: self.unique_name("liquide"),
                kind,
                eos: Eos::Tait { p0: 101325.0, b: 3.0e8, n: 7.0, thermal: Thermal::Beta { rho0: 1000.0, t0: 293.15, beta: 3.0e-4 } },
                cp: 3000.0,
                conductivity: 0.3,
                mu_ref: 1.0e-3,
                mu_law: super::materials::PropertyLaw::Arrhenius { e_r: 2000.0 },
                k_law: super::materials::PropertyLaw::Constant,
                t_ref: 293.15,
                e0: 0.0,
                sigma: 0.04,
                color: [0.8, 0.3, 0.7],
                molar_mass: 0.05,
                diffusivity: 1.0e-9,
            },
        }
    }
    pub fn template_solid(&self) -> Solid {
        Solid { name: self.unique_name("solide"), rho: 2000.0, cp: 800.0, conductivity: 2.0, color: [0.7, 0.5, 0.4] }
    }

    pub fn rename_fluid(&mut self, i: usize, new: &str) {
        let old = self.fluids[i].name.clone();
        if old == new || new.trim().is_empty() {
            return;
        }
        let new = new.trim().to_string();
        if self.fluids.iter().enumerate().any(|(k, p)| k != i && p.name == new) {
            return; // nom déjà pris
        }
        self.fluids[i].name = new.clone();
        for (a, b) in self.miscible.iter_mut() {
            if *a == old { *a = new.clone() }
            if *b == old { *b = new.clone() }
        }
        for c in self.changes.iter_mut() {
            if c.liquid == old { c.liquid = new.clone() }
            if c.vapor == old { c.vapor = new.clone() }
        }
        for r in self.reactions.iter_mut() {
            for (n, _) in r.reactants.iter_mut().chain(r.products.iter_mut()) {
                if *n == old { *n = new.clone() }
            }
        }
        if self.ambient == old {
            self.ambient = new;
        }
    }
    pub fn rename_solid(&mut self, i: usize, new: &str) {
        let new = new.trim();
        if new.is_empty() || self.solids.iter().enumerate().any(|(k, s)| k != i && s.name == new) {
            return;
        }
        self.solids[i].name = new.to_string();
    }
    /// Supprime un fluide et tout ce qui le référence (miscibilité, changements de phase, réactions).
    pub fn remove_fluid(&mut self, i: usize) {
        let old = self.fluids.remove(i).name;
        self.miscible.retain(|(a, b)| *a != old && *b != old);
        self.changes.retain(|c| c.liquid != old && c.vapor != old);
        self.reactions.retain(|r| !r.reactants.iter().chain(r.products.iter()).any(|(n, _)| *n == old));
        if self.ambient == old {
            self.ambient = self.fluids.iter().find(|p| p.is_gas()).map(|p| p.name.clone()).unwrap_or_default();
        }
    }
    pub fn remove_solid(&mut self, i: usize) {
        self.solids.remove(i);
    }

    pub fn is_miscible(&self, a: &str, b: &str) -> bool {
        a != b && self.miscible.iter().any(|(x, y)| (x == a && y == b) || (x == b && y == a))
    }
    pub fn set_miscible(&mut self, a: &str, b: &str, on: bool) {
        if a == b {
            return;
        }
        self.miscible.retain(|(x, y)| !((x == a && y == b) || (x == b && y == a)));
        if on {
            self.miscible.push((a.to_string(), b.to_string()));
        }
    }

    // ---------------------------------------------------------------- conversion
    pub fn build(&self) -> Result<Built, String> {
        if self.fluids.is_empty() {
            return Err("aucun fluide défini : ajoutez au moins un gaz (fluide ambiant)".into());
        }
        if self.fluids.len() > MAX_PHASES {
            return Err(format!("{} fluides : maximum {MAX_PHASES}", self.fluids.len()));
        }
        let mut warnings = vec![];
        let mut reg = Registry::new();
        for p in &self.fluids {
            let mut p = p.clone();
            p.e0 = 0.0; // recalculé par add_phase_change
            reg.add_phase(p);
        }
        for s in &self.solids {
            reg.add_solid(s.clone());
        }
        let idx = |n: &str| self.fluids.iter().position(|p| p.name == n);
        for c in &self.changes {
            match (idx(&c.liquid), idx(&c.vapor)) {
                (Some(l), Some(v)) => {
                    if reg.phases[l].kind != PhaseKind::Liquid || !matches!(reg.phases[v].eos, Eos::IdealGas { .. }) {
                        warnings.push(format!("changement de phase « {} → {} » ignoré : il faut un liquide et un gaz parfait", c.liquid, c.vapor));
                    } else if reg.changes.iter().any(|q| q.liquid == l || q.vapor == v) {
                        warnings.push(format!("changement de phase « {} → {} » ignoré : le liquide ou la vapeur est déjà utilisé par une autre paire", c.liquid, c.vapor));
                    } else {
                        reg.add_phase_change(l, v, c.l_ref, c.t_ref, c.p_ref);
                    }
                }
                _ => warnings.push(format!("changement de phase « {} → {} » ignoré : fluide inexistant", c.liquid, c.vapor)),
            }
        }
        for (a, b) in &self.miscible {
            match (idx(a), idx(b)) {
                (Some(x), Some(y)) if x != y => reg.set_miscible(x, y, true),
                _ => warnings.push(format!("miscibilité « {a} ↔ {b} » ignorée : fluide inexistant")),
            }
        }
        for r in &self.reactions {
            let conv = |v: &Vec<(String, f64)>| -> Option<Vec<(usize, f64)>> { v.iter().map(|(n, nu)| idx(n).map(|k| (k, *nu))).collect() };
            match (conv(&r.reactants), conv(&r.products)) {
                (Some(re), Some(pr)) => reg.reactions.push(Reaction {
                    name: r.name.clone(),
                    reactants: re,
                    products: pr,
                    pre_exp: r.pre_exp,
                    activation_energy: r.activation_energy,
                    orders: r.orders.clone(),
                }),
                _ => warnings.push(format!("réaction « {} » ignorée : fluide inexistant", r.name)),
            }
        }
        warnings.extend(reg.validate());
        let ambient = idx(&self.ambient)
            .filter(|&k| reg.phases[k].is_gas())
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas() && !reg.changes.iter().any(|c| c.vapor == k)))
            .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas()))
            .ok_or_else(|| "aucun gaz défini : un gaz ambiant est nécessaire (il remplit les bords ouverts et les cellules vidées)".to_string())?;
        Ok(Built { registry: reg, ambient, warnings })
    }

    /// Les noms des matériaux par défaut absents de la bibliothèque (pour proposer de les restaurer un par un).
    pub fn missing_defaults(&self) -> (Vec<String>, Vec<String>) {
        let d = MaterialLibrary::default();
        (
            d.fluids.iter().filter(|p| self.fluid_index(&p.name).is_none()).map(|p| p.name.clone()).collect(),
            d.solids.iter().filter(|s| self.solid_index(&s.name).is_none()).map(|s| s.name.clone()).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_library_reproduces_the_historical_registry() {
        let b = MaterialLibrary::default().build().unwrap();
        let names: Vec<&str> = b.registry.phases.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["air", "eau", "vapeur", "huile"]);
        assert_eq!(b.registry.solids.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["acier", "verre"]);
        assert_eq!(b.ambient, 0);
        assert_eq!(b.registry.changes.len(), 1);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
        // identique au constructeur historique (e0 de la vapeur fixé par la chaleur latente)
        let (r, _, _, _) = Registry::water_air_steam();
        assert!((b.registry.phases[2].e0 - r.phases[2].e0).abs() < 1e-9);
    }

    #[test]
    fn serde_roundtrip_and_old_config_defaults() {
        let lib = MaterialLibrary::default();
        let json = serde_json::to_string(&lib).unwrap();
        let back: MaterialLibrary = serde_json::from_str(&json).unwrap();
        assert_eq!(lib, back);
        // un champ absent (config ancien) retombe sur les valeurs par défaut
        let partial: MaterialLibrary = serde_json::from_str("{}").unwrap();
        assert_eq!(partial, MaterialLibrary::default());
    }

    #[test]
    fn deleting_and_renaming_keeps_references_consistent() {
        let mut lib = MaterialLibrary::default();
        lib.set_miscible("eau", "huile", true);
        lib.rename_fluid(1, "H2O");
        assert!(lib.is_miscible("H2O", "huile") && lib.changes[0].liquid == "H2O");
        lib.remove_fluid(lib.fluid_index("vapeur").unwrap());
        assert!(lib.changes.is_empty());
        let b = lib.build().unwrap();
        assert_eq!(b.registry.phases.len(), 3);
        assert!(b.registry.is_miscible(1, 2));
        // suppression de tous les gaz : erreur claire
        let mut lib = MaterialLibrary::default();
        lib.remove_fluid(0);
        lib.remove_fluid(1); // vapeur (devenue indice 1)
        assert!(lib.build().is_err());
    }

    #[test]
    fn miscible_groups_are_transitive_and_validated() {
        let mut lib = MaterialLibrary::default();
        let mut alcool = lib.template_fluid(PhaseKind::Liquid);
        alcool.name = "alcool".into();
        lib.fluids.push(alcool);
        lib.set_miscible("eau", "alcool", true);
        lib.set_miscible("alcool", "huile", true);
        let b = lib.build().unwrap();
        let g: Vec<Vec<usize>> = b.registry.miscible_groups().into_iter().filter(|g| g.len() > 1).collect();
        assert_eq!(g, vec![vec![1, 3, 4]]);
        assert!(b.warnings.iter().any(|w| w.contains("transitivité")), "{:?}", b.warnings);
        // gaz + liquide miscibles : refusé par la validation
        lib.set_miscible("air", "eau", true);
        let b = lib.build().unwrap();
        assert!(b.warnings.iter().any(|w| w.contains("gaz et liquides")), "{:?}", b.warnings);
    }

    #[test]
    fn reaction_mass_balance_is_checked() {
        let mut lib = MaterialLibrary::default();
        // 2 H2O(l) -> 2 H2O(g) : équilibrée ; eau -> huile : non
        lib.reactions.push(ReactionDef { name: "ok".into(), reactants: vec![("eau".into(), 2.0)], products: vec![("vapeur".into(), 2.0)], pre_exp: 1.0, activation_energy: 1e4, orders: vec![] });
        lib.reactions.push(ReactionDef { name: "bad".into(), reactants: vec![("eau".into(), 1.0)], products: vec![("huile".into(), 1.0)], pre_exp: 1.0, activation_energy: 1e4, orders: vec![] });
        let b = lib.build().unwrap();
        assert!(b.warnings.iter().any(|w| w.contains("« bad »") && w.contains("masse non conservée")), "{:?}", b.warnings);
        assert!(!b.warnings.iter().any(|w| w.contains("« ok »")));
        // la chaleur de réaction eau -> vapeur retrouve la chaleur latente à 0 K de référence : L − Δcp·T_ref
        let r = &b.registry;
        let h = r.reactions[0].heat_of_reaction(&r.phases) / (2.0 * 0.018015);
        let pc = &r.changes[0];
        assert!((h - (pc.l_ref - pc.dcp * pc.t_ref)).abs() < 1.0, "h={h}");
    }

    #[test]
    fn roles_fall_back_to_properties_when_names_differ() {
        let mut lib = MaterialLibrary::default();
        lib.rename_fluid(0, "azote");
        lib.rename_fluid(1, "solvant");
        lib.rename_fluid(2, "gaz de solvant");
        let b = lib.build().unwrap();
        let r = Roles::resolve(&b.registry, b.ambient);
        assert_eq!((r.air, r.water, r.vapor, r.oil), (Some(0), Some(1), Some(2), Some(3)));
    }
}
