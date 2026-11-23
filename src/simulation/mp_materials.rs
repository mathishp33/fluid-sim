//! Bibliothèque de matériaux de l'utilisateur (fluides, solides, miscibilité, changements de phase, réactions) :
//! sérialisable (config.json), éditable, et convertie en `Registry` (indices) pour le solveur.
//!
//! Tout est référencé PAR NOM dans la bibliothèque (donc stable quand on ajoute / supprime / réordonne) ; `build()` fait la
//! correspondance nom → indice. Les matériaux par défaut (air, eau, vapeur, huile / acier, verre) sont des entrées ordinaires :
//! l'utilisateur peut les modifier ou les supprimer, et `MaterialLibrary::default()` les restitue.
//!
//! Réactions chimiques : `ReactionDef` (par noms) → `Reaction` (indices), cinétique dans `mp_species`. Les énergies chimiques viennent des
//! enthalpies de formation des fluides (`Phase::h_formation`) ; les mélanges nommés (`MixtureDef`, ex. air = O2 + N2) servent de fluide
//! ambiant et de pinceaux. `combustion_preset()` ajoute O2, N2, CH4, H2, CO, CO2, les mélanges stœchiométriques et les mécanismes globaux
//! (Westbrook–Dryer pour CH4 et CO, Marinov pour H2) avec dissociation à l'équilibre.

use serde::{Deserialize, Serialize};

use super::materials::PropertyLaw;
use super::phases::{Eos, Phase, PhaseKind, Radiator, Reaction, Registry, Solid, Thermal, MAX_PHASES, R_GAS};

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

/// paire liquide ↔ solide granulaire (gel / fusion) définie par noms : température de fusion (K), chaleur latente (J/kg) ; surfusion
/// (voir `phases::Fusion`)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FusionDef {
    pub liquid: String,
    pub solid: String,
    pub t_melt: f64,
    pub l_fusion: f64,
    /// surfusion maximale avant la germination spontanée (K) ; 0 : gèle dès T_f
    #[serde(default)]
    pub supercooling: f64,
    /// vitesse du front de dendrites à 1 K de surfusion (m/s) et son exposant : v = a (ΔT / 1 K)^b
    #[serde(default = "default_growth_a")]
    pub growth_a: f64,
    #[serde(default = "default_growth_b")]
    pub growth_b: f64,
}

fn default_growth_a() -> f64 {
    super::phases::ICE_GROWTH_A
}
fn default_growth_b() -> f64 {
    super::phases::ICE_GROWTH_B
}

impl FusionDef {
    /// paire sans surfusion (front de dendrites de la glace par défaut)
    pub fn new(liquid: &str, solid: &str, t_melt: f64, l_fusion: f64) -> Self {
        FusionDef {
            liquid: liquid.into(),
            solid: solid.into(),
            t_melt,
            l_fusion,
            supercooling: 0.0,
            growth_a: default_growth_a(),
            growth_b: default_growth_b(),
        }
    }
}

/// Réaction par noms de fluides (voir `phases::Reaction` pour la loi de vitesse et les unités)
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReactionDef {
    pub name: String,
    pub reactants: Vec<(String, f64)>,
    pub products: Vec<(String, f64)>,
    /// A (SI : (m³/mol)^(ordre global − 1) / s / K^b)
    pub pre_exp: f64,
    /// Ea (J/mol)
    pub activation_energy: f64,
    #[serde(default)]
    pub orders: Vec<f64>,
    #[serde(default)]
    pub temp_exponent: f64,
    #[serde(default)]
    pub reversible: bool,
    /// espèces modulant la vitesse sans être consommées (nom, ordre)
    #[serde(default)]
    pub modifiers: Vec<(String, f64)>,
    /// épaisseur de flamme laminaire (m) pour le modèle de flamme épaissie ; 0 = aucun
    #[serde(default)]
    pub flame_thickness: f64,
    /// réaction en volume d'une phase granulaire (pyrolyse : voir `phases::Reaction::bulk`)
    #[serde(default)]
    pub bulk: bool,
    /// combustion d'une phase granulaire porteuse de son oxydant (poudre noire) : vitesse de régression de la surface des grains à
    /// 1 atm (m/s), exposant de la loi de Vieille, température d'allumage (K) — voir `phases::Reaction::burn_rate`
    #[serde(default)]
    pub burn_rate: f64,
    #[serde(default)]
    pub burn_exponent: f64,
    #[serde(default)]
    pub ignition_temp: f64,
}

/// Mélange nommé : fractions VOLUMIQUES (= molaires pour des gaz parfaits) de fluides de la bibliothèque, normalisées à 1.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MixtureDef {
    pub name: String,
    pub parts: Vec<(String, f64)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialLibrary {
    pub fluids: Vec<Phase>,
    pub solids: Vec<Solid>,
    /// paires de fluides miscibles (par nom)
    pub miscible: Vec<(String, String)>,
    pub changes: Vec<ChangeDef>,
    /// paires de fusion liquide ↔ solide granulaire (eau ↔ glace)
    pub fusions: Vec<FusionDef>,
    pub reactions: Vec<ReactionDef>,
    /// mélanges nommés (air sec, prémélanges…) : pinceaux et fluide ambiant possible
    pub mixtures: Vec<MixtureDef>,
    /// fluide OU mélange qui entre par les bords ouverts / remplit les cellules vidées (par nom ; si absent : premier gaz)
    pub ambient: String,
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        MaterialLibrary {
            fluids: vec![Phase::air(), Phase::water(), Phase::steam(), Phase::oil()],
            solids: vec![Solid::steel(), Solid::glass()],
            miscible: vec![],
            changes: vec![ChangeDef { liquid: "eau".into(), vapor: "vapeur".into(), l_ref: 2.257e6, t_ref: 373.15, p_ref: 101325.0 }],
            fusions: vec![],
            reactions: vec![],
            mixtures: vec![],
            ambient: "air".into(),
        }
    }
}

/// Résultat de `MaterialLibrary::build`
pub struct Built {
    pub registry: Registry,
    /// indice du fluide ambiant (composant principal si l'ambiant est un mélange)
    pub ambient: usize,
    /// composition volumique de l'ambiant si c'est un mélange (sinon vide)
    pub ambient_mix: Vec<f64>,
    /// mélanges (nom, fractions volumiques par indice de fluide)
    pub mixtures: Vec<(String, Vec<f64>)>,
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
    /// première phase granulaire (sable, poudre…)
    pub grains: Option<usize>,
    /// phase granulaire combustible (réactif d'une réaction de surface : charbon…)
    pub coal: Option<usize>,
    /// phase granulaire qui se décompose en chauffant (réactif d'une réaction en volume : bois…)
    pub wood: Option<usize>,
    /// résidu granulaire de cette décomposition (charbon de bois)
    pub charcoal: Option<usize>,
    /// phase granulaire qui brûle sans air (réactif d'une combustion de poudre : poudre noire)
    pub powder: Option<usize>,
    /// solide granulaire d'une paire de fusion (glace)
    pub ice: Option<usize>,
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
        let grains = by_name(&["sable", "sand"], Some(PhaseKind::Granular)).or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_granular()));
        let coal = by_name(&["charbon", "coal", "carbone"], Some(PhaseKind::Granular)).or_else(|| {
            reg.reactions.iter().find_map(|r| r.surface(&reg.phases).map(|s| s.solid))
        });
        // (pyrolyse : réaction en volume qui n'est pas une combustion de poudre)
        let pyro = || reg.reactions.iter().find(|r| r.bulk && r.burn().is_none());
        let wood = by_name(&["bois", "wood"], Some(PhaseKind::Granular)).or_else(|| pyro().and_then(|r| r.reactants.first().map(|&(k, _)| k)));
        let charcoal = by_name(&["charbon de bois", "charcoal"], Some(PhaseKind::Granular))
            .or_else(|| pyro().and_then(|r| r.products.iter().map(|&(k, _)| k).find(|&k| reg.phases[k].is_granular())));
        let powder = by_name(&["poudre noire", "poudre", "gunpowder", "black powder"], Some(PhaseKind::Granular)).or_else(|| {
            reg.reactions.iter().find(|r| r.burn().is_some()).and_then(|r| r.reactants.first().map(|&(k, _)| k))
        });
        let ice = by_name(&["glace", "ice"], Some(PhaseKind::Granular))
            .or_else(|| reg.fusions.iter().find(|f| Some(f.liquid) == water).map(|f| f.solid))
            .or_else(|| reg.fusions.first().map(|f| f.solid));
        Roles { air, water, vapor, oil, plate, grains, coal, wood, charcoal, powder, ice }
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
                mu_law: PropertyLaw::Sutherland { s: 110.4 },
                k_law: PropertyLaw::Constant,
                t_ref: 293.15,
                e0: 0.0,
                sigma: 0.0,
                color: [0.4, 0.8, 0.4],
                molar_mass: 0.029,
                diffusivity: 2.0e-5,
                h_formation: 0.0,
                s_standard: 0.0,
                cp_slope: 0.0,
                radiator: Radiator::None,
                grain_diameter: 0.0,
                alpha_max: 0.6,
                friction_angle: 0.0,
            },
            PhaseKind::Liquid => Phase {
                name: self.unique_name("liquide"),
                kind,
                eos: Eos::Tait { p0: 101325.0, b: 3.0e8, n: 7.0, thermal: Thermal::Beta { rho0: 1000.0, t0: 293.15, beta: 3.0e-4 } },
                cp: 3000.0,
                conductivity: 0.3,
                mu_ref: 1.0e-3,
                mu_law: PropertyLaw::Arrhenius { e_r: 2000.0 },
                k_law: PropertyLaw::Constant,
                t_ref: 293.15,
                e0: 0.0,
                sigma: 0.04,
                color: [0.8, 0.3, 0.7],
                molar_mass: 0.05,
                diffusivity: 1.0e-9,
                h_formation: 0.0,
                s_standard: 0.0,
                cp_slope: 0.0,
                radiator: Radiator::None,
                grain_diameter: 0.0,
                alpha_max: 0.6,
                friction_angle: 0.0,
            },
            PhaseKind::Granular => Phase { name: self.unique_name("grains"), ..Phase::sand() },
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
        for f in self.fusions.iter_mut() {
            if f.liquid == old { f.liquid = new.clone() }
            if f.solid == old { f.solid = new.clone() }
        }
        for r in self.reactions.iter_mut() {
            for (n, _) in r.reactants.iter_mut().chain(r.products.iter_mut()).chain(r.modifiers.iter_mut()) {
                if *n == old { *n = new.clone() }
            }
        }
        for m in self.mixtures.iter_mut() {
            for (n, _) in m.parts.iter_mut() {
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
    /// Supprime un fluide et tout ce qui le référence (miscibilité, changements de phase, paires de fusion, réactions).
    pub fn remove_fluid(&mut self, i: usize) {
        let old = self.fluids.remove(i).name;
        self.miscible.retain(|(a, b)| *a != old && *b != old);
        self.changes.retain(|c| c.liquid != old && c.vapor != old);
        self.fusions.retain(|f| f.liquid != old && f.solid != old);
        self.reactions.retain(|r| !r.reactants.iter().chain(r.products.iter()).any(|(n, _)| *n == old));
        for r in self.reactions.iter_mut() {
            r.modifiers.retain(|(n, _)| *n != old);
        }
        for m in self.mixtures.iter_mut() {
            m.parts.retain(|(n, _)| *n != old);
        }
        self.mixtures.retain(|m| !m.parts.is_empty());
        if self.ambient == old || (self.fluid_index(&self.ambient).is_none() && self.mixture_index(&self.ambient).is_none()) {
            self.ambient = self.fluids.iter().find(|p| p.is_gas()).map(|p| p.name.clone()).unwrap_or_default();
        }
    }
    pub fn mixture_index(&self, name: &str) -> Option<usize> {
        self.mixtures.iter().position(|m| m.name == name)
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
        // réactions d'abord : elles décident de la référence d'énergie chimique
        for r in &self.reactions {
            let conv = |v: &Vec<(String, f64)>| -> Option<Vec<(usize, f64)>> { v.iter().map(|(n, nu)| idx(n).map(|k| (k, *nu))).collect() };
            match (conv(&r.reactants), conv(&r.products), conv(&r.modifiers)) {
                (Some(re), Some(pr), Some(md)) => reg.reactions.push(Reaction {
                    name: r.name.clone(),
                    reactants: re,
                    products: pr,
                    pre_exp: r.pre_exp,
                    activation_energy: r.activation_energy,
                    orders: r.orders.clone(),
                    temp_exponent: r.temp_exponent,
                    reversible: r.reversible,
                    modifiers: md,
                    flame_thickness: r.flame_thickness,
                    bulk: r.bulk,
                    burn_rate: r.burn_rate,
                    burn_exponent: r.burn_exponent,
                    ignition_temp: r.ignition_temp,
                }),
                _ => warnings.push(format!("réaction « {} » ignorée : fluide inexistant", r.name)),
            }
        }
        if !reg.reactions.is_empty() {
            reg.apply_thermochemistry();
        }
        for c in &self.changes {
            match (idx(&c.liquid), idx(&c.vapor)) {
                (Some(l), Some(v)) => {
                    if reg.phases[l].kind != PhaseKind::Liquid || !matches!(reg.phases[v].eos, Eos::IdealGas { .. }) {
                        warnings.push(format!("changement de phase « {} → {} » ignoré : il faut un liquide et un gaz parfait", c.liquid, c.vapor));
                    } else if reg.changes.iter().any(|q| q.liquid == l || q.vapor == v) {
                        warnings.push(format!("changement de phase « {} → {} » ignoré : le liquide ou la vapeur est déjà utilisé par une autre paire", c.liquid, c.vapor));
                    } else {
                        // vapeur produite / consommée par une réaction (H2O de combustion) : c'est elle qui garde son enthalpie de formation
                        let anchor_vapor = !reg.reactions.is_empty() && reg.is_reactive(v);
                        reg.add_phase_change_anchored(l, v, c.l_ref, c.t_ref, c.p_ref, anchor_vapor);
                    }
                }
                _ => warnings.push(format!("changement de phase « {} → {} » ignoré : fluide inexistant", c.liquid, c.vapor)),
            }
        }
        for f in &self.fusions {
            match (idx(&f.liquid), idx(&f.solid)) {
                (Some(l), Some(s)) => {
                    if reg.phases[l].kind != PhaseKind::Liquid || reg.phases[s].kind != PhaseKind::Granular {
                        warnings.push(format!("fusion « {} ↔ {} » ignorée : il faut un liquide et une phase granulaire", f.liquid, f.solid));
                    } else if reg.fusions.iter().any(|q| q.liquid == l || q.solid == s) {
                        warnings.push(format!("fusion « {} ↔ {} » ignorée : le liquide ou le solide est déjà utilisé par une autre paire", f.liquid, f.solid));
                    } else if !(f.t_melt > 0.0 && f.l_fusion > 0.0) {
                        warnings.push(format!("fusion « {} ↔ {} » ignorée : température et chaleur latente doivent être > 0", f.liquid, f.solid));
                    } else if !(f.supercooling >= 0.0 && f.growth_a > 0.0 && f.growth_b >= 0.0) {
                        warnings.push(format!("fusion « {} ↔ {} » ignorée : surfusion ≥ 0, vitesse du front > 0 et exposant ≥ 0 attendus", f.liquid, f.solid));
                    } else {
                        let fu = reg.add_fusion(l, s, f.t_melt, f.l_fusion);
                        fu.supercooling = f.supercooling;
                        fu.growth_a = f.growth_a;
                        fu.growth_b = f.growth_b;
                    }
                }
                _ => warnings.push(format!("fusion « {} ↔ {} » ignorée : fluide inexistant", f.liquid, f.solid)),
            }
        }
        for (a, b) in &self.miscible {
            match (idx(a), idx(b)) {
                (Some(x), Some(y)) if x != y => reg.set_miscible(x, y, true),
                _ => warnings.push(format!("miscibilité « {a} ↔ {b} » ignorée : fluide inexistant")),
            }
        }
        warnings.extend(reg.validate());
        // mélanges
        let np = reg.phases.len();
        let mut mixtures = vec![];
        for m in &self.mixtures {
            let mut a = vec![0.0; np];
            let mut ok = true;
            for (n, x) in &m.parts {
                match idx(n) {
                    Some(k) if *x >= 0.0 => a[k] += x,
                    _ => ok = false,
                }
            }
            let tot: f64 = a.iter().sum();
            if !ok || !(tot > 0.0) {
                warnings.push(format!("mélange « {} » ignoré : composant inexistant ou fractions nulles", m.name));
                continue;
            }
            a.iter_mut().for_each(|x| *x /= tot);
            mixtures.push((m.name.clone(), a));
        }
        let mut ambient_mix = vec![];
        let ambient_named_mix = mixtures.iter().find(|(n, _)| *n == self.ambient).map(|(_, a)| a.clone());
        let ambient = if let Some(a) = ambient_named_mix.filter(|a| (0..np).all(|k| a[k] == 0.0 || reg.phases[k].is_gas())) {
            let main = (0..np).max_by(|&x, &y| a[x].partial_cmp(&a[y]).unwrap()).unwrap();
            ambient_mix = a;
            main
        } else {
            if mixtures.iter().any(|(n, _)| *n == self.ambient) {
                warnings.push(format!("mélange ambiant « {} » ignoré : l'ambiant doit être gazeux", self.ambient));
            }
            idx(&self.ambient)
                .filter(|&k| reg.phases[k].is_gas())
                .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas() && !reg.changes.iter().any(|c| c.vapor == k)))
                .or_else(|| (0..reg.phases.len()).find(|&k| reg.phases[k].is_gas()))
                .ok_or_else(|| "aucun gaz défini : un gaz ambiant est nécessaire (il remplit les bords ouverts et les cellules vidées)".to_string())?
        };
        Ok(Built { registry: reg, ambient, ambient_mix, mixtures, warnings })
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

/// Nom des matériaux du préréglage combustion
pub const O2: &str = "oxygène";
pub const N2: &str = "azote";
pub const CH4: &str = "méthane";
pub const H2: &str = "hydrogène";
pub const CO: &str = "monoxyde de carbone";
pub const CO2: &str = "dioxyde de carbone";
pub const H2O_G: &str = "vapeur";
pub const AIR_SEC: &str = "air sec (O2 + N2)";
pub const MIX_CH4_AIR: &str = "méthane–air stœchiométrique";
pub const MIX_H2_AIR: &str = "hydrogène–air stœchiométrique";
pub const RX_CH4: &str = "combustion du méthane (Westbrook–Dryer)";
pub const RX_H2: &str = "combustion de l'hydrogène ⇌ dissociation de H2O";
pub const RX_CO: &str = "oxydation de CO ⇌ dissociation de CO2";
pub const COAL: &str = "charbon";
pub const RX_COAL: &str = "combustion du charbon (C + O2 → CO2, surface des grains)";
pub const RX_COAL_CO: &str = "gazéification du charbon (C + ½O2 → CO, surface des grains)";
pub const RX_BOUDOUARD: &str = "réduction de CO2 par le charbon (C + CO2 → 2 CO, Boudouard)";
pub const WOOD: &str = "bois";
pub const CHARCOAL: &str = "charbon de bois";
pub const RX_WOOD: &str = "pyrolyse du bois (bois → charbon de bois + CO, CO2, H2O, CH4, H2)";
pub const RX_CHARCOAL_CO: &str = "gazéification du charbon de bois (C + ½O2 → CO, surface des grains)";
pub const RX_CHARCOAL: &str = "combustion du charbon de bois (C + O2 → CO2, surface des grains)";
pub const RX_CHARCOAL_BOUD: &str = "réduction de CO2 par le charbon de bois (C + CO2 → 2 CO)";
pub const HOUILLE: &str = "houille";
pub const RX_HOUILLE: &str = "pyrolyse de la houille (houille → charbon + CH4, CO, CO2, H2O, H2)";
pub const GLACE: &str = "glace";
pub const CIRE: &str = "cire";
pub const CIRE_FONDUE: &str = "cire fondue";
pub const LAVE: &str = "lave";
pub const BASALTE: &str = "basalte";
/// surfusion maximale d'une eau très pure dans un récipient lisse (K) : préréglage `with_supercooled_water`
pub const SUPERCOOLING_PURE_WATER: f64 = 20.0;
pub const POUDRE: &str = "poudre noire";
pub const FUMEE: &str = "fumée de poudre";
pub const RX_POUDRE: &str = "combustion de la poudre noire (→ CO2, CO, N2, H2O, H2, fumée)";

/// Gaz parfait de combustion. cp(T) = cp + pente·(T − 298.15) : les deux coefficients sont ajustés pour reproduire EXACTEMENT
/// h(2000 K) − h(298 K) et s°(2000 K) des tables NIST-JANAF (donc la température de flamme et les équilibres de dissociation vers
/// 2000 K) ; cp(298 K) reste à quelques % de la valeur tabulée. μ, k à 293 K avec lois de Sutherland ; D = diffusivité dans l'air à 293 K.
#[allow(clippy::too_many_arguments)]
fn gas(name: &str, m: f64, cp: f64, slope: f64, mu: f64, s_mu: f64, k: f64, d: f64, hf: f64, s0: f64, color: [f32; 3], rad: Radiator) -> Phase {
    Phase {
        name: name.into(),
        kind: PhaseKind::Gas,
        eos: Eos::IdealGas { r_spec: R_GAS / m },
        cp,
        conductivity: k,
        mu_ref: mu,
        mu_law: PropertyLaw::Sutherland { s: s_mu as f32 },
        k_law: PropertyLaw::Sutherland { s: (1.5 * s_mu) as f32 },
        t_ref: 293.15,
        e0: 0.0,
        sigma: 0.0,
        color,
        molar_mass: m,
        diffusivity: d,
        h_formation: hf,
        s_standard: s0,
        cp_slope: slope,
        radiator: rad,
        grain_diameter: 0.0,
        alpha_max: 0.6,
        friction_angle: 0.0,
    }
}

/// cp(298 K) et pente (J/(kg·K), J/(kg·K²)) de H2O(g) ajustés comme ci-dessus (JANAF : h₂₀₀₀ − h₂₉₈ = 72.79 kJ/mol, s°₂₀₀₀ = 264.97 J/(mol·K))
pub const H2O_CP: (f64, f64) = (1860.5, 0.6037);

/// Fluides du préréglage combustion (Δh_f°, s° : tables NIST-JANAF, état gazeux, 298.15 K, 1 bar).
pub fn combustion_fluids() -> Vec<Phase> {
    vec![
        gas(O2, 0.031998, 944.2, 0.1679, 2.04e-5, 127.0, 0.0259, 2.0e-5, 0.0, 205.15, [0.55, 0.75, 1.0], Radiator::None),
        gas(N2, 0.028014, 1032.0, 0.1711, 1.76e-5, 111.0, 0.0255, 2.0e-5, 0.0, 191.61, [0.20, 0.22, 0.30], Radiator::None),
        gas(CH4, 0.016043, 2226.0, 2.9, 1.10e-5, 198.0, 0.0332, 2.1e-5, -74.87e3, 186.25, [0.30, 0.85, 0.35], Radiator::None),
        gas(H2, 0.002016, 14147.0, 1.511, 8.8e-6, 72.0, 0.180, 7.5e-5, 0.0, 130.68, [1.0, 0.55, 0.80], Radiator::None),
        gas(CO, 0.028010, 1039.6, 0.1770, 1.75e-5, 118.0, 0.0250, 2.0e-5, -110.53e3, 197.66, [0.95, 0.55, 0.20], Radiator::None),
        gas(CO2, 0.044009, 951.4, 0.3168, 1.47e-5, 240.0, 0.0162, 1.6e-5, -393.52e3, 213.79, [0.45, 0.45, 0.45], Radiator::CarbonDioxide),
    ]
}

/// Mécanismes globaux (unités SI, mol/m³) :
///  * CH4 + 2 O2 → CO2 + 2 H2O : forme de Westbrook & Dryer (1981), Ea = 48.4 kcal/mol, ordres [CH4]^−0.3 [O2]^1.3 ; A RECALÉ (1.5e10 s⁻¹
///    au lieu de 1.3e8) pour que la flamme stœchiométrique résolue de CE solveur (transport, thermochimie) ait la vitesse mesurée
///    S_L = 0.38 m/s — c'est ainsi que sont construits les mécanismes globaux à une étape (calage sur S_L).
///  * H2 + ½ O2 ⇌ H2O : forme de Marinov et al. (1996), Ea = 35 kcal/mol, [H2]^1 [O2]^0.5 ; A recalé à 1.5e10 SI (1.8e10 d'origine) pour
///    S_L = 2.2 m/s (stœchiométrique). Réversible → dissociation de la vapeur d'eau aux températures de flamme.
///  Vitesses obtenues (flamme épaissie, moyenne sur la course dans un tube) : CH4 0.33–0.43 m/s et H2 1.9–2.5 m/s pour des mailles de
///  0.05 mm à 1 cm — la flamme épaissie conserve S_L à ±15 % quelle que soit la grille.
///  * CO + ½ O2 ⇌ CO2 : Westbrook & Dryer, A = 10^14.6 cgs = 1.26e10 SI, Ea = 40 kcal/mol, [CO]^1 [O2]^0.25 [H2O]^0.5 ; réversible →
///    dissociation de CO2.
pub fn combustion_reactions() -> Vec<ReactionDef> {
    let kcal = 4184.0;
    vec![
        ReactionDef {
            name: RX_CH4.into(),
            reactants: vec![(CH4.into(), 1.0), (O2.into(), 2.0)],
            products: vec![(CO2.into(), 1.0), (H2O_G.into(), 2.0)],
            pre_exp: 1.5e10,
            activation_energy: 48.4 * kcal,
            orders: vec![-0.3, 1.3],
            flame_thickness: 0.45e-3,
            ..Default::default()
        },
        ReactionDef {
            name: RX_H2.into(),
            reactants: vec![(H2.into(), 1.0), (O2.into(), 0.5)],
            products: vec![(H2O_G.into(), 1.0)],
            pre_exp: 1.5e10,
            activation_energy: 35.0 * kcal,
            orders: vec![1.0, 0.5],
            reversible: true,
            flame_thickness: 0.35e-3,
            ..Default::default()
        },
        ReactionDef {
            name: RX_CO.into(),
            reactants: vec![(CO.into(), 1.0), (O2.into(), 0.5)],
            products: vec![(CO2.into(), 1.0)],
            pre_exp: 1.26e10,
            activation_energy: 40.0 * kcal,
            orders: vec![1.0, 0.25],
            reversible: true,
            modifiers: vec![(H2O_G.into(), 0.5)],
            ..Default::default()
        },
    ]
}

/// Combustion du charbon à la surface des grains (réactions de surface, voir `phases::Reaction`). Cinétique du char de Field (1969),
/// 8 710 exp(−17 967/T) g/(cm²·s·atm O2), convertie en vitesse de surface k_s = A·T·exp(−Ea/RT) m/s par p_O2 = C_O2 R T :
/// A = 87 100 R / (M_C · 101 325) = 596 m/(s·K), Ea = 149.4 kJ/mol. Le carbone part en CO et en CO2 dans le rapport d'Arthur (1951),
/// CO/CO2 = 2500 exp(−6240/T) (1 vers 800 K, 14 à 1200 K) : C + ½O2 → CO porte la cinétique de Field, C + O2 → CO2 la même divisée par
/// le rapport (A/2500, Ea − 6240 R). Le CO brûle ensuite dans le gaz au-dessus des braises (CO + ½O2 ⇌ CO2, flamme bleue ; l'humidité
/// de l'air la catalyse). Les deux réactions partagent l'oxygène qui diffuse jusqu'aux grains (k_d = 2 D_O2/d) : à 1500 K, k_s ≈ 5.7 m/s
/// contre k_d ≈ 2.3 m/s pour des grains de 0.3 mm, la combustion vive est limitée par l'apport d'oxygène (régime III) ; elle démarre
/// vers 800–900 K.
pub fn coal_reactions() -> Vec<ReactionDef> {
    let a = 87100.0 * R_GAS / (0.012011 * 101325.0);
    vec![
        ReactionDef {
            name: RX_COAL_CO.into(),
            reactants: vec![(COAL.into(), 1.0), (O2.into(), 0.5)],
            products: vec![(CO.into(), 1.0)],
            pre_exp: a,
            activation_energy: 17967.0 * R_GAS,
            orders: vec![0.0, 1.0],
            temp_exponent: 1.0,
            ..Default::default()
        },
        ReactionDef {
            name: RX_COAL.into(),
            reactants: vec![(COAL.into(), 1.0), (O2.into(), 1.0)],
            products: vec![(CO2.into(), 1.0)],
            pre_exp: a / 2500.0,
            activation_energy: (17967.0 - 6240.0) * R_GAS,
            orders: vec![0.0, 1.0],
            temp_exponent: 1.0,
            ..Default::default()
        },
        // Boudouard (endothermique, +172.5 kJ/mol) : dans la zone chaude d'un lit où l'oxygène est épuisé, le CO2 est réduit en CO, qui
        // brûle au-dessus des braises au contact de l'air (flamme bleue). Ea = 250 kJ/mol, A tel que C–CO2 soit 1e5 fois plus lent que
        // C–O2 à 1073 K (Laurendeau 1978) : négligeable sous 1100 K, k_s ≈ 3e-4 m/s à 1400 K, 1.4e-3 m/s à 1500 K
        ReactionDef {
            name: RX_BOUDOUARD.into(),
            reactants: vec![(COAL.into(), 1.0), (CO2.into(), 1.0)],
            products: vec![(CO.into(), 2.0)],
            pre_exp: 470.0,
            activation_energy: 250.0e3,
            orders: vec![0.0, 1.0],
            temp_exponent: 1.0,
            ..Default::default()
        },
    ]
}

/// Pyrolyse du bois : réaction EN VOLUME (le bois chauffé se décompose dans toute la masse du morceau), du premier ordre en bois,
/// CH1.44O0.66 → 0.44 C + 0.30 CO + 0.08 CO2 + 0.20 H2O + 0.18 CH4 + 0.16 H2. Rendements massiques : charbon de bois 22 % (le « carbone
/// fixe » d'un bois sec), gaz combustibles 49 % (CO 35 %, CH4 12 %, H2 1.3 %), CO2 15 %, eau 15 % ; les goudrons, qui se craquent et
/// brûlent dans la flamme, sont comptés en CH4 et H2 (gaz sec : CO 42 %, CH4 25 %, H2 22 %, CO2 11 % en volume). Chaleur de pyrolyse
/// +0.18 MJ/kg (légèrement endothermique ; mesures : −0.5 à +0.5 MJ/kg). Vitesse : somme des trois voies (gaz, goudrons, charbon) de
/// Shafizadeh & Chin avec les constantes de Thurner & Mann (1981, chêne), approchée par une seule loi d'Arrhenius entre 700 et 800 K :
/// A = 3.7e6 s⁻¹, Ea = 108.8 kJ/mol — 13 min à 600 K, 36 s à 700 K, 3.4 s à 800 K, 0.56 s à 900 K. Le charbon de bois brûle ensuite
/// comme le charbon (réactions de surface : cinétique de Field, rapport CO/CO2 d'Arthur, Boudouard).
pub fn wood_reactions() -> Vec<ReactionDef> {
    let mut v = vec![ReactionDef {
        name: RX_WOOD.into(),
        reactants: vec![(WOOD.into(), 1.0)],
        products: vec![(CHARCOAL.into(), 0.44), (CO.into(), 0.30), (CO2.into(), 0.08), (H2O_G.into(), 0.20), (CH4.into(), 0.18), (H2.into(), 0.16)],
        pre_exp: 3.7e6,
        activation_energy: 108.8e3,
        orders: vec![1.0],
        bulk: true,
        ..Default::default()
    }];
    for (r, name) in coal_reactions().into_iter().zip([RX_CHARCOAL_CO, RX_CHARCOAL, RX_CHARCOAL_BOUD]) {
        let mut r = r;
        r.name = name.into();
        r.reactants[0].0 = CHARCOAL.into();
        v.push(r);
    }
    v
}

/// Pyrolyse de la houille (réaction EN VOLUME, du premier ordre, comme le bois) : CH0.8O0.08 → 0.763 C + 0.187 CH4 + 0.04 CO + 0.01 CO2
/// + 0.02 H2O + 0.006 H2. Matières volatiles 35 % de la masse (CH4 21 %, CO 8 %, CO2 3 %, eau 2.6 %, H2 0.1 % ; goudrons comptés en CH4),
/// le reste est le char `coal` ; chaleur de pyrolyse +0.08 MJ/kg. Vitesse de dégazage de Badzioch & Hawksley (1970, charbons
/// bitumineux) : A = 3.12e5 s⁻¹, Ea = 74 kJ/mol (0.1 s⁻¹ à 600 K, 1 s⁻¹ à 700 K), bornée par la conduction interne des morceaux
/// (k / (1 + k τ) ; 1.2 cm : τ ≈ 19 s). Ce sont ces gaz qui font la grande flamme d'un feu de houille ; le char brûle ensuite en braises.
pub fn houille_reactions() -> Vec<ReactionDef> {
    vec![ReactionDef {
        name: RX_HOUILLE.into(),
        reactants: vec![(HOUILLE.into(), 1.0)],
        products: vec![(COAL.into(), 0.763), (CH4.into(), 0.187), (CO.into(), 0.04), (CO2.into(), 0.01), (H2O_G.into(), 0.02), (H2.into(), 0.006)],
        pre_exp: 3.12e5,
        activation_energy: 74.0e3,
        orders: vec![1.0],
        bulk: true,
        ..Default::default()
    }]
}

/// Combustion de la poudre noire (réaction EN VOLUME qui porte son oxydant, `burn_rate`) : une « mole » de 100 g de poudre (75 % KNO3,
/// 15 % charbon de bois, 10 % S) → 0.60 CO2 + 0.15 CO + 0.371 N2 + 0.05 H2O + 0.03 H2 (42 % de la masse, 12.0 mol/kg : 269 L/kg aux
/// conditions normales) + 58 % de fumée (K2CO3, K2SO4, K2S). Analyse de Noble & Abel : 43 % de gaz (en volume CO2 49 %, N2 33 %,
/// CO 12 %), 56 % de solides, 274 cm³/g, 725.7 cal/g. Chaleur d'explosion 3.0 MJ/kg, flamme vers 2500 K. La surface des grains régresse
/// à r = 1.1 cm/s (p / 1 atm)^0.15 (2.3 s par pouce à l'air libre ; exposant 0.13–0.16 mesuré au-dessus de 0.3 MPa : la poudre noire
/// est peu sensible à la pression) dès qu'ils atteignent 580 K (auto-allumage vers 580 °F)
pub fn gunpowder_reactions() -> Vec<ReactionDef> {
    vec![ReactionDef {
        name: RX_POUDRE.into(),
        reactants: vec![(POUDRE.into(), 1.0)],
        products: vec![(CO2.into(), 0.60), (CO.into(), 0.15), (N2.into(), 0.371), (H2O_G.into(), 0.05), (H2.into(), 0.03), (FUMEE.into(), 1.0)],
        orders: vec![1.0],
        bulk: true,
        burn_rate: 0.011,
        burn_exponent: 0.15,
        ignition_temp: 580.0,
        ..Default::default()
    }]
}

/// fractions molaires de l'air sec : O2 20.95 %, N2 + Ar 79.05 %
const X_O2_AIR: f64 = 0.2095;

pub fn combustion_mixtures() -> Vec<MixtureDef> {
    let r = (1.0 - X_O2_AIR) / X_O2_AIR; // N2 par O2
    let stoich = |fuel: &str, o2_per_fuel: f64| {
        let tot = 1.0 + o2_per_fuel * (1.0 + r);
        MixtureDef { name: String::new(), parts: vec![(fuel.into(), 1.0 / tot), (O2.into(), o2_per_fuel / tot), (N2.into(), o2_per_fuel * r / tot)] }
    };
    vec![
        MixtureDef { name: AIR_SEC.into(), parts: vec![(O2.into(), X_O2_AIR), (N2.into(), 1.0 - X_O2_AIR)] },
        MixtureDef { name: MIX_CH4_AIR.into(), ..stoich(CH4, 2.0) },
        MixtureDef { name: MIX_H2_AIR.into(), ..stoich(H2, 0.5) },
    ]
}

impl MaterialLibrary {
    /// Ajoute (sans rien supprimer ni écraser) le préréglage combustion : O2, N2, CH4, H2, CO, CO2, les réactions, les mélanges et la
    /// miscibilité de tous les gaz entre eux (un gaz se mélange toujours à un autre gaz). La vapeur existante reçoit la thermochimie de
    /// H2O(g) (pente de cp, rayonnement) si elle n'en a pas ; elle est créée sinon.
    pub fn add_combustion_preset(&mut self) {
        for f in combustion_fluids() {
            if self.fluid_index(&f.name).is_none() {
                self.fluids.push(f);
            }
        }
        match self.fluid_index(H2O_G) {
            Some(i) => {
                let v = &mut self.fluids[i];
                if v.is_gas() {
                    if v.cp_slope == 0.0 {
                        (v.cp, v.cp_slope) = H2O_CP;
                    }
                    if v.radiator == Radiator::None {
                        v.radiator = Radiator::Water;
                    }
                    if v.h_formation == 0.0 {
                        v.h_formation = -241.826e3;
                        v.s_standard = 188.84;
                    }
                }
            }
            None => {
                let mut v = Phase::steam();
                (v.cp, v.cp_slope) = H2O_CP;
                self.fluids.push(v);
            }
        }
        for r in combustion_reactions() {
            if !self.reactions.iter().any(|q| q.name == r.name) {
                self.reactions.push(r);
            }
        }
        for m in combustion_mixtures() {
            if self.mixture_index(&m.name).is_none() {
                self.mixtures.push(m);
            }
        }
        let gases: Vec<String> = self.fluids.iter().filter(|p| p.is_gas()).map(|p| p.name.clone()).collect();
        for (i, a) in gases.iter().enumerate() {
            for b in &gases[i + 1..] {
                if !self.is_miscible(a, b) {
                    self.set_miscible(a, b, true);
                }
            }
        }
    }
    /// Copie avec le préréglage combustion et l'air sec (O2 + N2) comme fluide ambiant (scénarios de flammes / explosions).
    /// Bibliothèque avec au moins une phase granulaire (ajoute le sable s'il n'y en a aucune) : scénarios à grains
    pub fn with_granular(&self) -> MaterialLibrary {
        let mut l = self.clone();
        if !l.fluids.iter().any(|p| p.is_granular()) {
            let mut sand = Phase::sand();
            sand.name = l.unique_name("sable");
            l.fluids.push(sand);
        }
        l
    }

    pub fn with_combustion(&self) -> MaterialLibrary {
        let mut l = self.clone();
        l.add_combustion_preset();
        l.ambient = AIR_SEC.into();
        l
    }

    /// Préréglage combustion + bois et charbon de bois (phases granulaires), pyrolyse du bois et combustion du charbon de bois (scénario
    /// « feu de bois »)
    pub fn with_wood(&self) -> MaterialLibrary {
        let mut l = self.with_combustion();
        for p in [Phase::wood(), Phase::charcoal()] {
            if l.fluid_index(&p.name).is_none() {
                l.fluids.push(p);
            }
        }
        for r in wood_reactions() {
            if !l.reactions.iter().any(|q| q.name == r.name) {
                l.reactions.push(r);
            }
        }
        l
    }

    /// Préréglage combustion + charbon (phase granulaire) et sa combustion de surface, houille et sa pyrolyse (scénario « feu de charbon »)
    pub fn with_coal(&self) -> MaterialLibrary {
        let mut l = self.with_combustion();
        if l.fluid_index(COAL).is_none() {
            l.fluids.push(Phase::coal());
        }
        if l.fluid_index(HOUILLE).is_none() {
            l.fluids.push(Phase::bituminous_coal());
        }
        for r in coal_reactions().into_iter().chain(houille_reactions()) {
            if !l.reactions.iter().any(|q| q.name == r.name) {
                l.reactions.push(r);
            }
        }
        l
    }

    /// Glace (cristaux, phase granulaire) et gel / fusion de l'eau : T_f = 273.15 K, L_f = 333.6 kJ/kg (eau ajoutée si absente)
    pub fn with_ice(&self) -> MaterialLibrary {
        let mut l = self.clone();
        if l.fluid_index("eau").is_none() {
            l.fluids.push(Phase::water());
        }
        if l.fluid_index(GLACE).is_none() {
            l.fluids.push(Phase::ice());
        }
        if !l.fusions.iter().any(|f| f.solid == GLACE) {
            l.fusions.push(FusionDef::new("eau", GLACE, 273.15, 333.55e3));
        }
        l
    }

    /// Cire de paraffine fondue et en paillettes, et sa paire de fusion : elle fige à 58 °C, L_f = 200 kJ/kg
    pub fn with_wax(&self) -> MaterialLibrary {
        let mut l = self.clone();
        if l.fluid_index(CIRE_FONDUE).is_none() {
            l.fluids.push(Phase::liquid_wax());
        }
        if l.fluid_index(CIRE).is_none() {
            l.fluids.push(Phase::wax());
        }
        if !l.fusions.iter().any(|f| f.solid == CIRE) {
            l.fusions.push(FusionDef::new(CIRE_FONDUE, CIRE, 331.15, 200.0e3));
        }
        l
    }

    /// Lave basaltique, basalte en grains (sable noir) et leur paire de fusion — solidification à 1100 °C, entre le liquidus (~1200 °C)
    /// et le solidus (~1000 °C) d'un basalte, L_f = 400 kJ/kg —, roche volcanique (solide)
    pub fn with_lava(&self) -> MaterialLibrary {
        let mut l = self.clone();
        if l.fluid_index(LAVE).is_none() {
            l.fluids.push(Phase::lava());
        }
        if l.fluid_index(BASALTE).is_none() {
            l.fluids.push(Phase::basalt());
        }
        if !l.fusions.iter().any(|f| f.solid == BASALTE) {
            l.fusions.push(FusionDef::new(LAVE, BASALTE, 1373.15, 400.0e3));
        }
        if l.solid_index("roche").is_none() {
            l.solids.push(Solid::rock());
        }
        l
    }

    /// Comme `with_ice`, avec une eau très pure qui se surfond : germination spontanée seulement sous −20 °C
    /// (`SUPERCOOLING_PURE_WATER` ; l'eau du robinet gèle vers −5 °C, de fines gouttelettes pures vers −38 °C) ; au-dessus, il faut un germe
    /// de glace (des cristaux, ou le front de dendrites venu d'une voisine)
    pub fn with_supercooled_water(&self) -> MaterialLibrary {
        let mut l = self.with_ice();
        for f in l.fusions.iter_mut().filter(|f| f.solid == GLACE) {
            f.supercooling = f.supercooling.max(SUPERCOOLING_PURE_WATER);
        }
        l
    }

    /// Préréglage combustion + poudre noire (phase granulaire qui brûle sans air), sa fumée et sa combustion (scénario « tas de
    /// poudre »)
    pub fn with_gunpowder(&self) -> MaterialLibrary {
        let mut l = self.with_combustion();
        for p in [Phase::gunpowder(), Phase::powder_smoke()] {
            if l.fluid_index(&p.name).is_none() {
                l.fluids.push(p);
            }
        }
        for r in gunpowder_reactions() {
            if !l.reactions.iter().any(|q| q.name == r.name) {
                l.reactions.push(r);
            }
        }
        l
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
        lib.reactions.push(ReactionDef { name: "ok".into(), reactants: vec![("eau".into(), 2.0)], products: vec![("vapeur".into(), 2.0)], pre_exp: 1.0, activation_energy: 1e4, ..Default::default() });
        lib.reactions.push(ReactionDef { name: "bad".into(), reactants: vec![("eau".into(), 1.0)], products: vec![("huile".into(), 1.0)], pre_exp: 1.0, activation_energy: 1e4, ..Default::default() });
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
