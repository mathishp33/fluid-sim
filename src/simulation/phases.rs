//! Registre multi-matériaux : gaz, liquides (EOS réelles) et solides, + paires de changement de phase.
//!
//! Tout est en f64 : p ~ 1e5 Pa pour des liquides de module ~3e8 Pa, la résolution f32 (1e-7 relatif) ne
//! suffit pas à représenter proprement l'hydrostatique + la compressibilité.
//!
//! Modèles :
//!  * Gaz : gaz parfait ρ = p/(R T), cv = cp − R, c² = γ R T.
//!  * Liquide : équation de Tait  (p + B)/(p0 + B) = (ρ/ρ_T(T))^n  (eau : B = 3.07e8 Pa, n = 7.15 → c = 1482 m/s à 20 °C),
//!    ρ_T(T) = masse volumique à p0 (formule de Kell pour l'eau : maximum à 4 °C, ou dilatation β constante).
//!  * Changement de phase : Clausius–Clapeyron avec chaleur latente de Kirchhoff L(T) = L_ref + (cp_v − cp_l)(T − T_ref),
//!    p_sat(T) intégrée analytiquement ; les décalages d'énergie e0 sont fixés pour que h_v − h_l = L(T) EXACTEMENT
//!    (la chaleur latente restituée par le bilan d'énergie est donc celle utilisée par p_sat : cohérence thermodynamique).

use super::materials::PropertyLaw;
use serde::{Deserialize, Serialize};

/// Température et pression de l'état standard thermochimique (enthalpies de formation, entropies standard).
pub const T_STD: f64 = 298.15;
pub const P_STD: f64 = 1.0e5;
/// Constante des gaz parfaits (J/(mol·K))
pub const R_GAS: f64 = 8.314462618;

/// Rayonnement thermique d'un gaz (modèle optiquement mince, coefficients d'absorption moyens de Planck du TNF Workshop / RADCAL).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum Radiator {
    /// transparent (N2, O2, air, liquides…)
    #[default]
    None,
    /// vapeur d'eau H2O
    Water,
    /// dioxyde de carbone CO2
    CarbonDioxide,
}

impl Radiator {
    /// coefficient d'absorption moyen de Planck a_p (1/(m·atm)) ; polynômes en 1000/T valides 300–2500 K (T borné à [500 ; 2500] K)
    pub fn planck_mean(self, t: f64) -> f64 {
        let t = t.clamp(500.0, 2500.0);
        let x = 1000.0 / t;
        let poly = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &ci| acc * x + ci);
        match self {
            Radiator::None => 0.0,
            // TNF : a = c0 + c1 (1000/T) + … + c5 (1000/T)^5
            Radiator::Water => poly(&[-0.23093, -1.12390, 9.41530, -2.99880, 0.51382, -1.86840e-5]).max(0.0),
            Radiator::CarbonDioxide => poly(&[18.741, -121.310, 273.500, -194.050, 56.310, -5.8169]).max(0.0),
        }
    }
}

/// Nombre maximal de phases fluides gérées par les solveurs CPU et GPU (`gpu_multiphase::GPU_MAX_PHASES`).
pub const MAX_PHASES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PhaseKind {
    Gas,
    Liquid,
    /// grains (sable, poudre, charbon…) : phase dispersée sans interface, qui sédimente (voir GRANULAIRE.md et `mp_granular`)
    Granular,
}

/// Masse volumique d'un liquide à la pression de référence p0, en fonction de T.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Thermal {
    /// ρ = rho0 / (1 + β (T − t0))
    Beta { rho0: f64, t0: f64, beta: f64 },
    /// Formule de Kell (eau pure, 0–150 °C, précision ~1e-4 relatif), anomalie à 4 °C comprise.
    Kell,
}

impl Thermal {
    pub fn rho_p0(&self, t: f64) -> f64 {
        match *self {
            Thermal::Beta { rho0, t0, beta } => rho0 / (1.0 + beta * (t - t0)),
            Thermal::Kell => {
                let c = (t - 273.15).clamp(0.0, 150.0);
                let num = 999.83952 + 16.945176 * c - 7.9870401e-3 * c * c - 46.170461e-6 * c.powi(3)
                    + 105.56302e-9 * c.powi(4)
                    - 280.54253e-12 * c.powi(5);
                num / (1.0 + 16.879850e-3 * c)
            }
        }
    }
    /// Coefficient de dilatation volumique β = −(1/ρ) dρ/dT
    pub fn beta(&self, t: f64) -> f64 {
        match *self {
            Thermal::Beta { beta, t0, .. } => beta / (1.0 + beta * (t - t0)),
            Thermal::Kell => {
                let h = 0.05;
                -(self.rho_p0(t + h) - self.rho_p0(t - h)) / (2.0 * h) / self.rho_p0(t)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Eos {
    IdealGas { r_spec: f64 },
    Tait { p0: f64, b: f64, n: f64, thermal: Thermal },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub name: String,
    pub kind: PhaseKind,
    pub eos: Eos,
    pub cp: f64,           // J/(kg·K) isobare
    pub conductivity: f64, // W/(m·K) à t_ref
    pub mu_ref: f64,       // Pa·s à t_ref
    pub mu_law: PropertyLaw,
    pub k_law: PropertyLaw,
    pub t_ref: f64,
    pub e0: f64,           // décalage d'énergie interne (J/kg), fixé par `Registry::add_phase_change`
    pub sigma: f64,        // tension superficielle de CE liquide face aux gaz (N/m) ; 0 pour un gaz
    /// couleur d'affichage (RGB 0..1)
    #[serde(default = "default_color")]
    pub color: [f32; 3],
    /// masse molaire (kg/mol) : sert aux bilans stœchiométriques des réactions chimiques
    #[serde(default = "default_molar_mass")]
    pub molar_mass: f64,
    /// coefficient de diffusion (m²/s) dans les mélanges miscibles (diffusion de Fick entre espèces d'un même groupe)
    #[serde(default)]
    pub diffusivity: f64,
    /// enthalpie standard de formation Δh_f° (J/mol) à 298.15 K, DANS L'ÉTAT DE CE FLUIDE (H2O liquide : −285.83 kJ/mol, vapeur : −241.83) ;
    /// fixe la référence d'énergie chimique quand des réactions sont définies (voir `Registry::apply_thermochemistry`)
    #[serde(default)]
    pub h_formation: f64,
    /// entropie standard s° (J/(mol·K)) à 298.15 K et 1 bar : constante d'équilibre des réactions réversibles
    #[serde(default)]
    pub s_standard: f64,
    /// variation linéaire de la chaleur massique : cp(T) = cp + cp_slope·(T − 298.15) (J/(kg·K²)) ; 0 = cp constant
    #[serde(default)]
    pub cp_slope: f64,
    /// rayonnement thermique (gaz de combustion)
    #[serde(default)]
    pub radiator: Radiator,
    /// phase granulaire : diamètre des grains (m) — vitesse de chute, surface d'échange thermique (6 α / d)
    #[serde(default)]
    pub grain_diameter: f64,
    /// phase granulaire : fraction volumique maximale des grains tassés (empilement aléatoire ≈ 0.6)
    #[serde(default = "default_alpha_max")]
    pub alpha_max: f64,
    /// phase granulaire : angle de frottement interne φ (degrés) — frottement de Coulomb des grains tassés (angle de talus ≈ φ) ;
    /// 0 = sans frottement (les grains tassés coulent comme un liquide lourd). Absent d'un fichier de matériaux antérieur : 32° (sable),
    /// sans effet sur les phases non granulaires
    #[serde(default = "default_friction_angle")]
    pub friction_angle: f64,
}

fn default_alpha_max() -> f64 {
    0.6
}

fn default_friction_angle() -> f64 {
    32.0
}

fn default_color() -> [f32; 3] {
    [0.6, 0.6, 0.6]
}
fn default_molar_mass() -> f64 {
    0.018
}

impl Phase {
    pub fn is_gas(&self) -> bool {
        self.kind == PhaseKind::Gas
    }
    pub fn is_granular(&self) -> bool {
        self.kind == PhaseKind::Granular
    }
    /// Chaleur massique pour l'énergie interne e = c·T + e0 (cv pour un gaz ; cp ≈ cv pour un liquide, écart < 1 % pour l'eau)
    pub fn c_e(&self) -> f64 {
        match self.eos {
            Eos::IdealGas { r_spec } => self.cp - r_spec,
            Eos::Tait { .. } => self.cp,
        }
    }
    /// gaz parfait : constante spécifique R/M (J/(kg·K)) ; liquide : 0
    pub fn r_spec(&self) -> f64 {
        match self.eos {
            Eos::IdealGas { r_spec } => r_spec,
            Eos::Tait { .. } => 0.0,
        }
    }
    /// chaleur massique de l'énergie interne à T : de/dT = c_e + cp_slope·(T − 298.15)
    pub fn c_e_at(&self, t: f64) -> f64 {
        self.c_e() + self.cp_slope * (t - T_STD)
    }
    pub fn cp_at(&self, t: f64) -> f64 {
        self.cp + self.cp_slope * (t - T_STD)
    }
    /// énergie interne massique e(T) = c_e·T + ½·cp_slope·(T − 298.15)² + e0
    pub fn e_int(&self, t: f64) -> f64 {
        let d = t - T_STD;
        self.c_e() * t + 0.5 * self.cp_slope * d * d + self.e0
    }
    /// enthalpie massique h = e + p/ρ (gaz parfait : + r T ; liquide : p/ρ ≈ 100 J/kg négligé, cohérent avec e0)
    pub fn h_mass(&self, t: f64) -> f64 {
        self.e_int(t) + self.r_spec() * t
    }
    /// coefficient de diffusion moléculaire à (T, p) : gaz D ∝ T^1.75 / p (Fuller / Chapman–Enskog), liquide D ∝ T / μ(T) (Stokes–Einstein)
    pub fn diffusivity_at(&self, t: f64, p: f64) -> f64 {
        let tr = self.t_ref.max(1.0);
        if self.is_gas() {
            self.diffusivity * (t.max(50.0) / tr).powf(1.75) * (101325.0 / p.max(1e3))
        } else {
            self.diffusivity * (t.max(50.0) / tr) * (self.mu_ref / self.mu(t).max(1e-12))
        }
    }
    /// enthalpie molaire (J/mol)
    pub fn h_molar(&self, t: f64) -> f64 {
        self.molar_mass * self.h_mass(t)
    }
    /// entropie molaire standard (1 bar) s°(T) = s°(298.15) + M ∫ cp(T)/T dT  (J/(mol·K))
    pub fn s_molar(&self, t: f64) -> f64 {
        let t = t.max(1.0);
        let a = self.cp - self.cp_slope * T_STD;
        self.s_standard + self.molar_mass * (a * (t / T_STD).ln() + self.cp_slope * (t - T_STD))
    }
    /// enthalpie libre molaire standard g°(T) = h − T s° (J/mol)
    pub fn g_molar(&self, t: f64) -> f64 {
        self.h_molar(t) - t * self.s_molar(t)
    }
    pub fn rho(&self, p: f64, t: f64) -> f64 {
        match self.eos {
            Eos::IdealGas { r_spec } => p.max(1.0) / (r_spec * t.max(1.0)),
            Eos::Tait { p0, b, n, thermal } => thermal.rho_p0(t) * ((p + b).max(1.0) / (p0 + b)).powf(1.0 / n),
        }
    }
    /// Compressibilité ISOTHERME −(1/V) dV/dp = (1/ρ) dρ/dp|_T
    pub fn kappa_t(&self, p: f64, _t: f64) -> f64 {
        match self.eos {
            Eos::IdealGas { .. } => 1.0 / p.max(1.0),
            Eos::Tait { b, n, .. } => 1.0 / (n * (p + b).max(1.0)),
        }
    }
    /// Dilatation thermique β = −(1/ρ) dρ/dT|_p
    pub fn beta(&self, _p: f64, t: f64) -> f64 {
        match self.eos {
            Eos::IdealGas { .. } => 1.0 / t.max(1.0),
            Eos::Tait { thermal, .. } => thermal.beta(t),
        }
    }
    /// Vitesse du son isentropique² (diagnostic, pas de temps acoustique, tests)
    pub fn sound_speed2(&self, p: f64, t: f64) -> f64 {
        match self.eos {
            Eos::IdealGas { r_spec } => self.cp / (self.cp - r_spec) * r_spec * t,
            Eos::Tait { n, b, .. } => n * (p + b) / self.rho(p, t),
        }
    }
    pub fn mu(&self, t: f64) -> f64 {
        self.mu_ref * self.mu_law.factor(t as f32, self.t_ref as f32) as f64
    }
    pub fn k(&self, t: f64) -> f64 {
        self.conductivity * self.k_law.factor(t as f32, self.t_ref as f32) as f64
    }

    // ---------- bibliothèque ----------
    pub fn air() -> Phase {
        Phase {
            name: "air".into(),
            kind: PhaseKind::Gas,
            eos: Eos::IdealGas { r_spec: 287.05 },
            cp: 1006.0,
            conductivity: 0.0257,
            mu_ref: 1.81e-5,
            mu_law: PropertyLaw::Sutherland { s: 110.4 },
            k_law: PropertyLaw::Sutherland { s: 194.0 },
            t_ref: 293.15,
            e0: 0.0,
            sigma: 0.0,
            color: [0.05, 0.07, 0.11],
            molar_mass: 0.02897,
            diffusivity: 2.0e-5,
            h_formation: 0.0,
            s_standard: 0.0,
            cp_slope: 0.0,
            radiator: Radiator::None,
            grain_diameter: 0.0,
            alpha_max: 0.6,
            friction_angle: 0.0,
        }
    }
    pub fn water() -> Phase {
        Phase {
            name: "eau".into(),
            kind: PhaseKind::Liquid,
            eos: Eos::Tait { p0: 101325.0, b: 3.07e8, n: 7.15, thermal: Thermal::Kell },
            cp: 4186.0,
            conductivity: 0.598,
            mu_ref: 1.002e-3,
            mu_law: PropertyLaw::Vogel { b: 247.8, c: 140.0 },
            k_law: PropertyLaw::Linear { slope: 0.0035 },
            t_ref: 293.15,
            e0: 0.0,
            sigma: 0.0728,
            color: [0.10, 0.42, 0.88],
            molar_mass: 0.018015,
            diffusivity: 1.0e-9,
            h_formation: -285.83e3,
            s_standard: 69.95,
            cp_slope: 0.0,
            radiator: Radiator::None,
            grain_diameter: 0.0,
            alpha_max: 0.6,
            friction_angle: 0.0,
        }
    }
    /// Vapeur d'eau (gaz parfait, R = 461.5 J/(kg·K))
    pub fn steam() -> Phase {
        Phase {
            name: "vapeur".into(),
            kind: PhaseKind::Gas,
            eos: Eos::IdealGas { r_spec: 461.5 },
            cp: 1996.0,
            conductivity: 0.0248,
            mu_ref: 1.23e-5,
            mu_law: PropertyLaw::Sutherland { s: 961.0 },
            k_law: PropertyLaw::Constant,
            t_ref: 373.15,
            e0: 0.0,
            sigma: 0.0,
            color: [0.90, 0.93, 0.96],
            molar_mass: 0.018015,
            diffusivity: 2.5e-5,
            h_formation: -241.826e3,
            s_standard: 188.84,
            cp_slope: 0.0,
            radiator: Radiator::Water,
            grain_diameter: 0.0,
            alpha_max: 0.6,
            friction_angle: 0.0,
        }
    }
    /// Sable de quartz (grains de 0.3 mm) : phase granulaire. Grains incompressibles (Tait très raide, K ≈ 37 GPa), dilatation du
    /// quartz ; viscosité de mélange de l'ordre de celle de l'eau (la friction des tas viendra d'un modèle dédié, étape 3)
    pub fn sand() -> Phase {
        Phase {
            name: "sable".into(),
            kind: PhaseKind::Granular,
            eos: Eos::Tait { p0: 101325.0, b: 5.3e9, n: 7.0, thermal: Thermal::Beta { rho0: 2650.0, t0: 293.15, beta: 3.5e-5 } },
            cp: 830.0,
            conductivity: 1.4,
            mu_ref: 1.0e-3,
            mu_law: PropertyLaw::Constant,
            k_law: PropertyLaw::Constant,
            t_ref: 293.15,
            e0: 0.0,
            sigma: 0.0,
            color: [0.86, 0.74, 0.47],
            molar_mass: 0.06008,
            diffusivity: 0.0,
            h_formation: 0.0,
            s_standard: 0.0,
            cp_slope: 0.0,
            radiator: Radiator::None,
            grain_diameter: 3.0e-4,
            alpha_max: 0.6,
            // sable sec : angle de talus ~ 30–35°
            friction_angle: 32.0,
        }
    }
    /// Huile végétale (Tait avec n et B typiques des huiles, dilatation β constante)
    pub fn oil() -> Phase {
        Phase {
            name: "huile".into(),
            kind: PhaseKind::Liquid,
            eos: Eos::Tait {
                p0: 101325.0,
                b: 2.9e8,
                n: 9.0,
                thermal: Thermal::Beta { rho0: 915.0, t0: 293.15, beta: 7.0e-4 },
            },
            cp: 1970.0,
            conductivity: 0.17,
            mu_ref: 0.065,
            mu_law: PropertyLaw::Arrhenius { e_r: 3500.0 },
            k_law: PropertyLaw::Constant,
            t_ref: 293.15,
            e0: 0.0,
            sigma: 0.032,
            color: [0.85, 0.62, 0.10],
            molar_mass: 0.885,
            diffusivity: 1.0e-10,
            h_formation: 0.0,
            s_standard: 0.0,
            cp_slope: 0.0,
            radiator: Radiator::None,
            grain_diameter: 0.0,
            alpha_max: 0.6,
            friction_angle: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Solid {
    pub name: String,
    pub rho: f64,
    pub cp: f64,
    pub conductivity: f64,
    #[serde(default = "default_solid_color")]
    pub color: [f32; 3],
}

fn default_solid_color() -> [f32; 3] {
    [0.6, 0.6, 0.6]
}

impl Solid {
    pub fn steel() -> Solid {
        Solid { name: "acier".into(), rho: 7850.0, cp: 490.0, conductivity: 45.0, color: [0.55, 0.57, 0.60] }
    }
    pub fn glass() -> Solid {
        Solid { name: "verre".into(), rho: 2500.0, cp: 840.0, conductivity: 1.0, color: [0.62, 0.82, 0.80] }
    }
    pub fn rho_cp(&self) -> f64 {
        self.rho * self.cp
    }
}

/// Paire liquide ↔ vapeur (ébullition, condensation, cavitation)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseChange {
    pub liquid: usize,
    pub vapor: usize,
    pub l_ref: f64, // J/kg à t_ref (eau : 2.257e6 à 100 °C)
    pub t_ref: f64, // K
    pub p_ref: f64, // Pa (pression de saturation à t_ref)
    pub r_v: f64,
    pub dcp: f64, // cp_v − cp_l
}

impl PhaseChange {
    /// Chaleur latente L(T) (Kirchhoff)
    pub fn latent(&self, t: f64) -> f64 {
        self.l_ref + self.dcp * (t - self.t_ref)
    }
    /// ln(p_sat/p_ref) = (L_ref − Δc T_ref)/R_v (1/T_ref − 1/T) + (Δc/R_v) ln(T/T_ref)
    pub fn p_sat(&self, t: f64) -> f64 {
        let t = t.clamp(200.0, 640.0);
        let a = (self.l_ref - self.dcp * self.t_ref) / self.r_v;
        self.p_ref * (a * (1.0 / self.t_ref - 1.0 / t) + self.dcp / self.r_v * (t / self.t_ref).ln()).exp()
    }
    /// Température de saturation à la pression p (inversion par bissection, p_sat est strictement croissante)
    pub fn t_sat(&self, p: f64) -> f64 {
        let (mut lo, mut hi) = (200.0, 640.0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.p_sat(mid) < p {
                lo = mid
            } else {
                hi = mid
            }
        }
        0.5 * (lo + hi)
    }
}

/// Réaction chimique entre fluides du registre (moteur cinétique : `mp_species::react_step`).
///
/// Vitesse (mol de réaction par m³ de cellule et par s), loi d'Arrhenius modifiée et loi d'action de masse :
///   k(T) = A · T^b · exp(−Ea / (R T)),   r = k(T) · Π_réactifs c_i^{o_i} · Π_modulateurs c_j^{o_j} · (1 − Q/Kc si réversible)
/// avec c = m/M (mol/m³), o_i = ordre partiel (défaut : ν_i), Q = Π c_prod^ν / Π c_réac^ν, Kc(T) = exp(−ΔG°/(R T)) · (P°/(R T))^Δn
/// (ΔG° calculé avec les h_f°, s° et cp(T) des fluides : l'équilibre — dissociation de CO2 et H2O dans les flammes — est
/// thermodynamiquement cohérent). Masse : −ν_i M_i ξ pour les réactifs, +ν_j M_j ξ pour les produits (bilan exact imposé).
/// Énergie : l'énergie interne e de la cellule est INCHANGÉE ; les décalages e0 (fixés par les enthalpies de formation, voir
/// `Registry::apply_thermochemistry`) portent la chaleur de réaction, comme pour la chaleur latente.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reaction {
    pub name: String,
    /// (indice de phase, coefficient stœchiométrique ν > 0)
    pub reactants: Vec<(usize, f64)>,
    pub products: Vec<(usize, f64)>,
    /// facteur pré-exponentiel A (unités SI : (m³/mol)^(ordre global − 1) / s / K^b)
    pub pre_exp: f64,
    /// énergie d'activation Ea (J/mol)
    pub activation_energy: f64,
    /// ordre partiel par réactif (même longueur que `reactants` ; vide = ordre = ν)
    #[serde(default)]
    pub orders: Vec<f64>,
    /// exposant de température b de la loi d'Arrhenius modifiée
    #[serde(default)]
    pub temp_exponent: f64,
    /// réaction réversible (vitesse inverse déduite de l'équilibre thermodynamique)
    #[serde(default)]
    pub reversible: bool,
    /// espèces qui modulent la vitesse sans être consommées (ex. H2O dans l'oxydation de CO) : (indice, ordre)
    #[serde(default)]
    pub modifiers: Vec<(usize, f64)>,
    /// épaisseur de flamme laminaire δ_L (m) pour le modèle de flamme épaissie (TFM) ; 0 = pas de modèle de flamme
    #[serde(default)]
    pub flame_thickness: f64,
}

impl Reaction {
    /// bilan de masse stœchiométrique : (Σ ν M réactifs, Σ ν M produits) en kg
    pub fn mass_balance(&self, phases: &[Phase]) -> (f64, f64) {
        let sum = |v: &[(usize, f64)]| v.iter().filter_map(|&(k, nu)| phases.get(k).map(|p| nu * p.molar_mass)).sum::<f64>();
        (sum(&self.reactants), sum(&self.products))
    }
    /// énergie interne de réaction à 0 K de la convention e0 (J/mol, > 0 = endothermique) : Σ ν M e0 (produits) − Σ ν M e0 (réactifs)
    pub fn heat_of_reaction(&self, phases: &[Phase]) -> f64 {
        let sum = |v: &[(usize, f64)]| v.iter().filter_map(|&(k, nu)| phases.get(k).map(|p| nu * p.molar_mass * p.e0)).sum::<f64>();
        sum(&self.products) - sum(&self.reactants)
    }
    fn sum_over(&self, phases: &[Phase], f: impl Fn(&Phase) -> f64) -> f64 {
        let s = |v: &[(usize, f64)]| v.iter().filter_map(|&(k, nu)| phases.get(k).map(|p| nu * f(p))).sum::<f64>();
        s(&self.products) - s(&self.reactants)
    }
    /// enthalpie de réaction Δh_r(T) (J/mol, < 0 = exothermique)
    pub fn delta_h(&self, phases: &[Phase], t: f64) -> f64 {
        self.sum_over(phases, |p| p.h_molar(t))
    }
    /// énergie interne de réaction Δu_r(T) (J/mol) : chaleur dégagée à volume constant = −Δu_r
    pub fn delta_u(&self, phases: &[Phase], t: f64) -> f64 {
        self.sum_over(phases, |p| p.molar_mass * p.e_int(t))
    }
    /// Δn (mol) = Σ ν produits − Σ ν réactifs
    pub fn delta_n(&self) -> f64 {
        self.products.iter().map(|x| x.1).sum::<f64>() - self.reactants.iter().map(|x| x.1).sum::<f64>()
    }
    /// ln Kc(T), Kc en (mol/m³)^Δn : ln Kc = −ΔG°/(R T) + Δn ln(P°/(R T))
    pub fn ln_kc(&self, phases: &[Phase], t: f64) -> f64 {
        let dg = self.sum_over(phases, |p| p.g_molar(t));
        -dg / (R_GAS * t) + self.delta_n() * (P_STD / (R_GAS * t)).ln()
    }
    /// constante de vitesse directe k(T)
    pub fn k_forward(&self, t: f64) -> f64 {
        let t = t.max(1.0);
        self.pre_exp * if self.temp_exponent != 0.0 { t.powf(self.temp_exponent) } else { 1.0 } * (-self.activation_energy / (R_GAS * t)).exp()
    }
    /// ordre partiel du i-ème réactif
    pub fn order(&self, i: usize) -> f64 {
        self.orders.get(i).copied().unwrap_or(self.reactants[i].1)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    pub phases: Vec<Phase>,
    pub solids: Vec<Solid>,
    pub changes: Vec<PhaseChange>,
    /// paires (a < b) de phases MISCIBLES (se mélangent à l'échelle moléculaire : pas d'interface entre elles). La relation est
    /// utilisée par fermeture transitive : voir `miscible_groups`.
    pub miscible: Vec<(usize, usize)>,
    /// réactions chimiques (cinétique : `mp_species::react_step`)
    pub reactions: Vec<Reaction>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_phase(&mut self, p: Phase) -> usize {
        self.phases.push(p);
        self.phases.len() - 1
    }
    pub fn add_solid(&mut self, s: Solid) -> usize {
        self.solids.push(s);
        self.solids.len() - 1
    }
    /// Déclare que `liquid` et `vapor` sont les deux états d'une même substance.
    /// Fixe e0_vapeur (à partir du liquide) pour que h_v(T_ref) − h_l(T_ref) = L_ref ; à cp constants : e0_v − e0_l = L_ref − (cp_v − cp_l) T_ref
    /// ⇒ h_v − h_l = L(T) pour tout T.
    pub fn add_phase_change(&mut self, liquid: usize, vapor: usize, l_ref: f64, t_ref: f64, p_ref: f64) {
        self.add_phase_change_anchored(liquid, vapor, l_ref, t_ref, p_ref, false);
    }
    /// Comme `add_phase_change` ; `anchor_vapor` = true : la vapeur garde son e0 (fixé par sa thermochimie) et c'est celui du liquide qui
    /// est ajusté (cas d'une vapeur produite par une réaction, ex. H2O de combustion).
    pub fn add_phase_change_anchored(&mut self, liquid: usize, vapor: usize, l_ref: f64, t_ref: f64, p_ref: f64, anchor_vapor: bool) {
        let r_v = match self.phases[vapor].eos {
            Eos::IdealGas { r_spec } => r_spec,
            _ => panic!("la vapeur doit être un gaz parfait"),
        };
        let dcp = self.phases[vapor].cp - self.phases[liquid].cp;
        if anchor_vapor {
            let h_v = self.phases[vapor].h_mass(t_ref);
            let l = &mut self.phases[liquid];
            l.e0 = 0.0;
            l.e0 = h_v - l_ref - l.h_mass(t_ref);
        } else {
            let h_l = self.phases[liquid].h_mass(t_ref);
            let v = &mut self.phases[vapor];
            v.e0 = 0.0;
            v.e0 = h_l + l_ref - v.h_mass(t_ref);
        }
        self.changes.push(PhaseChange { liquid, vapor, l_ref, t_ref, p_ref, r_v, dcp });
    }
    /// Références d'énergie chimique : e0 = Δh_f°/M − cp·298.15  ⇒  h(298.15 K) = Δh_f°/M pour chaque fluide (gaz : h = e + rT ;
    /// liquide : h ≈ e). Appelé quand des réactions sont définies, AVANT les changements de phase.
    pub fn apply_thermochemistry(&mut self) {
        for p in self.phases.iter_mut() {
            p.e0 = 0.0;
            p.e0 = p.h_formation / p.molar_mass.max(1e-9) - p.h_mass(T_STD);
        }
    }
    /// chaleur latente h_v(T) − h_l(T) d'une paire (J/kg), cohérente avec les énergies internes du solveur (cp(T) compris)
    pub fn latent_heat(&self, pc: &PhaseChange, t: f64) -> f64 {
        self.phases[pc.vapor].h_mass(t) - self.phases[pc.liquid].h_mass(t)
    }
    /// fluide k impliqué dans au moins une réaction (réactif, produit ou modulateur) ?
    pub fn is_reactive(&self, k: usize) -> bool {
        self.reactions.iter().any(|r| r.reactants.iter().chain(r.products.iter()).chain(r.modifiers.iter()).any(|x| x.0 == k))
    }
    /// eau + vapeur d'eau + air (valeurs réelles à 100 °C / 1 atm). Retourne (registre, id_air, id_eau, id_vapeur).
    pub fn water_air_steam() -> (Registry, usize, usize, usize) {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let eau = r.add_phase(Phase::water());
        let vap = r.add_phase(Phase::steam());
        r.add_phase_change(eau, vap, 2.257e6, 373.15, 101325.0);
        (r, air, eau, vap)
    }
    pub fn change_for_liquid(&self, liquid: usize) -> Option<&PhaseChange> {
        self.changes.iter().find(|c| c.liquid == liquid)
    }

    // ---------------------------------------------------------------- miscibilité
    pub fn is_miscible(&self, a: usize, b: usize) -> bool {
        a != b && self.miscible.contains(&(a.min(b), a.max(b)))
    }
    pub fn set_miscible(&mut self, a: usize, b: usize, on: bool) {
        if a == b {
            return;
        }
        let pair = (a.min(b), a.max(b));
        self.miscible.retain(|&q| q != pair);
        if on {
            self.miscible.push(pair);
        }
    }
    /// Groupes de phases mutuellement miscibles (fermeture transitive des paires déclarées). Chaque phase appartient à exactement un
    /// groupe (les non-miscibles sont des singletons). Ordre : par plus petit indice de membre ; membres triés.
    pub fn miscible_groups(&self) -> Vec<Vec<usize>> {
        let n = self.phases.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut y = x;
            while p[y] != r {
                let nx = p[y];
                p[y] = r;
                y = nx;
            }
            r
        }
        for &(a, b) in &self.miscible {
            if a < n && b < n {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = vec![];
        let mut idx = vec![usize::MAX; n];
        for k in 0..n {
            let r = find(&mut parent, k);
            if idx[r] == usize::MAX {
                idx[r] = groups.len();
                groups.push(vec![]);
            }
            groups[idx[r]].push(k);
        }
        groups
    }

    /// Cohérence du registre : liste de messages (vide = OK). Ne modifie rien.
    pub fn validate(&self) -> Vec<String> {
        let mut w = vec![];
        let n = self.phases.len();
        if n == 0 {
            w.push("aucun fluide défini".to_string());
        }
        if n > MAX_PHASES {
            w.push(format!("{n} fluides : maximum {MAX_PHASES}"));
        }
        for (i, a) in self.phases.iter().enumerate() {
            if self.phases[..i].iter().any(|b| b.name == a.name) {
                w.push(format!("nom de fluide en double : « {} »", a.name));
            }
        }
        for (i, a) in self.solids.iter().enumerate() {
            if self.solids[..i].iter().any(|b| b.name == a.name) {
                w.push(format!("nom de solide en double : « {} »", a.name));
            }
        }
        for g in self.miscible_groups().iter().filter(|g| g.len() > 1) {
            let names: Vec<&str> = g.iter().map(|&k| self.phases[k].name.as_str()).collect();
            let gas = g.iter().filter(|&&k| self.phases[k].is_gas()).count();
            if gas != 0 && gas != g.len() {
                w.push(format!("groupe miscible {names:?} mélange gaz et liquides : un gaz n'est pas miscible à un liquide (utiliser le changement de phase / la dissolution)"));
            }
            for (x, &a) in g.iter().enumerate() {
                for &b in &g[x + 1..] {
                    if !self.is_miscible(a, b) {
                        w.push(format!("« {} » et « {} » sont déclarés miscibles indirectement (transitivité) : traités comme miscibles", self.phases[a].name, self.phases[b].name));
                    }
                }
            }
        }
        for c in &self.changes {
            if c.liquid >= n || c.vapor >= n {
                w.push("changement de phase : indice de fluide invalide".to_string());
            }
        }
        for r in &self.reactions {
            if r.reactants.is_empty() || r.products.is_empty() {
                w.push(format!("réaction « {} » : réactifs ou produits vides", r.name));
                continue;
            }
            if r.reactants.iter().chain(r.products.iter()).any(|&(k, nu)| k >= n || !(nu > 0.0)) {
                w.push(format!("réaction « {} » : fluide inexistant ou coefficient ≤ 0", r.name));
                continue;
            }
            if r.modifiers.iter().any(|&(k, _)| k >= n) {
                w.push(format!("réaction « {} » : modulateur de vitesse inexistant", r.name));
                continue;
            }
            let (m_in, m_out) = r.mass_balance(&self.phases);
            if (m_in - m_out).abs() > 1e-3 * m_in.max(m_out) {
                w.push(format!("réaction « {} » : masse non conservée ({:.4} kg → {:.4} kg par mole de réaction)", r.name, m_in, m_out));
            }
            if !r.orders.is_empty() && r.orders.len() != r.reactants.len() {
                w.push(format!("réaction « {} » : {} ordres pour {} réactifs (les ordres manquants valent ν)", r.name, r.orders.len(), r.reactants.len()));
            }
            if r.reversible && r.reactants.iter().chain(r.products.iter()).any(|&(k, _)| self.phases[k].s_standard == 0.0) {
                w.push(format!("réaction réversible « {} » : entropie standard s° non renseignée pour un des fluides (équilibre faux)", r.name));
            }
            // réactifs dans des groupes non miscibles : la réaction n'a lieu que dans les cellules d'interface (taux volumique)
            let groups = self.miscible_groups();
            let gid = |k: usize| groups.iter().position(|g| g.contains(&k)).unwrap_or(usize::MAX);
            let g0 = gid(r.reactants[0].0);
            if r.reactants.iter().any(|&(k, _)| gid(k) != g0) {
                w.push(format!("réaction « {} » : réactifs non miscibles entre eux — elle n'aura lieu que dans les cellules où ils se touchent", r.name));
            }
            if r.reactants.iter().chain(r.products.iter()).all(|&(k, _)| self.phases[k].h_formation == 0.0) {
                w.push(format!("réaction « {} » : enthalpies de formation toutes nulles — aucune chaleur de réaction", r.name));
            }
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kell_density_values() {
        let w = Phase::water();
        let p0 = 101325.0;
        for &(c, rho) in &[(0.0, 999.84), (4.0, 999.97), (20.0, 998.21), (60.0, 983.2), (100.0, 958.4)] {
            let r = w.rho(p0, 273.15 + c);
            assert!((r - rho).abs() < 0.3, "ρ({c}°C)={r} attendu {rho}");
        }
    }

    #[test]
    fn tait_sound_speed_and_compressibility() {
        let w = Phase::water();
        let c = w.sound_speed2(101325.0, 293.15).sqrt();
        assert!((c - 1482.0).abs() < 15.0, "c={c}");
        // module de compressibilité K = 1/κ ≈ 2.2 GPa
        let k = 1.0 / w.kappa_t(101325.0, 293.15);
        assert!((k - 2.2e9).abs() < 0.1e9, "K={k}");
        // +100 MPa → ρ augmente d'environ 4.5 %
        let r = w.rho(1.0e8, 293.15) / w.rho(101325.0, 293.15);
        assert!(r > 1.035 && r < 1.05, "ratio {r}");
        assert!((w.beta(101325.0, 293.15) - 2.07e-4).abs() < 0.2e-4);
    }

    #[test]
    fn ideal_gas() {
        let a = Phase::air();
        assert!((a.rho(101325.0, 293.15) - 1.204).abs() < 2e-3);
        let c = a.sound_speed2(101325.0, 293.15).sqrt();
        assert!((c - 343.0).abs() < 1.5, "c={c}");
    }

    #[test]
    fn saturation_curve_matches_steam_tables() {
        let (r, _, _, _) = Registry::water_air_steam();
        let pc = &r.changes[0];
        assert!((pc.p_sat(373.15) - 101325.0).abs() < 1.0);
        // tables : 20 °C 2339 Pa, 60 °C 19946 Pa, 150 °C 4.76e5 Pa
        for &(c, p) in &[(20.0, 2339.0), (60.0, 19946.0), (150.0, 4.76e5)] {
            let ps = pc.p_sat(273.15 + c);
            assert!((ps / p - 1.0).abs() < 0.05, "p_sat({c}°C)={ps} table {p}");
        }
        // inversion
        assert!((pc.t_sat(pc.p_sat(330.0)) - 330.0).abs() < 1e-6);
        // chaleur latente : 2.454e6 à 20 °C, 2.257e6 à 100 °C
        assert!((pc.latent(293.15) / 2.454e6 - 1.0).abs() < 0.02, "L20={}", pc.latent(293.15));
    }

    #[test]
    fn latent_heat_is_enthalpy_difference() {
        let (r, _, eau, vap) = Registry::water_air_steam();
        let pc = &r.changes[0];
        for &t in &[300.0, 373.15, 450.0] {
            let (l, v) = (&r.phases[eau], &r.phases[vap]);
            let rv = pc.r_v;
            let dh = (v.e_int(t) + rv * t) - l.e_int(t);
            assert!((dh - pc.latent(t)).abs() < 1.0, "T={t} dh={dh} L={}", pc.latent(t));
        }
    }
}
