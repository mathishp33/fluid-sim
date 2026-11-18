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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseKind {
    Gas,
    Liquid,
}

/// Masse volumique d'un liquide à la pression de référence p0, en fonction de T.
#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub enum Eos {
    IdealGas { r_spec: f64 },
    Tait { p0: f64, b: f64, n: f64, thermal: Thermal },
}

#[derive(Clone, Debug)]
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
}

impl Phase {
    pub fn is_gas(&self) -> bool {
        self.kind == PhaseKind::Gas
    }
    /// Chaleur massique pour l'énergie interne e = c·T + e0 (cv pour un gaz ; cp ≈ cv pour un liquide, écart < 1 % pour l'eau)
    pub fn c_e(&self) -> f64 {
        match self.eos {
            Eos::IdealGas { r_spec } => self.cp - r_spec,
            Eos::Tait { .. } => self.cp,
        }
    }
    pub fn e_int(&self, t: f64) -> f64 {
        self.c_e() * t + self.e0
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
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solid {
    pub name: String,
    pub rho: f64,
    pub cp: f64,
    pub conductivity: f64,
}

impl Solid {
    pub fn steel() -> Solid {
        Solid { name: "acier".into(), rho: 7850.0, cp: 490.0, conductivity: 45.0 }
    }
    pub fn glass() -> Solid {
        Solid { name: "verre".into(), rho: 2500.0, cp: 840.0, conductivity: 1.0 }
    }
    pub fn rho_cp(&self) -> f64 {
        self.rho * self.cp
    }
}

/// Paire liquide ↔ vapeur (ébullition, condensation, cavitation)
#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Default)]
pub struct Registry {
    pub phases: Vec<Phase>,
    pub solids: Vec<Solid>,
    pub changes: Vec<PhaseChange>,
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
    /// Fixe e0_vapeur − e0_liquide = L_ref − (cp_v − cp_l) T_ref  ⇒  h_v − h_l = L(T) pour tout T.
    pub fn add_phase_change(&mut self, liquid: usize, vapor: usize, l_ref: f64, t_ref: f64, p_ref: f64) {
        let r_v = match self.phases[vapor].eos {
            Eos::IdealGas { r_spec } => r_spec,
            _ => panic!("la vapeur doit être un gaz parfait"),
        };
        let dcp = self.phases[vapor].cp - self.phases[liquid].cp;
        self.phases[vapor].e0 = self.phases[liquid].e0 + l_ref - dcp * t_ref;
        self.changes.push(PhaseChange { liquid, vapor, l_ref, t_ref, p_ref, r_v, dcp });
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
