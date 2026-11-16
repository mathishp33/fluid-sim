use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FluidKind {
    Gas,
    Liquid,
}

// Loi d'évolution d'une propriété avec la température, sous forme de FACTEUR sans dimension f(T) tel que
// f(T_ref) = 1 : propriété(T) = propriété_ref · f(T). La valeur de référence reste celle du matériau
// (`viscosity`, `conductivity`), l'UI et les benchmarks existants gardent donc leur sens à T_ref.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum PropertyLaw {
    #[default]
    Constant,
    // Sutherland (gaz) : f = (T/T_ref)^{3/2}·(T_ref + S)/(T + S). Air : S = 110,4 K (μ), 194 K (k).
    Sutherland { s: f32 },
    // Vogel / Andrade (liquides, μ) : f = 10^{ b·(1/(T − c) − 1/(T_ref − c)) }. Eau : b = 247,8 K, c = 140 K.
    Vogel { b: f32, c: f32 },
    // Arrhenius (liquides très visqueux) : f = exp(E_R·(1/T − 1/T_ref)), E_R = E/R en K.
    Arrhenius { e_r: f32 },
    // Linéaire relatif : f = 1 + pente·(T − T_ref)  (k de l'eau : ~ +0,35 %/K)
    Linear { slope: f32 },
}

impl PropertyLaw {
    // identifiant + 2 paramètres pour le shader (mêmes formules dans `law_factor` du WGSL)
    pub fn gpu_params(&self) -> (u32, f32, f32) {
        match *self {
            PropertyLaw::Constant => (0, 0.0, 0.0),
            PropertyLaw::Sutherland { s } => (1, s, 0.0),
            PropertyLaw::Vogel { b, c } => (2, b, c),
            PropertyLaw::Arrhenius { e_r } => (3, e_r, 0.0),
            PropertyLaw::Linear { slope } => (4, slope, 0.0),
        }
    }

    pub fn factor(&self, t: f32, t_ref: f32) -> f32 {
        let t = t.clamp(100.0, 4000.0);
        let f = match *self {
            PropertyLaw::Constant => 1.0,
            PropertyLaw::Sutherland { s } => {
                let r = t / t_ref;
                r * r.sqrt() * (t_ref + s) / (t + s)
            }
            PropertyLaw::Vogel { b, c } => {
                let (d, dr) = ((t - c).max(20.0), (t_ref - c).max(20.0));
                (std::f32::consts::LN_10 * b * (1.0 / d - 1.0 / dr)).exp()
            }
            PropertyLaw::Arrhenius { e_r } => (e_r * (1.0 / t - 1.0 / t_ref)).exp(),
            PropertyLaw::Linear { slope } => 1.0 + slope * (t - t_ref),
        };
        f.clamp(1e-2, 1e2) // garde-fou numérique, hors plage physique on sature
    }
}

// un seul matériau à la fois, tout en SI
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FluidMaterial {
    pub kind: FluidKind,
    pub rho_ref: f32, // kg/m³ à (t_ref, p_ref) : référence hydrostatique et de flottabilité
    pub t_ref: f32, // K
    pub p_ref: f32, // Pa
    pub viscosity: f32, // ν_ref, m²/s à (t_ref, p_ref). Viscosité dynamique de référence μ_ref = ν_ref·ρ_ambient.
    pub tracer_diffusion: f32, // D, m^2/s : diffusion du TRACEUR (colorant), pas de la chaleur
    pub conductivity: f32, // k_ref, W/(m·K) à t_ref
    pub cp: f32, // J/(kg·K)
    pub r_spec: f32, // J/(kg·K), gaz uniquement
    pub beta: f32, // 1/K, liquide uniquement
    // dépendance en température de μ et k (utilisée si `FluidSim::variable_properties`)
    #[serde(default)]
    pub viscosity_law: PropertyLaw,
    #[serde(default)]
    pub conductivity_law: PropertyLaw,
    // solide (un seul matériau solide pour l'instant)
    pub solid_rho: f32,
    pub solid_cp: f32,
    pub solid_conductivity: f32,
}

impl FluidMaterial {
    pub fn air() -> Self {
        Self { kind: FluidKind::Gas, rho_ref: 1.204, t_ref: 293.15, p_ref: 101_325.0,
            viscosity: 1.516e-5, tracer_diffusion: 2.0e-5, conductivity: 0.0257, cp: 1005.0,
            r_spec: 287.05, beta: 0.0,
            // Sutherland : μ(293 K) = 1,81e-5 Pa·s, k(293 K) = 0,0257 W/(m·K) ; μ(600 K) ≈ 3,0e-5, k(600 K) ≈ 0,046
            viscosity_law: PropertyLaw::Sutherland { s: 110.4 },
            conductivity_law: PropertyLaw::Sutherland { s: 194.0 },
            solid_rho: 2400.0, solid_cp: 880.0, solid_conductivity: 1.0 }
    }
    pub fn water() -> Self {
        Self { kind: FluidKind::Liquid, rho_ref: 998.0, t_ref: 293.15, p_ref: 101_325.0,
            viscosity: 1.0e-6, tracer_diffusion: 1.0e-9, conductivity: 0.6, cp: 4182.0,
            r_spec: 0.0, beta: 2.07e-4,
            // μ(20 °C) = 1,00e-3, μ(40 °C) = 0,65e-3, μ(80 °C) = 0,35e-3 Pa·s ; k : 0,598 -> 0,640 W/(m·K) de 20 à 40 °C
            viscosity_law: PropertyLaw::Vogel { b: 247.8, c: 140.0 },
            conductivity_law: PropertyLaw::Linear { slope: 0.0035 },
            solid_rho: 2400.0, solid_cp: 880.0, solid_conductivity: 1.0 }
    }

    pub fn honey() -> Self {
        Self { kind: FluidKind::Liquid, rho_ref: 1420.0, t_ref: 293.15, p_ref: 101_325.0,
            viscosity: 0.099276, tracer_diffusion: 1.0e-9, conductivity: 0.6, cp: 4182.0,
            r_spec: 0.0, beta: 2.07e-4,
            // ordre de grandeur (le miel varie beaucoup selon l'humidité) : μ divisée par ~7 entre 20 et 40 °C
            viscosity_law: PropertyLaw::Arrhenius { e_r: 8700.0 },
            conductivity_law: PropertyLaw::Linear { slope: 0.0035 },
            solid_rho: 2400.0, solid_cp: 880.0, solid_conductivity: 1.0 }
    }

    // rho (kg/m^3) en fonction de T (K) et de la pression thermodynamique P_th (Pa).
    // Gaz : gaz parfait rho = P_th/(r*T)
    // Liquide : dilatation linéaire, indépendant de P
    pub fn density(&self, t: f32, p_th: f32) -> f32 {
        match self.kind {
            FluidKind::Gas => p_th / (self.r_spec * t.max(1.0)),
            FluidKind::Liquid => (self.rho_ref * (1.0 - self.beta * (t - self.t_ref)))
                .max(0.5 * self.rho_ref), // pas d'ébullition modélisée
        }
    }

    // Viscosité dynamique μ(T) en Pa·s : μ_ref·f(T), avec μ_ref = ν_ref·ρ_ambient.
    // `variable = false` : μ constante (= comportement ν constante quand ρ est lui aussi pris à ρ_ambient).
    #[inline]
    pub fn mu(&self, t: f32, variable: bool) -> f32 {
        let mu_ref = self.viscosity * self.rho_ambient();
        if variable { mu_ref * self.viscosity_law.factor(t, self.t_ref) } else { mu_ref }
    }

    // Conductivité thermique du fluide k(T), W/(m·K)
    #[inline]
    pub fn k_fluid(&self, t: f32, variable: bool) -> f32 {
        if variable { self.conductivity * self.conductivity_law.factor(t, self.t_ref) } else { self.conductivity }
    }

    // Gaz : p_ref/(r·T_ref), pour que ρ(T_ref) = ρ_ambient exactement (sinon une force parasite g·1e-4 apparaît).
    #[inline]
    pub fn rho_ambient(&self) -> f32 {
        match self.kind {
            FluidKind::Gas => self.p_ref / (self.r_spec * self.t_ref.max(1.0)),
            FluidKind::Liquid => self.rho_ref,
        }
    }

    // gamma = cp/cv
    pub fn gamma(&self) -> f32 {
        self.cp / (self.cp - self.r_spec).max(1.0)
    }
    // Boussinesq (liquide, ou gaz sur GPU) : (rho_ref − rho)/rho_ref
    #[inline]
    pub fn buoyancy_factor(&self, t: f32) -> f32 {
        match self.kind {
            FluidKind::Gas => (t - self.t_ref) / t.max(1.0), // 1 − T_ref/T
            FluidKind::Liquid => self.beta * (t - self.t_ref),
        }
    }

    // Low-Mach (momentum complet) : accélération vers le haut = g·(rho_ref/rho − 1)
    #[inline]
    pub fn buoyancy_from_rho(&self, rho: f32) -> f32 {
        self.rho_ambient() / rho.max(1e-6) - 1.0
    }
    pub fn heat_capacity_ref(&self) -> f32 {
        self.rho_ambient() * self.cp
    }
    pub fn heat_capacity_solid(&self) -> f32 {
        self.solid_rho * self.solid_cp
    }
    pub fn alpha_fluid(&self) -> f32 {
        self.conductivity / (self.rho_ambient() * self.cp)
    }
    pub fn alpha_solid(&self) -> f32 {
        self.solid_conductivity / (self.solid_rho * self.solid_cp)
    }
}