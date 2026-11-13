use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FluidKind {
    Gas,
    Liquid,
}

// un seul matériau à la fois, tout en SI
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FluidMaterial {
    pub kind: FluidKind,
    pub rho_ref: f32, // kg/m³ à (t_ref, p_ref) : référence hydrostatique et de flottabilité
    pub t_ref: f32, // K
    pub p_ref: f32, // Pa
    pub viscosity: f32, // ν, m^2/s (cinématique, supposée constante)
    pub tracer_diffusion: f32, // D, m^2/s : diffusion du TRACEUR (colorant), pas de la chaleur
    pub conductivity: f32, // k, W/(m·K)
    pub cp: f32, // J/(kg·K)
    pub r_spec: f32, // J/(kg·K), gaz uniquement
    pub beta: f32, // 1/K, liquide uniquement
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
            solid_rho: 2400.0, solid_cp: 880.0, solid_conductivity: 1.0 }
    }
    pub fn water() -> Self {
        Self { kind: FluidKind::Liquid, rho_ref: 998.0, t_ref: 293.15, p_ref: 101_325.0,
            viscosity: 1.0e-6, tracer_diffusion: 1.0e-9, conductivity: 0.6, cp: 4182.0,
            r_spec: 0.0, beta: 2.07e-4,
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
    
    pub fn heat_capacity_ref(&self) -> f32 { self.rho_ambient() * self.cp }
    pub fn heat_capacity_solid(&self) -> f32 { self.solid_rho * self.solid_cp }
    
    pub fn alpha_fluid(&self) -> f32 { self.conductivity / (self.rho_ambient() * self.cp) }
    pub fn alpha_solid(&self) -> f32 { self.solid_conductivity / (self.solid_rho * self.solid_cp) }
}