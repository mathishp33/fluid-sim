use std::cmp::PartialEq;
use std::fmt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::simulation::gpu::gpu::{GpuContext, GpuFields, StepConfig};
use crate::simulation::materials::{FluidKind, FluidMaterial};

// Vocabulaire (à ne pas confondre) :
//   tracer       : colorant passif, sans dimension, 0..1. Il ne pèse rien et n'influence jamais la physique.
//   rho          : masse volumique réelle (kg/m³), DÉDUITE de T et P_th. Jamais advectée.
//   temperature  : K, absolue.
//   pressure     : pression du SOLVEUR (interne, dépend de dt). Ne pas l'afficher en Pa.
//   pressure_pa  : pression réelle (Pa) = P_ref + hydrostatique + dynamique.
//   p_th         : pression thermodynamique uniforme (Pa), utilisée par le gaz parfait.

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ComputeMethod {
    CPU,
    GPU,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryType {
    Wall,
    Inlet,
    Outlet,
}

/// Condition aux limites, en SI. `offset` est en cellules (rampe depuis les coins).
/// Les anciens config.json (vitesses en cellules/s) se chargent sans les vitesses : elles valent 0.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Boundary {
    pub boundary_type: BoundaryType,
    pub velocity_x_ms: f32, // m/s
    pub velocity_y_ms: f32, // m/s
    #[serde(alias = "density")]
    pub tracer: f32, // 0..1
    pub offset: usize, // cellules
    pub temperature: f32, // K
}

impl Default for Boundary {
    fn default() -> Self {
        Self::wall()
    }
}

impl Boundary {
    pub fn wall() -> Self {
        Self {
            boundary_type: BoundaryType::Wall,
            velocity_x_ms: 0.0,
            velocity_y_ms: 0.0,
            tracer: 0.0,
            offset: 1,
            temperature: 293.15,
        }
    }

    /// Même CL en unités du solveur (vitesses en cellules/s).
    pub fn to_grid(&self, dx: f32) -> GridBoundary {
        GridBoundary {
            boundary_type: self.boundary_type,
            vx: self.velocity_x_ms / dx,
            vy: self.velocity_y_ms / dx,
            tracer: self.tracer,
            offset: self.offset,
            temperature: self.temperature,
        }
    }
}

/// Boundary en unités du solveur (cellules/s) : ce que lisent cpu.rs et le GPU.
#[derive(Clone, Copy, Debug)]
pub struct GridBoundary {
    pub boundary_type: BoundaryType,
    pub vx: f32,
    pub vy: f32,
    pub tracer: f32,
    pub offset: usize,
    pub temperature: f32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    Fluid = 0,
    Wall = 1,
    Sink = 2,
    Emitter = 3,
    Accelerator = 4,
    Teleporter = 5,
}

impl CellType {
    pub const PAINTABLE: [Self; 5] = [Self::Wall, Self::Sink, Self::Emitter, Self::Accelerator, Self::Teleporter];

    #[inline]
    pub fn is_special(self) -> bool {
        self as u8 >= 2
    }
}

impl fmt::Display for CellType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CellType::Fluid => "Fluid",
            CellType::Wall => "Wall",
            CellType::Sink => "Sink",
            CellType::Emitter => "Emitter",
            CellType::Accelerator => "Accelerator",
            CellType::Teleporter => "Teleporter",
        })
    }
}

/// Effets de cellule, en SI : vitesses en m/s, accélérations en m/s².
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CellEffect {
    Sink { rate: f32 },
    Emitter { vx: f32, vy: f32, #[serde(alias = "density")] tracer: f32 },
    Accelerator { ax: f32, ay: f32 },
    Teleporter { dest_idx: usize },
}

impl CellEffect {
    pub fn default_for(kind: CellType) -> Option<Self> {
        match kind {
            CellType::Sink => Some(Self::Sink { rate: 0.05 }), // m/s
            CellType::Emitter => Some(Self::Emitter { vx: 0.0, vy: 0.0, tracer: 1.0 }),
            CellType::Accelerator => Some(Self::Accelerator { ax: 0.5, ay: 0.0 }), // m/s²
            CellType::Teleporter => None, // dest_idx dépend de la cellule, voir set_cell
            _ => None,
        }
    }

    /// Même effet en unités du solveur (cellules/s, cellules/s²).
    pub fn to_grid(self, dx: f32) -> Self {
        match self {
            Self::Sink { rate } => Self::Sink { rate: rate / dx },
            Self::Emitter { vx, vy, tracer } => Self::Emitter { vx: vx / dx, vy: vy / dx, tracer },
            Self::Accelerator { ax, ay } => Self::Accelerator { ax: ax / dx, ay: ay / dx },
            t @ Self::Teleporter { .. } => t,
        }
    }
}

/// Comportement thermique d'une cellule SOLIDE (Wall).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThermalCell {
    Isothermal(f32), // température imposée (K)
    Heater(f32),     // puissance volumique (W/m³)
}

// Les cellules en bordure (x = 0, x = w-1, y = 0, y = h-1) sont des cellules fantômes.
//
// Unités : le SOLVEUR travaille en unités de grille (vitesses en cellules/s, ν et α divisés par dx²).
// Les interfaces (matériau, rho, T, pressure_pa) sont en SI.
pub struct FluidSim {
    pub width: usize,
    pub height: usize,

    pub material: FluidMaterial,
    pub dx: f32, // m/cellule
    pub gravity: f32, // m/s²
    pub p_th: f32, // pression thermodynamique (Pa)
    pub dpdt: f32, // dP_th/dt (Pa/s), dernier pas (boîte fermée)
    pub last_dt: f32, // dernier dt utilisé (s), pour convertir la pression du solveur en Pa
    pub rho_min: f32, // kg/m³, cellules non solides

    // Gaz : true = low-Mach (ρ variable, divergence ≠ 0), CPU et GPU.
    // false (ou liquide) = incompressible + Boussinesq.
    pub low_mach: bool,
    // Parois : true = adhérence (no-slip, physique) ; false = glissement libre.
    pub no_slip: bool,

    pub tracer: Vec<f32>,
    pub rho: Vec<f32>, // kg/m³, dérivé de T
    pub pressure: Vec<f32>, // pression du SOLVEUR (interne)
    pub pressure_pa: Vec<f32>, // pression réelle (Pa)
    pub temperature: Vec<f32>, // K
    pub heat_source: Vec<f32>, // W/m³, consommé puis remis à zéro à chaque pas (pinceau, chauffages)
    // cellules solides thermiques : 0 = passive, 1 = isotherme (valeur = K), 2 = chauffage (valeur = W/m³)
    pub thermal_kind: Vec<u8>,
    pub thermal_value: Vec<f32>,
    pub thermal_count: usize,
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub divergence: Vec<f32>, // divergence réelle après projection (1/s)
    pub source: Vec<f32>, // divergence cible S (1/s) ; 0 si incompressible
    pub solids: Vec<CellType>,
    pub special_cells: Vec<(usize, CellEffect)>,
    // Facteur de sur-relaxation du Red-Black Gauss-Seidel (1.0 = Gauss-Seidel pur).
    pub sor_omega: f32,

    pub(crate) tracer_temp: Vec<f32>,
    pub(crate) velocity_x_temp: Vec<f32>,
    pub(crate) velocity_y_temp: Vec<f32>,
    pub(crate) temperature_temp: Vec<f32>,

    // 1 si la face droite (resp. basse) de la cellule i laisse passer le fluide
    pub(crate) open_u: Vec<u8>,
    pub(crate) open_v: Vec<u8>,
    // conductance de la face pour le solveur de pression : open * (ρ_ref/ρ_face)
    // (= open si incompressible)
    pub(crate) cond_u: Vec<f32>,
    pub(crate) cond_v: Vec<f32>,

    pub left_boundary: Boundary,
    pub right_boundary: Boundary,
    pub top_boundary: Boundary,
    pub bottom_boundary: Boundary,

    compute_method: ComputeMethod,
    gpu_context: Option<GpuContext>,
}

impl FluidSim {
    #[inline]
    pub(crate) fn idx(&self, x: usize, y: usize) -> usize {
        x + y * self.width
    }

    pub fn new(width: usize, height: usize, dx: f32, fm: FluidMaterial, g: f32) -> Self {
        let size = width * height;

        let gpu_context = GpuContext::try_new(width, height);
        let compute_method = if gpu_context.is_some() {
            ComputeMethod::GPU
        } else {
            eprintln!("No GPU found: falling back to CPU");
            ComputeMethod::CPU
        };

        FluidSim {
            width,
            height,

            material: fm,
            dx,
            gravity: g,
            p_th: fm.p_ref,
            dpdt: 0.0,
            last_dt: 0.016,
            rho_min: fm.rho_ambient(),
            low_mach: fm.kind == FluidKind::Gas,
            no_slip: true,

            tracer: vec![0.0; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            pressure: vec![0.0; size],
            rho: vec![fm.rho_ambient(); size],
            pressure_pa: vec![fm.p_ref; size],
            divergence: vec![0.0; size],
            source: vec![0.0; size],
            temperature: vec![fm.t_ref; size],
            heat_source: vec![0.0; size],
            thermal_kind: vec![0; size],
            thermal_value: vec![0.0; size],
            thermal_count: 0,

            solids: vec![CellType::Fluid; size],
            special_cells: Vec::new(),
            sor_omega: 1.5,

            tracer_temp: vec![0.0; size],
            velocity_x_temp: vec![0.0; size],
            velocity_y_temp: vec![0.0; size],
            temperature_temp: vec![0.0; size],

            open_u: vec![0; size],
            open_v: vec![0; size],
            cond_u: vec![0.0; size],
            cond_v: vec![0.0; size],

            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),

            compute_method,
            gpu_context,
        }
    }

    // conversions SI -> unités de grille
    #[inline] fn inv_dx2(&self) -> f32 { 1.0 / (self.dx * self.dx) }
    pub(crate) fn nu_grid(&self) -> f32 { self.material.viscosity * self.inv_dx2() }
    pub(crate) fn tracer_diff_grid(&self) -> f32 { self.material.tracer_diffusion * self.inv_dx2() }
    pub(crate) fn gravity_grid(&self) -> f32 { self.gravity / self.dx }

    /// Le mode low-Mach n'existe que pour un gaz.
    pub fn low_mach_active(&self) -> bool {
        self.low_mach && self.material.kind == FluidKind::Gas
    }

    pub fn has_outlet(&self) -> bool {
        [self.left_boundary, self.right_boundary, self.top_boundary, self.bottom_boundary]
            .iter()
            .any(|b| b.boundary_type == BoundaryType::Outlet)
    }

    pub fn has_sink(&self) -> bool {
        self.special_cells.iter().any(|(_, e)| matches!(e, CellEffect::Sink { .. }))
    }

    /// Boîte fermée : ni Outlet ni Sink. P_th devient alors une inconnue (dP_th/dt).
    pub fn is_closed(&self) -> bool {
        !self.has_outlet() && !self.has_sink()
    }

    /// ρ (kg/m³) depuis T et P_th. Met aussi à jour rho_min (cellules non solides).
    pub fn update_rho(&mut self) {
        let m = self.material;
        let p_th = self.p_th;
        self.rho.par_iter_mut().zip(self.temperature.par_iter())
            .for_each(|(r, &t)| *r = m.density(t, p_th));

        self.rho_min = self.rho.par_iter().zip(self.solids.par_iter())
            .filter(|&(_, &s)| s != CellType::Wall)
            .map(|(&r, _)| r)
            .reduce(|| f32::INFINITY, f32::min);
        if !self.rho_min.is_finite() {
            self.rho_min = m.rho_ambient();
        }
    }

    /// Pression réelle en Pa : P_th + hydrostatique + dynamique (solveur * ρ_ref·dx²/dt).
    /// y vers le bas ; référence = bord haut.
    pub fn update_pressure_pa(&mut self, dt: f32) {
        let m = self.material;
        let (w, g, dx) = (self.width, self.gravity, self.dx);
        let base = if m.kind == FluidKind::Gas { self.p_th } else { m.p_ref };
        self.last_dt = dt;
        let k = self.pressure_scale();

        self.pressure_pa.par_iter_mut().zip(self.pressure.par_iter()).enumerate()
            .for_each(|(i, (pa, &ps))| {
                let depth = (i / w) as f32 * dx;
                *pa = base + m.rho_ambient() * g * depth + k * ps;
            });
    }

    /// Facteur solveur -> Pa : P_dyn = pressure * ρ_ref·dx²/dt
    pub fn pressure_scale(&self) -> f32 {
        self.material.rho_ambient() * self.dx * self.dx / self.last_dt.max(1e-6)
    }

    pub fn update_derived(&mut self, dt: f32) {
        self.update_rho();
        self.update_pressure_pa(dt);
    }

    pub fn compute_method(&self) -> ComputeMethod {
        self.compute_method
    }

    pub fn set_compute_method(&mut self, method: ComputeMethod) {
        self.compute_method = match method {
            ComputeMethod::GPU if self.gpu_context.is_none() => ComputeMethod::CPU,
            m => m,
        };
    }

    pub fn set_cell(&mut self, x: usize, y: usize, kind: CellType, effect: Option<CellEffect>) {
        if x == 0 || x >= self.width - 1 || y == 0 || y >= self.height - 1 {
            return;
        }
        let idx = self.idx(x, y);
        let old = self.solids[idx];

        if old == kind && !kind.is_special() {
            return;
        }

        if old.is_special() {
            if let Some(p) = self.special_cells.iter().position(|(i, _)| *i == idx) {
                self.special_cells.swap_remove(p);
            }
        }

        self.solids[idx] = kind;
        if kind != CellType::Wall {
            self.clear_thermal(idx);
        }

        if kind.is_special() {
            let eff = effect
                .or_else(|| CellEffect::default_for(kind))
                .unwrap_or(CellEffect::Teleporter { dest_idx: idx }); // téléporteur sans destination = no-op
            self.special_cells.push((idx, eff));
        }

        if kind == CellType::Wall {
            self.tracer[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
            self.divergence[idx] = 0.0;
            self.source[idx] = 0.0;

            self.tracer_temp[idx] = 0.0;
            self.velocity_x_temp[idx] = 0.0;
            self.velocity_y_temp[idx] = 0.0;
            return;
        }

        if old != CellType::Wall {
            return;
        }

        // une cellule solide redevient fluide : on prend la moyenne des voisins fluides
        let neighbors = [
            self.idx(x - 1, y),
            self.idx(x + 1, y),
            self.idx(x, y - 1),
            self.idx(x, y + 1),
        ];

        let mut tracer_sum = 0.0;
        let mut velocity_x_sum = 0.0;
        let mut velocity_y_sum = 0.0;
        let mut pressure_sum = 0.0;
        let mut count = 0.0;

        for neighbor in neighbors {
            if self.solids[neighbor] == CellType::Wall {
                continue;
            }
            tracer_sum += self.tracer[neighbor];
            velocity_x_sum += self.velocity_x[neighbor];
            velocity_y_sum += self.velocity_y[neighbor];
            pressure_sum += self.pressure[neighbor];
            count += 1.0;
        }

        if count > 0.0 {
            self.tracer[idx] = tracer_sum / count;
            self.velocity_x[idx] = velocity_x_sum / count;
            self.velocity_y[idx] = velocity_y_sum / count;
            self.pressure[idx] = pressure_sum / count;
        } else {
            self.tracer[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
        }

        self.divergence[idx] = 0.0;
        self.source[idx] = 0.0;
        self.tracer_temp[idx] = self.tracer[idx];
        self.velocity_x_temp[idx] = self.velocity_x[idx];
        self.velocity_y_temp[idx] = self.velocity_y[idx];
    }

    #[inline]
    pub fn cell_velocity_from(vx: &[f32], vy: &[f32], w: usize, h: usize, x: usize, y: usize) -> (f32, f32) {
        let i = x + y * w;
        let ux = if x == 0 {
            vx[i]
        } else if x == w - 1 {
            vx[i - 1]
        } else {
            0.5 * (vx[i - 1] + vx[i])
        };
        let uy = if y == 0 {
            vy[i]
        } else if y == h - 1 {
            vy[i - w]
        } else {
            0.5 * (vy[i - w] + vy[i])
        };
        (ux, uy)
    }

    pub fn cell_velocity(&self, x: usize, y: usize) -> (f32, f32) {
        Self::cell_velocity_from(&self.velocity_x, &self.velocity_y, self.width, self.height, x, y)
    }

    /// Somme du tracer (colorant) sur les cellules intérieures. Sans dimension.
    pub fn total_tracer(&self) -> f64 {
        let w = self.width;
        (1..self.height - 1)
            .into_par_iter()
            .map(|y| self.tracer[1 + y * w..w - 1 + y * w].iter().map(|&d| d as f64).sum::<f64>())
            .sum()
    }

    /// Masse du fluide en kg par mètre de profondeur (simulation 2D) : Σ ρ·dx² sur les cellules non solides.
    pub fn total_mass(&self) -> f64 {
        let w = self.width;
        let dx2 = (self.dx * self.dx) as f64;
        (1..self.height - 1)
            .into_par_iter()
            .map(|y| {
                (1..w - 1)
                    .map(|x| {
                        let i = x + y * w;
                        if self.solids[i] == CellType::Wall { 0.0 } else { self.rho[i] as f64 }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>() * dx2
    }

    /// Pinceau de chauffage : source de chaleur (W/m³) qui tire T vers `target` avec une constante de temps ~ 1/rate.
    /// La chaleur ajoutée passe par l'équation d'énergie (donc dilatation en low-Mach), pas par un forçage de T.
    pub fn heat_towards(&mut self, idx: usize, target: f32, rate: f32) {
        let c = if self.solids[idx] == CellType::Wall {
            self.material.heat_capacity_solid()
        } else {
            self.rho[idx] * self.material.cp
        };
        self.heat_source[idx] = c * rate * (target - self.temperature[idx]);
    }

    pub fn thermal_at(&self, idx: usize) -> Option<ThermalCell> {
        match self.thermal_kind[idx] {
            1 => Some(ThermalCell::Isothermal(self.thermal_value[idx])),
            2 => Some(ThermalCell::Heater(self.thermal_value[idx])),
            _ => None,
        }
    }

    fn clear_thermal(&mut self, idx: usize) {
        if self.thermal_kind[idx] != 0 {
            self.thermal_kind[idx] = 0;
            self.thermal_value[idx] = 0.0;
            self.thermal_count -= 1;
        }
    }

    /// Donne (ou retire, avec None) un comportement thermique à une cellule SOLIDE. Sans effet sur un fluide.
    pub fn set_thermal(&mut self, x: usize, y: usize, effect: Option<ThermalCell>) {
        if x == 0 || x >= self.width - 1 || y == 0 || y >= self.height - 1 {
            return;
        }
        let idx = self.idx(x, y);
        if self.solids[idx] != CellType::Wall {
            return;
        }
        self.clear_thermal(idx);
        match effect {
            Some(ThermalCell::Isothermal(t)) => {
                self.thermal_kind[idx] = 1;
                self.thermal_value[idx] = t;
                self.thermal_count += 1;
                self.temperature[idx] = t;
            }
            Some(ThermalCell::Heater(p)) => {
                self.thermal_kind[idx] = 2;
                self.thermal_value[idx] = p;
                self.thermal_count += 1;
            }
            None => {}
        }
    }

    // Avant le pas : T imposée aux isothermes, puissance des chauffages ajoutée à la source de chaleur.
    fn apply_thermal_cells(&mut self) {
        if self.thermal_count == 0 {
            return;
        }
        for i in 0..self.thermal_kind.len() {
            match self.thermal_kind[i] {
                1 => self.temperature[i] = self.thermal_value[i],
                2 => self.heat_source[i] += self.thermal_value[i],
                _ => {}
            }
        }
    }

    // Après le pas : les isothermes reprennent leur température (l'énergie échangée vient d'un réservoir infini).
    fn reset_isothermal(&mut self) {
        if self.thermal_count == 0 {
            return;
        }
        for i in 0..self.thermal_kind.len() {
            if self.thermal_kind[i] == 1 {
                self.temperature[i] = self.thermal_value[i];
            }
        }
    }

    pub fn step(&mut self, dt: f32, pressure_iterations: usize) {
        self.apply_thermal_cells();
        if self.compute_method == ComputeMethod::GPU {
            if let Some(mut gpu) = self.gpu_context.take() {
                self.update_rho(); // ρ(T, P_th) : rho_min pour les sous-pas
                let p_th_old = self.p_th; // le GPU projette avec ρ(P_th du début de pas), comme le CPU

                // Low-Mach : S et dP_th/dt sont calculés ici (CPU, O(n)) puis envoyés au GPU.
                // Boîte fermée : le flux de bord est déduit des CL (les vitesses n'existent pas encore).
                let q_out = if self.low_mach_active() && self.is_closed() {
                    self.boundary_outflow_analytic(dt)
                } else {
                    0.0
                };
                self.compute_source(dt, Some(q_out));

                self.step_gpu(&mut gpu, dt, pressure_iterations, p_th_old);
                self.gpu_context = Some(gpu);

                if self.is_closed() {
                    self.remove_pressure_mean();
                }
                self.heat_source.fill(0.0); // consommée par le shader
                self.update_pressure_pa(dt);
                self.reset_isothermal();
                return;
            }
        }
        self.step_cpu(dt, pressure_iterations);
        self.reset_isothermal();
    }

    // Pas complet sur GPU, en deux phases (une seule synchronisation par phase) :
    //  1. vitesses : advection, flottabilité, cellules spéciales, CL, limitation, projection (div u = S) ;
    //  2. scalaires : sous-pas tracer + température, cellules spéciales, fantômes.
    // Même algorithme que step_cpu. Différence connue : la capacité thermique ρ·cp du gaz low-Mach
    // utilise le T courant du sous-pas (le CPU utilise le ρ du début de pas).
    fn step_gpu(&mut self, gpu: &mut GpuContext, dt: f32, pressure_iterations: usize, p_th: f32) {
        let m = self.material;
        let dx = self.dx;
        let specials: Vec<(usize, CellEffect)> = self.special_cells.iter().map(|&(i, e)| (i, e.to_grid(dx))).collect();
        let cfg = StepConfig {
            dt, pressure_iterations, omega: self.sor_omega,
            tracer_diffusion: self.tracer_diff_grid(),
            vmax: self.velocity_limit(dt),
            viscosity: self.nu_grid(),
            viscosity_substeps: self.viscosity_substeps(dt),
            gravity: self.gravity_grid(),
            t_ref: m.t_ref,
            beta: m.beta,
            kind: matches!(m.kind, FluidKind::Liquid) as u32, // 0 = gaz, 1 = liquide
            low_mach: self.low_mach_active(),
            no_slip: self.no_slip,
            p_th,
            r_spec: m.r_spec,
            rho_amb: m.rho_ambient(),
            dpdt: self.dpdt,
            k_fluid: m.conductivity,
            k_solid: m.solid_conductivity,
            cp: m.cp,
            cap_solid: m.heat_capacity_solid(),
            inv_dx2: self.inv_dx2(),

            boundaries: [self.left_boundary.to_grid(dx), self.right_boundary.to_grid(dx),
                self.top_boundary.to_grid(dx), self.bottom_boundary.to_grid(dx)],
            special_cells: &specials,
        };

        gpu.step_velocity(&cfg, &mut GpuFields {
            velocity_x: &mut self.velocity_x,
            velocity_y: &mut self.velocity_y,
            pressure: &mut self.pressure,
            divergence: &mut self.divergence,
            tracer: &mut self.tracer,
            temperature: &mut self.temperature,
            solids: &self.solids,
            source: &self.source,
            heat_source: &self.heat_source,
        });

        let substeps = self.scalar_substeps(dt);
        gpu.step_scalars(&cfg, substeps, &mut self.tracer, &mut self.temperature);
    }
}