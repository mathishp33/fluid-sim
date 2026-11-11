use std::cmp::PartialEq;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::simulation::gpu::gpu::{GpuContext, GpuFields, StepConfig};

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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Boundary {
    pub boundary_type: BoundaryType,
    pub velocity_x: f32,
    pub velocity_y: f32,
    pub density: f32,
    pub offset: usize,
}

impl Boundary {
    pub fn wall() -> Self {
        Self {
            boundary_type: BoundaryType::Wall,
            velocity_x: 0.0,
            velocity_y: 0.0,
            density: 0.0,
            offset: 1,
        }
    }
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

    pub fn to_string(self) -> String {
        match self {
            CellType::Fluid => {
                "Fluid".parse().unwrap()
            }
            CellType::Wall => {
                "Wall".parse().unwrap()
            }
            CellType::Sink => {
                "Sink".parse().unwrap()
            }
            CellType::Emitter => {
                "Emitter".parse().unwrap()
            }
            CellType::Accelerator => {
                "Accelerator".parse().unwrap()
            }
            CellType::Teleporter => {
                "Teleporter".parse().unwrap()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CellEffect {
    Sink { rate: f32 },
    Emitter { vx: f32, vy: f32, density: f32 },
    Accelerator { ax: f32, ay: f32 },
    Teleporter { dest_idx: usize },
}

impl CellEffect {
    pub fn default_for(kind: CellType) -> Option<Self> {
        match kind {
            CellType::Sink => Some(Self::Sink {
                rate: 5.0
            }),
            CellType::Emitter => Some(Self::Emitter {
                vx: 0.0, vy: 0.0, density: 1.0
            }),
            CellType::Accelerator => Some(Self::Accelerator {
                ax: 50.0, ay: 0.0
            }),
            CellType::Teleporter => None, // dest_idx dépend de la cellule, voir set_cell
            _ => None,
        }
    }
}

// les cellules en bordure (x = 0, x = w-1, y = 0, y = h-1) sont des cellules fantômes.
pub struct FluidSim {
    pub width: usize,
    pub height: usize,

    pub density: Vec<f32>,
    pub velocity_x: Vec<f32>,
    pub velocity_y: Vec<f32>,
    pub diffusion_rate: f32,
    pub pressure: Vec<f32>,
    pub divergence: Vec<f32>,
    pub solids: Vec<CellType>,
    pub special_cells: Vec<(usize, CellEffect)>,
    // Facteur de sur-relaxation du Red-Black Gauss-Seidel (1.0 = Gauss-Seidel pur).
    // Avec peu d'itérations par frame, rester vers 1.0-1.5 ; 2/(1+sin(pi/n)) ne vaut que
    // si le solveur a le temps de converger.
    pub sor_omega: f32,

    pub(crate) density_temp: Vec<f32>,
    pub(crate) velocity_x_temp: Vec<f32>,
    pub(crate) velocity_y_temp: Vec<f32>,

    // 1 si la face droite (resp. basse) de la cellule i laisse passer le fluide
    // (aucune des deux cellules n'est solide, ou face de bord vers un Outlet).
    pub(crate) open_u: Vec<u8>,
    pub(crate) open_v: Vec<u8>,

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

    pub fn new(width: usize, height: usize, start_density: f32, diffusion_rate: f32) -> Self {
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

            density: vec![start_density; size],
            velocity_x: vec![0.0; size],
            velocity_y: vec![0.0; size],
            diffusion_rate,
            pressure: vec![0.0; size],
            divergence: vec![0.0; size],
            solids: vec![CellType::Fluid; size],
            special_cells: Vec::new(),
            sor_omega: 1.5,

            density_temp: vec![0.0; size],
            velocity_x_temp: vec![0.0; size],
            velocity_y_temp: vec![0.0; size],

            open_u: vec![0; size],
            open_v: vec![0; size],

            left_boundary: Boundary::wall(),
            right_boundary: Boundary::wall(),
            top_boundary: Boundary::wall(),
            bottom_boundary: Boundary::wall(),

            compute_method,
            gpu_context,
        }
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

        if kind.is_special() {
            let eff = effect
                .or_else(|| CellEffect::default_for(kind))
                .unwrap_or(CellEffect::Teleporter { dest_idx: idx }); // téléporteur sans destination = no-op
            self.special_cells.push((idx, eff));
        }

        if kind == CellType::Wall {
            self.density[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
            self.divergence[idx] = 0.0;

            self.density_temp[idx] = 0.0;
            self.velocity_x_temp[idx] = 0.0;
            self.velocity_y_temp[idx] = 0.0;
            return;
        }

        if old != CellType::Wall {
            return;
        }

        let neighbors = [
            self.idx(x - 1, y),
            self.idx(x + 1, y),
            self.idx(x, y - 1),
            self.idx(x, y + 1),
        ];

        let mut density_sum = 0.0;
        let mut velocity_x_sum = 0.0;
        let mut velocity_y_sum = 0.0;
        let mut pressure_sum = 0.0;
        let mut count = 0.0;

        for neighbor in neighbors {
            if self.solids[neighbor] == CellType::Wall {
                continue;
            }
            density_sum += self.density[neighbor];
            velocity_x_sum += self.velocity_x[neighbor];
            velocity_y_sum += self.velocity_y[neighbor];
            pressure_sum += self.pressure[neighbor];
            count += 1.0;
        }

        if count > 0.0 {
            self.density[idx] = density_sum / count;
            self.velocity_x[idx] = velocity_x_sum / count;
            self.velocity_y[idx] = velocity_y_sum / count;
            self.pressure[idx] = pressure_sum / count;
        } else {
            self.density[idx] = 0.0;
            self.velocity_x[idx] = 0.0;
            self.velocity_y[idx] = 0.0;
            self.pressure[idx] = 0.0;
        }

        self.divergence[idx] = 0.0;
        self.density_temp[idx] = self.density[idx];
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

    pub fn total_mass(&self) -> f64 {
        let w = self.width;
        (1..self.height - 1)
            .into_par_iter()
            .map(|y| self.density[1 + y * w..w - 1 + y * w].iter().map(|&d| d as f64).sum::<f64>())
            .sum()
    }

    pub fn step(&mut self, dt: f32, pressure_iterations: usize) {
        if self.compute_method == ComputeMethod::GPU {
            if let Some(mut gpu) = self.gpu_context.take() {
                self.step_gpu(&mut gpu, dt, pressure_iterations);
                self.gpu_context = Some(gpu);
                return;
            }
        }
        self.step_cpu(dt, pressure_iterations);
    }

    // Pas complet sur GPU, en deux phases (une seule synchronisation par phase) :
    //  1. vitesses : advection, cellules spéciales, CL, limitation, projection ;
    //  2. densité : sous-pas, cellules spéciales, fantômes.
    fn step_gpu(&mut self, gpu: &mut GpuContext, dt: f32, pressure_iterations: usize) {
        let cfg = StepConfig {
            dt,
            pressure_iterations,
            omega: self.sor_omega,
            diffusion: self.diffusion_rate,
            vmax: self.velocity_limit(dt),
            boundaries: [self.left_boundary, self.right_boundary, self.top_boundary, self.bottom_boundary],
            special_cells: &self.special_cells,
        };

        gpu.step_velocity(&cfg, &mut GpuFields {
            velocity_x: &mut self.velocity_x,
            velocity_y: &mut self.velocity_y,
            pressure: &mut self.pressure,
            divergence: &mut self.divergence,
            density: &mut self.density,
            solids: &self.solids,
        });

        let substeps = self.density_substeps(dt);
        gpu.step_density(&cfg, substeps, &mut self.density);
    }
}