//! Hôte wgpu du solveur multiphasique GPU. ATTENTION : ce fichier n'a PAS pu être compilé dans l'environnement de développement
//! (pas de crate wgpu hors ligne) ; il reproduit ligne à ligne l'orchestration du banc WebGPU `gpu_run.js` qui, lui, a été exécuté
//! et comparé au CPU. Style d'API wgpu identique à gpu.rs (wgpu 0.19/0.20, pollster, bytemuck).
//!
//! Un seul buffer de stockage `S` (f32) + un uniforme `Params` (512 octets) : aucune limite de nombre de buffers.
//! Taille de S ≈ (≈ 30·np + 40)·nx·ny·4 octets (≈ 40 Mo pour 256×256, 3 phases) ; limite par défaut 128 Mo (≈ 380×380).

use std::collections::HashMap;

use super::mp_gpu_params::{gpu_layout_header, GpuState};
use super::multiphase::MultiphaseSim;

const WGSL_TEMPLATE: &str = include_str!("multiphase.template.wgsl");

pub struct MultiphaseGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    s_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    nx: usize,
    ny: usize,
    np: usize,
    total: usize,
    offsets: Vec<(&'static str, usize, usize)>,
    /// itérations PCG (préconditionneur de Jacobi) par itération de Newton : prendre ≳ 6·max(nx, ny)
    pub pcg_iter: u32,
    pub newton_iter: u32,
}

/// Nombre maximal de fluides du solveur GPU (tableau `ph: array<PhaseP, 4>` des shaders).
pub const GPU_MAX_PHASES: usize = 4;

/// Pourquoi le GPU ne peut PAS simuler cette scène (None = il le peut). Le GPU ne gère ni les cellules spéciales / entrées de bord /
/// solides thermiques, ni la miscibilité (groupes de fluides), ni plus de `GPU_MAX_PHASES` fluides : le CPU f64 les gère tous.
pub fn gpu_unsupported_reason(sim: &MultiphaseSim) -> Option<String> {
    if sim.reg.phases.len() > GPU_MAX_PHASES {
        return Some(format!("{} fluides (le GPU en gère {GPU_MAX_PHASES} au maximum)", sim.reg.phases.len()));
    }
    if sim.has_extras() {
        return Some("cellules spéciales, solides thermiques ou entrées de bord (CPU uniquement)".into());
    }
    if !sim.reg.miscible.is_empty() {
        return Some("fluides miscibles (CPU uniquement)".into());
    }
    if !sim.reg.reactions.is_empty() {
        return Some("réactions chimiques / combustion (CPU uniquement)".into());
    }
    if !sim.ambient_mix.is_empty() || sim.reg.phases.iter().any(|p| p.cp_slope != 0.0) {
        return Some("ambiant mélangé ou cp variable avec T (CPU uniquement)".into());
    }
    None
}

impl MultiphaseGpu {
    pub fn try_new(sim: &MultiphaseSim) -> Option<Self> {
        if gpu_unsupported_reason(sim).is_some() {
            return None;
        }
        let (nx, ny, np) = (sim.nx, sim.ny, sim.reg.phases.len());
        let (header, total, offsets) = gpu_layout_header(nx, ny, np);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let mut limits = wgpu::Limits::default();
        let bytes = (total * 4) as u32;
        if bytes > limits.max_storage_buffer_binding_size {
            limits.max_storage_buffer_binding_size = bytes.min(adapter.limits().max_storage_buffer_binding_size);
            limits.max_buffer_size = limits.max_storage_buffer_binding_size as u64;
        }
        if (total * 4) as u64 > limits.max_buffer_size {
            return None; // grille trop grande pour un seul buffer : réduire nx·ny ou scinder le buffer
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor { label: Some("multiphase device"), required_features: wgpu::Features::empty(), required_limits: limits },
            None,
        ))
        .ok()?;

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let s_buf = device.create_buffer(&wgpu::BufferDescriptor { label: Some("S"), size: (total * 4) as u64, usage, mapped_at_creation: false });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (total * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: super::mp_gpu_params::PARAMS_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let source = format!("{header}{WGSL_TEMPLATE}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("multiphase shaders"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        let layout_entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("mp bgl"), entries: &layout_entries });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mp bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mp pl"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let mut pipelines = HashMap::new();
        for name in kernel_names(&source) {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&name), layout: Some(&pl), module: &shader, entry_point: &name });
            pipelines.insert(name, p);
        }
        let mut g = MultiphaseGpu { device, queue, s_buf, params_buf, readback, bind_group, pipelines, nx, ny, np, total, offsets, pcg_iter: (6 * nx.max(ny)) as u32, newton_iter: 3 };
        g.upload(sim);
        Some(g)
    }

    /// Envoie l'état CPU (m_k, ε, p, u, v, solides) vers le GPU
    pub fn upload(&mut self, sim: &MultiphaseSim) {
        let mut img = vec![0f32; self.total];
        sim.gpu_state().write_into(sim, &mut img);
        self.queue.write_buffer(&self.s_buf, 0, bytemuck::cast_slice(&img));
    }

    fn off(&self, name: &str) -> usize {
        self.offsets.iter().find(|x| x.0 == name).unwrap().1
    }

    /// Un pas complet sur GPU. dt, sous-pas de conduction : choisis côté CPU à partir de l'état relu au pas précédent.
    pub fn step(&mut self, sim: &mut MultiphaseSim) -> f64 {
        sim.refresh();
        let dt = sim.choose_dt();
        let nsub = sim.conduction_substeps(dt);
        let mut params = sim.gpu_params_bytes(dt, nsub, self.newton_iter, self.pcg_iter);
        let parity = (sim.step_count % 2) as u32;
        let flags = u32::from_le_bytes([params[80], params[81], params[82], params[83]]);
        self.queue.write_buffer(&self.params_buf, 0, &params);
        params.clear();

        let (nx, ny, np) = (self.nx, self.ny, self.np);
        let n = nx * ny;
        let (nu, nv) = ((nx + 1) * ny, nx * (ny + 1));
        let nf = nu.max(nv);
        let nn = (nx + 1) * (ny + 1);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("multiphase step") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("multiphase"), timestamp_writes: None });
            pass.set_bind_group(0, &self.bind_group, &[]);
            let pipes = &self.pipelines;
            let mut d = |name: &str, count: usize| {
                pass.set_pipeline(&pipes[name]);
                pass.dispatch_workgroups(((count + 63) / 64).max(1) as u32, 1, 1);
            };
            let (st, vi, co, pc) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0, flags & 8 != 0);
            d("k_refresh", n);
            d("k_force_init", nf);
            if st {
                d("k_marker", n); d("k_smooth", n); d("k_curv", n); d("k_force_st_u", nu); d("k_force_st_v", nv);
            }
            if vi {
                d("k_visc_diag", n); d("k_visc_xy", nn); d("k_visc_u", nu); d("k_visc_v", nv);
            }
            d("k_save_mtot", n); d("k_zero_flux", nf); d("k_aw_init", np * n);
            for q in 0..2u32 {
                let dir = (parity + q) % 2;
                let (sfx, nfd) = if dir == 0 { ("x", nu) } else { ("y", nv) };
                d("k_an", n);
                d(&format!("k_sweep_flux_{sfx}"), nfd);
                d(&format!("k_sweep_scale_{sfx}"), n);
                d(&format!("k_sweep_apply_{sfx}"), nfd);
                d(&format!("k_sweep_update_{sfx}"), n);
            }
            d("k_mom_u", nu); d("k_mom_v", nv); d("k_add_forces", nf);
            d("k_refresh", n);
            if co {
                for _ in 0..nsub {
                    d("k_conduct_step", n);
                    d("k_conduct_commit", n);
                }
            }
            if pc {
                d("k_phase_change", n);
            }
            d("k_refresh", n); d("k_save_p", n);
            for _ in 0..self.newton_iter {
                d("k_closure", n); d("k_face_tmp", nf); d("k_rhs", n); d("k_stencil", n); d("k_pcg_init", n);
                d("k_dot_rz", n); d("k_fin_rz0", 1);
                for _ in 0..self.pcg_iter {
                    d("k_apply", n); d("k_dot_pap", n); d("k_fin_pap", 1); d("k_cg_update", n);
                    d("k_dot_rz", n); d("k_fin_rznew", 1); d("k_cg_dir", n);
                }
                d("k_p_update", n);
            }
            d("k_face_final", nf); d("k_pressure_work", n); d("k_refresh", n);
        }
        enc.copy_buffer_to_buffer(&self.s_buf, 0, &self.readback, 0, (self.total * 4) as u64);
        self.queue.submit(Some(enc.finish()));

        // relecture
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map readback");
        {
            let data = slice.get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&data);
            let take = |name: &str, len: usize| all[self.off(name)..self.off(name) + len].to_vec();
            let st = GpuState {
                mass: take("mass", np * n),
                eps: take("eps", n),
                prel: take("prel", n),
                u: take("U", nu),
                v: take("V", nv),
                solid: sim.solid.clone(),
            };
            sim.load_gpu_state(&st);
        }
        self.readback.unmap();
        sim.time += dt;
        sim.last_dt = dt;
        sim.step_count += 1;
        dt
    }
}

/// noms des points d'entrée `fn k_…` du source WGSL
fn kernel_names(src: &str) -> Vec<String> {
    let mut v = vec![];
    let mut rest = src;
    while let Some(i) = rest.find("fn k_") {
        let tail = &rest[i + 3..];
        let end = tail.find(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap_or(tail.len());
        v.push(tail[..end].to_string());
        rest = &tail[end..];
    }
    v.sort();
    v.dedup();
    v
}
