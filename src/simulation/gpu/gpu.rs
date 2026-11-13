use crate::simulation::fluid::{Boundary, BoundaryType, CellEffect, CellType};
use crate::simulation::materials::FluidKind;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    omega: f32,
    dt: f32,
    dts: f32,
    diffusion: f32,
    vmax: f32,
    special_count: u32,
    visc_k: f32,
    gravity: f32,
    thermal_fluid: f32,
    thermal_solid: f32,
    t_ref: f32,
    beta: f32,
    kind: u32,
    _pad: u32,
    boundary_type: [u32; 4], // gauche, droite, haut, bas
    boundary: [[f32; 4]; 4], // (vx, vy, density, offset) x 4
    boundary_temp: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSpecial {
    idx: u32,
    kind: u32,
    a: f32,
    b: f32,
    c: f32,
    dest: u32,
}

impl GpuSpecial {
    fn from_effect(idx: usize, effect: CellEffect) -> Self {
        let idx = idx as u32;
        match effect {
            CellEffect::Sink { rate } => Self {
                idx, kind: 1, a: rate, b: 0.0, c: 0.0, dest: 0
            },
            CellEffect::Emitter { vx, vy, density } => Self {
                idx, kind: 2, a: vx, b: vy, c: density, dest: 0
            },
            CellEffect::Accelerator { ax, ay } => Self {
                idx, kind: 3, a: ax, b: ay, c: 0.0, dest: 0
            },
            CellEffect::Teleporter { dest_idx } => Self {
                idx, kind: 4, a: 0.0, b: 0.0, c: 0.0, dest: dest_idx as u32
            },
        }
    }
}


pub struct StepConfig<'a> {
    pub dt: f32,
    pub pressure_iterations: usize,
    pub omega: f32,
    pub tracer: f32,
    pub vmax: f32,
    // gauche, droite, haut, bas
    pub special_cells: &'a [(usize, CellEffect)],
    pub viscosity: f32,
    pub viscosity_substeps: usize,
    pub gravity: f32,
    pub thermal_fluid: f32,
    pub thermal_solid: f32,
    pub t_ref: f32,
    pub beta: f32,
    pub kind: u32,
    pub boundaries: [Boundary; 4],
}

pub struct GpuFields<'a> {
    pub velocity_x: &'a mut [f32],
    pub velocity_y: &'a mut [f32],
    pub pressure: &'a mut [f32],
    pub divergence: &'a mut [f32],
    pub density: &'a mut [f32],
    pub solids: &'a [CellType],
    pub temperature: &'a mut [f32],
}

struct Pipelines {
    advect_velocity: wgpu::ComputePipeline,
    commit_velocity: wgpu::ComputePipeline,
    special_velocity: wgpu::ComputePipeline,
    close_faces: wgpu::ComputePipeline,
    boundary_lr: wgpu::ComputePipeline,
    boundary_tb: wgpu::ComputePipeline,
    limit_velocity: wgpu::ComputePipeline,
    divergence: wgpu::ComputePipeline,
    solve_red: wgpu::ComputePipeline,
    solve_black: wgpu::ComputePipeline,
    correct_velocity: wgpu::ComputePipeline,
    prepare_ghost_density: wgpu::ComputePipeline,
    density_step: wgpu::ComputePipeline,
    density_commit: wgpu::ComputePipeline,
    special_density: wgpu::ComputePipeline,
    apply_density_boundaries: wgpu::ComputePipeline,
    diffuse_velocity: wgpu::ComputePipeline,
    gravity: wgpu::ComputePipeline,
    prepare_ghost_temperature: wgpu::ComputePipeline,
    temperature_step: wgpu::ComputePipeline,
    temperature_commit: wgpu::ComputePipeline,
    apply_temperature_boundaries:wgpu::ComputePipeline,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub width: usize,
    pub height: usize,

    velocity_x_buf: wgpu::Buffer,
    velocity_y_buf: wgpu::Buffer,
    pressure_buf: wgpu::Buffer,
    divergence_buf: wgpu::Buffer,
    cells_buf: wgpu::Buffer,
    density_buf: wgpu::Buffer,
    scratch_buf: wgpu::Buffer,
    special_buf: wgpu::Buffer,
    special_capacity: usize,
    params_buf: wgpu::Buffer,

    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipelines: Pipelines,

    readback_buf: wgpu::Buffer,
}

impl GpuContext {
    pub fn try_new(width: usize, height: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fluid_sim device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).ok()?;

        let size = width * height;
        let f32_bytes = (size * size_of::<f32>()) as u64;

        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        let make_buffer = |label: &str, bytes: u64| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let velocity_x_buf = make_buffer("velocity_x", f32_bytes);
        let velocity_y_buf = make_buffer("velocity_y", f32_bytes);
        let pressure_buf = make_buffer("pressure", f32_bytes);
        let divergence_buf = make_buffer("divergence", f32_bytes);
        let density_buf = make_buffer("density", f32_bytes);
        let scratch_buf = make_buffer("scratch", 5 * f32_bytes);
        let cells_buf = make_buffer("cells", (size * size_of::<u32>()) as u64);

        let special_capacity = 16;
        let special_buf = make_buffer("specials", (special_capacity * size_of::<GpuSpecial>()) as u64);

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: f32_bytes * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid_sim shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fluid_sim.wgsl").into()),
        });

        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // 8 buffers de stockage (= limite par défaut de wgpu) + 1 uniform
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_sim bind group layout"),
            entries: &[
                storage_entry(0, false), // velocity_x
                storage_entry(1, false), // velocity_y
                storage_entry(2, false), // pressure
                storage_entry(3, false), // divergence
                storage_entry(4, true),  // cells
                storage_entry(5, false), // density
                storage_entry(6, false), // scratch
                storage_entry(7, true),  // specials
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = Self::make_bind_group(
            &device, &bind_group_layout,
            &velocity_x_buf, &velocity_y_buf, &pressure_buf, &divergence_buf,
            &cells_buf, &density_buf, &scratch_buf, &special_buf, &params_buf,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid_sim pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry_point: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point,
        });

        let pipelines = Pipelines {
            advect_velocity: make_pipeline("advect_velocity"),
            commit_velocity: make_pipeline("commit_velocity"),
            special_velocity: make_pipeline("special_velocity"),
            close_faces: make_pipeline("close_faces"),
            boundary_lr: make_pipeline("boundary_lr"),
            boundary_tb: make_pipeline("boundary_tb"),
            limit_velocity: make_pipeline("limit_velocity"),
            divergence: make_pipeline("calculate_divergence"),
            solve_red: make_pipeline("solve_red"),
            solve_black: make_pipeline("solve_black"),
            correct_velocity: make_pipeline("correct_velocity"),
            prepare_ghost_density: make_pipeline("prepare_ghost_density"),
            density_step: make_pipeline("density_step"),
            density_commit: make_pipeline("density_commit"),
            special_density: make_pipeline("special_density"),
            apply_density_boundaries: make_pipeline("apply_density_boundaries"),
            diffuse_velocity: make_pipeline("diffuse_velocity"),
            gravity: make_pipeline("buoyancy"),
            prepare_ghost_temperature: make_pipeline("prepare_ghost_temperature"),
            temperature_step: make_pipeline("temperature_step"),
            temperature_commit: make_pipeline("temperature_commit"),
            apply_temperature_boundaries: make_pipeline("apply_temperature_boundaries"),
        };

        Some(Self {
            device, queue, width, height,
            velocity_x_buf, velocity_y_buf, pressure_buf, divergence_buf,
            cells_buf, density_buf, scratch_buf, special_buf, special_capacity,
            params_buf,
            bind_group_layout, bind_group, pipelines,
            readback_buf,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        velocity_x: &wgpu::Buffer,
        velocity_y: &wgpu::Buffer,
        pressure: &wgpu::Buffer,
        divergence: &wgpu::Buffer,
        cells: &wgpu::Buffer,
        density: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        specials: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid_sim bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: density.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: scratch.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: specials.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: params.as_entire_binding() },
            ],
        })
    }

    fn upload_f32(&self, buffer: &wgpu::Buffer, data: &[f32]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    fn temperature_offset(&self) -> u64 {
        (4 * self.width * self.height * size_of::<f32>()) as u64
    }

    fn upload_specials(&mut self, specials: &[(usize, CellEffect)]) {
        if specials.len() > self.special_capacity {
            let capacity = specials.len().next_power_of_two().max(16);
            self.special_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("specials"),
                size: (capacity * size_of::<GpuSpecial>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.special_capacity = capacity;
            self.bind_group = Self::make_bind_group(
                &self.device, &self.bind_group_layout,
                &self.velocity_x_buf, &self.velocity_y_buf, &self.pressure_buf, &self.divergence_buf,
                &self.cells_buf, &self.density_buf, &self.scratch_buf, &self.special_buf, &self.params_buf,
            );
        }
        if !specials.is_empty() {
            let data: Vec<GpuSpecial> = specials.iter().map(|&(i, e)| GpuSpecial::from_effect(i, e)).collect();
            self.queue.write_buffer(&self.special_buf, 0, bytemuck::cast_slice(&data));
        }
    }

    fn upload_cells(&self, solids: &[CellType]) {
        let cells: Vec<u32> = solids.iter().map(|&c| c as u32).collect();
        self.queue.write_buffer(&self.cells_buf, 0, bytemuck::cast_slice(&cells));
    }

    fn write_params(&self, cfg: &StepConfig, dts: f32) {
        let ty = |b: &Boundary| match b.boundary_type {
            BoundaryType::Wall => 0u32,
            BoundaryType::Inlet => 1,
            BoundaryType::Outlet => 2,
        };
        let data = |b: &Boundary| [b.velocity_x, b.velocity_y, b.density, b.offset as f32];
        let b = &cfg.boundaries;

        let params = GpuParams {
            width: self.width as u32,
            height: self.height as u32,
            omega: cfg.omega,
            dt: cfg.dt,
            dts,
            diffusion: cfg.tracer,
            vmax: cfg.vmax,
            special_count: cfg.special_cells.len() as u32,
            boundary_type: [ty(&b[0]), ty(&b[1]), ty(&b[2]), ty(&b[3])],
            boundary: [data(&b[0]), data(&b[1]), data(&b[2]), data(&b[3])],
            visc_k: cfg.viscosity * cfg.dt / cfg.viscosity_substeps.max(1) as f32,
            gravity: cfg.gravity,
            t_ref: cfg.t_ref,
            beta: cfg.beta,
            kind: cfg.kind,
            _pad: 0,
            thermal_fluid: cfg.thermal_fluid,
            thermal_solid: cfg.thermal_solid,
            boundary_temp: [b[0].temperature, b[1].temperature, b[2].temperature, b[3].temperature],
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    fn read_back(&self, bytes: u64, f: impl FnOnce(&[f32])) {
        let slice = self.readback_buf.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("failed to map read buffer");

        {
            let data = slice.get_mapped_range();
            let bytes: &[u8] = &data;
            f(bytemuck::cast_slice(bytes));
        }
        self.readback_buf.unmap();
    }

    pub fn step_velocity(&mut self, cfg: &StepConfig, f: &mut GpuFields) {
        let size = self.width * self.height;
        assert_eq!(f.velocity_x.len(), size, "velocity_x: wrong size");
        assert_eq!(f.velocity_y.len(), size, "velocity_y: wrong size");
        assert_eq!(f.pressure.len(), size, "pressure: wrong size");
        assert_eq!(f.divergence.len(), size, "divergence: wrong size");
        assert_eq!(f.density.len(), size, "density: wrong size");
        assert_eq!(f.solids.len(), size, "solids: wrong size");
        assert_eq!(f.temperature.len(), size, "temperature: wrong size");

        self.upload_specials(cfg.special_cells);
        self.upload_cells(f.solids);
        self.upload_f32(&self.velocity_x_buf, f.velocity_x);
        self.upload_f32(&self.velocity_y_buf, f.velocity_y);
        self.upload_f32(&self.pressure_buf, f.pressure);
        self.upload_f32(&self.density_buf, f.density);
        self.queue.write_buffer(&self.scratch_buf, self.temperature_offset(), bytemuck::cast_slice(f.temperature));
        self.write_params(cfg, 0.0);

        let (w, h) = (self.width as u32, self.height as u32);
        let (gx, gy) = ((w + 7) / 8, (h + 7) / 8);
        let lines = (w.max(h) + 63) / 64;
        let p = &self.pipelines;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("velocity encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("velocity + projection"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            //trop de magie noire :(
            macro_rules! run {
                ($pipeline:expr, $x:expr, $y:expr) => {{
                    pass.set_pipeline(&$pipeline);
                    pass.dispatch_workgroups($x, $y, 1);
                }};
            }

            run!(p.advect_velocity, gx, gy);
            run!(p.commit_velocity, gx, gy);
            if cfg.viscosity > 0.0 {
                for _ in 0..cfg.viscosity_substeps {
                    run!(p.diffuse_velocity, gx, gy);
                    run!(p.commit_velocity, gx, gy);
                }
            }
            if cfg.gravity != 0.0 {
                run!(p.gravity, gx, gy);
            }
            run!(p.special_velocity, 1, 1);

            run!(p.close_faces, gx, gy);
            run!(p.boundary_lr, lines, 1);
            run!(p.boundary_tb, lines, 1); // après boundary_lr
            run!(p.limit_velocity, gx, gy);

            run!(p.divergence, gx, gy);
            for _ in 0..cfg.pressure_iterations {
                run!(p.solve_red, gx, gy);
                run!(p.solve_black, gx, gy);
            }
            run!(p.correct_velocity, gx, gy);
            run!(p.divergence, gx, gy); // divergence résiduelle
        }

        let field_bytes = (size * size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.velocity_x_buf, 0, &self.readback_buf, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&self.velocity_y_buf, 0, &self.readback_buf, field_bytes, field_bytes);
        encoder.copy_buffer_to_buffer(&self.pressure_buf, 0, &self.readback_buf, 2 * field_bytes, field_bytes);
        encoder.copy_buffer_to_buffer(&self.divergence_buf, 0, &self.readback_buf, 3 * field_bytes, field_bytes);
        self.queue.submit(Some(encoder.finish()));

        self.read_back(4 * field_bytes, |floats| {
            f.velocity_x.copy_from_slice(&floats[0..size]);
            f.velocity_y.copy_from_slice(&floats[size..2 * size]);
            f.pressure.copy_from_slice(&floats[2 * size..3 * size]);
            f.divergence.copy_from_slice(&floats[3 * size..4 * size]);
        });
    }

    // advection/diffusion de la densité en `substeps` sous-pas, cellules spéciales,
    // cellules fantômes
    // `substeps` dépend vmax après proj, calculé côté CPU
    pub fn step_density(&mut self, cfg: &StepConfig, substeps: usize, density: &mut [f32], temperature: &mut [f32]) {
        let size = self.width * self.height;
        assert_eq!(density.len(), size, "density: wrong size");
        assert_eq!(temperature.len(), size, "temperature: wrong size");

        let substeps = substeps.max(1);
        self.write_params(cfg, cfg.dt / substeps as f32);

        let (w, h) = (self.width as u32, self.height as u32);
        let (gx, gy) = ((w + 7) / 8, (h + 7) / 8);
        let lines = (w.max(h) + 63) / 64;
        let p = &self.pipelines;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("density encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("density"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            //magie noire
            macro_rules! run {
                ($pipeline:expr, $x:expr, $y:expr) => {{
                    pass.set_pipeline(&$pipeline);
                    pass.dispatch_workgroups($x, $y, 1);
                }};
            }

            run!(p.prepare_ghost_density, lines, 1);
            run!(p.prepare_ghost_temperature, lines, 1);
            for _ in 0..substeps {
                run!(p.density_step, gx, gy);
                run!(p.density_commit, gx, gy);
                run!(p.temperature_step, gx, gy);
                run!(p.temperature_commit, gx, gy);
            }
            run!(p.special_density, 1, 1);
            run!(p.apply_density_boundaries, lines, 1);
            run!(p.apply_temperature_boundaries, lines, 1);
        }

        let field_bytes = (size * size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.density_buf, 0, &self.readback_buf, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&self.scratch_buf, self.temperature_offset(), &self.readback_buf, field_bytes, field_bytes);
        self.queue.submit(Some(encoder.finish()));

        self.read_back(2 * field_bytes, |floats| {
            density.copy_from_slice(&floats[0..size]);
            temperature.copy_from_slice(&floats[size..2 * size]);
        });
    }
}