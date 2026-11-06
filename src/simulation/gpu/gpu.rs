use crate::simulation::fluid::BoundaryType;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    left_type: u32,
    right_type: u32,
    top_type: u32,
    bottom_type: u32,
}

fn boundary_type_code(b: BoundaryType) -> u32 {
    match b {
        BoundaryType::Wall => 0,
        BoundaryType::Inlet => 1,
        BoundaryType::Outlet => 2,
    }
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub width: usize,
    pub height: usize,

    pub velocity_x_buf: wgpu::Buffer,
    pub velocity_y_buf: wgpu::Buffer,
    pub pressure_buf: wgpu::Buffer,
    pub pressure_temp_buf: wgpu::Buffer,
    pub divergence_buf: wgpu::Buffer,
    pub solids_buf: wgpu::Buffer,

    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    divergence_pipeline: wgpu::ComputePipeline,
    solve_pressure_pipeline: wgpu::ComputePipeline,
    pressure_boundaries_pipeline: wgpu::ComputePipeline,
    correct_velocity_pipeline: wgpu::ComputePipeline,

    readback_buf: wgpu::Buffer,
}

impl GpuContext {
    pub fn new(width: usize, height: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("No GPU adaptator found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fluid_sim device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).expect("failed to create a GPU device");

        let size = width * height;
        let f32_byte_size = (size * std::mem::size_of::<f32>()) as u64;
        let u32_byte_size = (size * std::mem::size_of::<u32>()) as u64;

        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let make_f32_buffer = |label: &str| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: f32_byte_size,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let velocity_x_buf = make_f32_buffer("velocity_x");
        let velocity_y_buf = make_f32_buffer("velocity_y");
        let pressure_buf = make_f32_buffer("pressure");
        let pressure_temp_buf = make_f32_buffer("pressure_temp");
        let divergence_buf = make_f32_buffer("divergence");

        let solids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("solids"),
            size: u32_byte_size,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        // Staging : vitesse x | vitesse y | pression, relus en un seul map.
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: f32_byte_size * 3,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_sim bind group layout"),
            entries: &[
                storage_entry(0, false), // velocity_x
                storage_entry(1, false), // velocity_y
                storage_entry(2, false), // pressure
                storage_entry(3, false), // pressure_temp
                storage_entry(4, false), // divergence
                storage_entry(5, true),  // solids
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid_sim bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: velocity_x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocity_y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pressure_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pressure_temp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: divergence_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: solids_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
            ],
        });

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

        let divergence_pipeline = make_pipeline("calculate_divergence");
        let solve_pressure_pipeline = make_pipeline("solve_pressure_pass");
        let pressure_boundaries_pipeline = make_pipeline("apply_pressure_boundaries");
        let correct_velocity_pipeline = make_pipeline("correct_velocity");

        Self {
            device, queue, width, height,
            velocity_x_buf, velocity_y_buf,
            pressure_buf, pressure_temp_buf,
            divergence_buf, solids_buf,
            params_buf, bind_group,
            divergence_pipeline, solve_pressure_pipeline,
            pressure_boundaries_pipeline, correct_velocity_pipeline,
            readback_buf,
        }
    }

    pub fn upload_f32(&self, buffer: &wgpu::Buffer, data: &[f32]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn upload_solids(&self, data: &[bool]) {
        let as_u32: Vec<u32> = data.iter().map(|&b| b as u32).collect();
        self.queue.write_buffer(&self.solids_buf, 0, bytemuck::cast_slice(&as_u32));
    }

    pub fn readback_f32(&self, buffer: &wgpu::Buffer, out: &mut [f32]) {
        let byte_size = (out.len() * std::mem::size_of::<f32>()) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &self.readback_buf, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(0..byte_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("failed to map read buffer");

        let data = slice.get_mapped_range();
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.readback_buf.unmap();
    }

    pub fn enforce_incompressibility(
        &self,
        pressure_iterations: usize,
        left_type: BoundaryType,
        right_type: BoundaryType,
        top_type: BoundaryType,
        bottom_type: BoundaryType,
        velocity_x: &mut [f32],
        velocity_y: &mut [f32],
        pressure: &mut [f32],
    ) {
        let width = self.width as u32;
        let height = self.height as u32;

        let size = self.width * self.height;
        assert_eq!(velocity_x.len(), size, "velocity_x: wrong size");
        assert_eq!(velocity_y.len(), size, "velocity_y: wrong size");
        assert_eq!(pressure.len(), size, "pressure: wrong size");

        let field_bytes = (size * std::mem::size_of::<f32>()) as u64;

        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;
        let wg_border = (width.max(height) + 63) / 64;

        // Les paramètres sont identiques pour toutes les passes : une seule écriture.
        let params = GpuParams {
            width, height,
            left_type: boundary_type_code(left_type),
            right_type: boundary_type_code(right_type),
            top_type: boundary_type_code(top_type),
            bottom_type: boundary_type_code(bottom_type),
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("incompressibility encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("divergence"), timestamp_writes: None });
            pass.set_pipeline(&self.divergence_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("initial boundaries"), timestamp_writes: None });
            pass.set_pipeline(&self.pressure_boundaries_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_border, 1, 1);
        }

        for _ in 0..pressure_iterations {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("solve_pressure"), timestamp_writes: None });
                pass.set_pipeline(&self.solve_pressure_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            encoder.copy_buffer_to_buffer(&self.pressure_temp_buf, 0, &self.pressure_buf, 0, field_bytes);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pressure_boundaries"), timestamp_writes: None });
                pass.set_pipeline(&self.pressure_boundaries_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(wg_border, 1, 1);
            }
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("correct_velocity"), timestamp_writes: None });
            pass.set_pipeline(&self.correct_velocity_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Relecture : vitesse x | vitesse y | pression, côte à côte dans le staging buffer.
        encoder.copy_buffer_to_buffer(&self.velocity_x_buf, 0, &self.readback_buf, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&self.velocity_y_buf, 0, &self.readback_buf, field_bytes, field_bytes);
        encoder.copy_buffer_to_buffer(&self.pressure_buf, 0, &self.readback_buf, 2 * field_bytes, field_bytes);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(0..3 * field_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("failed to map read buffer");

        {
            let data = slice.get_mapped_range();
            let bytes: &[u8] = &data;
            let floats: &[f32] = bytemuck::cast_slice(bytes);
            velocity_x.copy_from_slice(&floats[0..size]);
            velocity_y.copy_from_slice(&floats[size..2 * size]);
            pressure.copy_from_slice(&floats[2 * size..3 * size]);
        }
        self.readback_buf.unmap();
    }
}