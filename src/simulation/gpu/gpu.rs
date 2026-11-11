use crate::simulation::fluid::CellType;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    omega: f32,
    pad: u32,
}

const FLAG_OPEN_U: u32 = 1;
const FLAG_OPEN_V: u32 = 2;
const FLAG_SOLID: u32 = 4;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub width: usize,
    pub height: usize,

    pub velocity_x_buf: wgpu::Buffer,
    pub velocity_y_buf: wgpu::Buffer,
    pub pressure_buf: wgpu::Buffer,
    pub divergence_buf: wgpu::Buffer,
    pub flags_buf: wgpu::Buffer,

    params_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    divergence_pipeline: wgpu::ComputePipeline,
    solve_red_pipeline: wgpu::ComputePipeline,
    solve_black_pipeline: wgpu::ComputePipeline,
    correct_velocity_pipeline: wgpu::ComputePipeline,

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
        let f32_byte_size = (size * size_of::<f32>()) as u64;
        let u32_byte_size = (size * size_of::<u32>()) as u64;

        let storage_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

        let make_f32_buffer = |label: &str| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: f32_byte_size,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let velocity_x_buf = make_f32_buffer("velocity_x");
        let velocity_y_buf = make_f32_buffer("velocity_y");
        let pressure_buf = make_f32_buffer("pressure");
        let divergence_buf = make_f32_buffer("divergence");

        let flags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flags"),
            size: u32_byte_size,
            usage: storage_usage,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: f32_byte_size * 4,
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid_sim bind group layout"),
            entries: &[
                storage_entry(0, false), // velocity_x
                storage_entry(1, false), // velocity_y
                storage_entry(2, false), // pressure
                storage_entry(3, false), // divergence
                storage_entry(4, true), // flags
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
                wgpu::BindGroupEntry { binding: 3, resource: divergence_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
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
        let solve_red_pipeline = make_pipeline("solve_red");
        let solve_black_pipeline = make_pipeline("solve_black");
        let correct_velocity_pipeline = make_pipeline("correct_velocity");

        Some(Self {
            device, queue, width, height,
            velocity_x_buf, velocity_y_buf,
            pressure_buf, divergence_buf, flags_buf,
            params_buf, bind_group,
            divergence_pipeline, solve_red_pipeline, solve_black_pipeline,
            correct_velocity_pipeline,
            readback_buf,
        })
    }

    pub fn upload_f32(&self, buffer: &wgpu::Buffer, data: &[f32]) {
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    pub fn upload_flags(&self, solids: &[CellType], open_u: &[u8], open_v: &[u8]) {
        let flags: Vec<u32> = solids
            .iter()
            .zip(open_u.iter().zip(open_v.iter()))
            .map(|(&s, (&ou, &ov))| {
                let mut f = 0;
                if ou != 0 { f |= FLAG_OPEN_U; }
                if ov != 0 { f |= FLAG_OPEN_V; }
                if s == CellType::Wall { f |= FLAG_SOLID; }
                f
            })
            .collect();
        self.queue.write_buffer(&self.flags_buf, 0, bytemuck::cast_slice(&flags));
    }

    pub fn enforce_incompressibility(&self, pressure_iterations: usize, omega: f32, velocity_x: &mut [f32],
                                    velocity_y: &mut [f32], pressure: &mut [f32], divergence: &mut [f32]) {
        let width = self.width as u32;
        let height = self.height as u32;

        let size = self.width * self.height;
        assert_eq!(velocity_x.len(), size, "velocity_x: wrong size");
        assert_eq!(velocity_y.len(), size, "velocity_y: wrong size");
        assert_eq!(pressure.len(), size, "pressure: wrong size");
        assert_eq!(divergence.len(), size, "divergence: wrong size");

        let field_bytes = (size * size_of::<f32>()) as u64;

        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        let params = GpuParams { width, height, omega, pad: 0 };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("incompressibility encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("projection"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);

            pass.set_pipeline(&self.divergence_pipeline);
            pass.dispatch_workgroups(wg_x, wg_y, 1);

            for _ in 0..pressure_iterations {
                pass.set_pipeline(&self.solve_red_pipeline);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                pass.set_pipeline(&self.solve_black_pipeline);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            pass.set_pipeline(&self.correct_velocity_pipeline);
            pass.dispatch_workgroups(wg_x, wg_y, 1);

            pass.set_pipeline(&self.divergence_pipeline);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        encoder.copy_buffer_to_buffer(&self.velocity_x_buf, 0, &self.readback_buf, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&self.velocity_y_buf, 0, &self.readback_buf, field_bytes, field_bytes);
        encoder.copy_buffer_to_buffer(&self.pressure_buf, 0, &self.readback_buf, 2 * field_bytes, field_bytes);
        encoder.copy_buffer_to_buffer(&self.divergence_buf, 0, &self.readback_buf, 3 * field_bytes, field_bytes);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback_buf.slice(0..4 * field_bytes);
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
            divergence.copy_from_slice(&floats[3 * size..4 * size]);
        }
        self.readback_buf.unmap();
    }
}