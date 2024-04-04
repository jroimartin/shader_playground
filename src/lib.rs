// WGSL's memory layout:
//
// Type                    Alignment in Bytes  Size in Bytes
// scalar (i32, u32, f32)                   4              4
// vec2<T>                                  8              8
// vec3<T>                                 16             12
// vec4<T>                                 16             16
//
// In the case of structs:
//
// AlignOf(S) = max(AlignOfMember(S, M1), ... , AlignOfMember(S, Mn))
//
// where `S` is the struct in question and `M` is a member of the
// struct.
//
// Reference: https://www.w3.org/TR/WGSL/#memory-layouts

use std::{
    borrow::Cow,
    fmt, fs, io, mem,
    path::{Path, PathBuf},
    slice,
    sync::{Arc, Mutex},
    time,
};

use wgpu::{include_wgsl, util::DeviceExt};
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug)]
pub enum Error {
    RequestAdapter,
    GetSurfaceConfig,
    GetTextureFormat,
    UninitRenderPipeline,
    Wgpu(wgpu::Error),
    CreateSurface(wgpu::CreateSurfaceError),
    RequestDevice(wgpu::RequestDeviceError),
    Surface(wgpu::SurfaceError),
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestAdapter => write!(f, "failed to request adapter"),
            Self::GetSurfaceConfig => write!(f, "failed to get surface configuration"),
            Self::GetTextureFormat => write!(f, "failed to get texture format"),
            Self::UninitRenderPipeline => write!(f, "uninitialized render pipeline"),
            Self::Wgpu(err) => write!(f, "wgpu error: {err}"),
            Self::CreateSurface(err) => write!(f, "create surface error: {err}"),
            Self::RequestDevice(err) => write!(f, "request device error: {err}"),
            Self::Surface(err) => write!(f, "surface error: {err}"),
            Self::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl From<wgpu::Error> for Error {
    fn from(err: wgpu::Error) -> Self {
        Self::Wgpu(err)
    }
}

impl From<wgpu::CreateSurfaceError> for Error {
    fn from(err: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(err)
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(err)
    }
}

impl From<wgpu::SurfaceError> for Error {
    fn from(err: wgpu::SurfaceError) -> Self {
        Self::Surface(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[allow(dead_code)]
#[repr(align(16))]
struct Vertex {
    coords: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const VERTEX_ATTR_ARRAY: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self as *const Self as *const u8, mem::size_of::<Self>()) }
    }

    fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::VERTEX_ATTR_ARRAY,
        }
    }
}

#[derive(Default)]
#[repr(align(8))]
struct Uniforms {
    mouse_coords: [f32; 2],
    surface_size: [f32; 2],
    time: f32,
}

impl Uniforms {
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self as *const Self as *const u8, mem::size_of::<Self>()) }
    }
}

pub struct ShaderPlayground<'a> {
    shader_path: PathBuf,
    device: wgpu::Device,
    queue: wgpu::Queue,
    wgpu_error: Arc<Mutex<Option<wgpu::Error>>>,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    texture_format: wgpu::TextureFormat,
    vertex_buffer: wgpu::Buffer,
    uniforms: Uniforms,
    uniforms_buffer: wgpu::Buffer,
    uniforms_bind_group: wgpu::BindGroup,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: Option<wgpu::RenderPipeline>,
    start_time: time::Instant,
}

impl<'a> ShaderPlayground<'a> {
    const VERTICES: [Vertex; 6] = [
        Vertex {
            coords: [-1., 1., 0.],
            color: [0., 1., 0.],
        },
        Vertex {
            coords: [-1., -1., 0.],
            color: [0., 1., 0.],
        },
        Vertex {
            coords: [1., -1., 0.],
            color: [0., 1., 0.],
        },
        Vertex {
            coords: [1., -1., 0.],
            color: [0., 1., 0.],
        },
        Vertex {
            coords: [1., 1., 0.],
            color: [0., 1., 0.],
        },
        Vertex {
            coords: [-1., 1., 0.],
            color: [0., 1., 0.],
        },
    ];

    pub async fn new<P: AsRef<Path>>(window: &'a Window, shader_path: P) -> Result<Self, Error> {
        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or(Error::RequestAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                },
                None,
            )
            .await?;

        let wgpu_error = Arc::new(Mutex::new(None));
        let handler_wgpu_error = Arc::clone(&wgpu_error);
        device.on_uncaptured_error(Box::new(move |err| {
            let mut wgpu_error = handler_wgpu_error.lock().expect("mutex lock");
            match wgpu_error.take() {
                Some(old_err) => panic!("unhandled error:\nold error: {old_err}\nnew error: {err}"),
                None => *wgpu_error = Some(err),
            }
        }));

        let size = window.inner_size();
        let surface_config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(Error::GetSurfaceConfig)?;
        surface.configure(&device, &surface_config);

        // The first texture format in the `formats` vector is
        // preferred.
        let texture_format = *surface
            .get_capabilities(&adapter)
            .formats
            .first()
            .ok_or(Error::GetTextureFormat)?;

        let vertex_buffer_contents: Vec<u8> = Self::VERTICES
            .iter()
            .flat_map(|v| v.as_bytes())
            .copied()
            .collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &vertex_buffer_contents,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniforms_bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut playground = ShaderPlayground {
            shader_path: shader_path.as_ref().to_owned(),
            device,
            queue,
            wgpu_error,
            surface,
            surface_config,
            texture_format,
            vertex_buffer,
            uniforms: Uniforms::default(),
            uniforms_buffer,
            uniforms_bind_group,
            pipeline_layout,
            render_pipeline: None,
            start_time: time::Instant::now(),
        };
        playground.update_render_pipeline()?;
        Ok(playground)
    }

    pub fn set_mouse_coords(&mut self, x: f32, y: f32) {
        self.uniforms.mouse_coords = [x, y];
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width.max(1);
        self.surface_config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
        self.uniforms.surface_size = [
            self.surface_config.width as f32,
            self.surface_config.height as f32,
        ];
    }

    pub fn update_render_pipeline(&mut self) -> Result<(), Error> {
        self.create_user_render_pipeline()
            .and_then(|pipeline| {
                self.render_pipeline = Some(pipeline);
                self.render()
            })
            .or_else(|err| {
                eprintln!("failed to set user render pipeline: {}", err);
                self.render_pipeline = Some(self.create_default_render_pipeline());
                self.render()
            })
    }

    fn create_default_render_pipeline(&self) -> wgpu::RenderPipeline {
        let shader_module = self
            .device
            .create_shader_module(include_wgsl!("default.wgsl"));
        self.create_render_pipeline(shader_module)
    }

    fn create_user_render_pipeline(&self) -> Result<wgpu::RenderPipeline, Error> {
        let shader_module = self.read_user_shader()?;
        Ok(self.create_render_pipeline(shader_module))
    }

    fn read_user_shader(&self) -> Result<wgpu::ShaderModule, Error> {
        let shader_code = fs::read_to_string(&self.shader_path)?;
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_code)),
            });

        match self.wgpu_error.lock().expect("mutex lock").take() {
            Some(err) => Err(err.into()),
            None => Ok(shader_module),
        }
    }

    fn create_render_pipeline(&self, shader_module: wgpu::ShaderModule) -> wgpu::RenderPipeline {
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vs_main",
                    buffers: &[Vertex::vertex_buffer_layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "fs_main",
                    targets: &[Some(self.texture_format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
    }

    pub fn render(&mut self) -> Result<(), Error> {
        let render_pipeline = self
            .render_pipeline
            .as_ref()
            .ok_or(Error::UninitRenderPipeline)?;

        self.uniforms.time = self.start_time.elapsed().as_secs_f32();

        self.queue
            .write_buffer(&self.uniforms_buffer, 0, self.uniforms.as_bytes());

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(render_pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_bind_group(0, &self.uniforms_bind_group, &[]);
            rpass.draw(0..Self::VERTICES.len() as u32, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }
}
