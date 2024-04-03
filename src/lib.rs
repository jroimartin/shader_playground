use std::{
    borrow::Cow,
    fmt, fs, io, mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use wgpu::{include_wgsl, util::DeviceExt};
use winit::{dpi::PhysicalSize, window::Window};

#[derive(Debug)]
pub enum Error {
    RequestAdapter,
    GetSurfaceConfig,
    GetTextureFormat,
    Wgpu(wgpu::Error),
    RequestDevice(wgpu::RequestDeviceError),
    Surface(wgpu::SurfaceError),
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RequestAdapter => write!(f, "failed to request adapter"),
            Error::GetSurfaceConfig => write!(f, "failed to get surface configuration"),
            Error::GetTextureFormat => write!(f, "failed to get texture format"),
            Error::Wgpu(err) => write!(f, "wgpu error: {err}"),
            Error::RequestDevice(err) => write!(f, "request device error: {err}"),
            Error::Surface(err) => write!(f, "surface error: {err}"),
            Error::Io(err) => write!(f, "i/o error: {err}"),
        }
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        Error::RequestDevice(err)
    }
}

impl From<wgpu::SurfaceError> for Error {
    fn from(err: wgpu::SurfaceError) -> Self {
        Error::Surface(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Io(err)
    }
}

struct Vertex {
    coords: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 2 * mem::size_of::<[f32; 3]>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

impl Vertex {
    fn to_bytes(&self) -> Vec<u8> {
        self.coords
            .iter()
            .chain(self.color.iter())
            .flat_map(|c| c.to_ne_bytes())
            .collect()
    }
}

pub struct ShaderPlayground<'a> {
    shader_path: PathBuf,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'a>,
    surface_config: wgpu::SurfaceConfiguration,
    texture_format: wgpu::TextureFormat,
    vertex_buffer: wgpu::Buffer,
    wgpu_error: Arc<Mutex<Option<wgpu::Error>>>,
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

        let surface = instance.create_surface(window).unwrap();
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
            *handler_wgpu_error.lock().expect("mutex lock") = Some(err);
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

        let vertex_buffer_contents: Vec<u8> =
            Self::VERTICES.iter().flat_map(|v| v.to_bytes()).collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &vertex_buffer_contents,
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(ShaderPlayground {
            shader_path: shader_path.as_ref().to_owned(),
            device,
            queue,
            surface,
            surface_config,
            texture_format,
            vertex_buffer,
            wgpu_error,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.surface_config.width = new_size.width.max(1);
        self.surface_config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn render(&self) -> Result<(), Error> {
        self.render_user_shader().or_else(|err| {
            eprintln!("failed to render user shader: {err}");
            self.render_default_shader()
        })
    }

    fn render_user_shader(&self) -> Result<(), Error> {
        self.render_with_shader_module(self.shader_module()?)
    }

    fn render_default_shader(&self) -> Result<(), Error> {
        let shader_module = self
            .device
            .create_shader_module(include_wgsl!("default.wgsl"));
        self.render_with_shader_module(shader_module)
    }

    fn render_with_shader_module(&self, shader_module: wgpu::ShaderModule) -> Result<(), Error> {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                // TODO(rm): add mouse coordinates uniform.
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
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
            });

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

            rpass.set_pipeline(&render_pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.draw(0..Self::VERTICES.len() as u32, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn shader_module(&self) -> Result<wgpu::ShaderModule, Error> {
        let shader_code = fs::read_to_string(&self.shader_path)?;
        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_code)),
            });

        match self.wgpu_error.lock().expect("mutex lock").take() {
            Some(err) => Err(Error::Wgpu(err)),
            None => Ok(shader_module),
        }
    }
}
