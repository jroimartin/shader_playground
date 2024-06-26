use std::{
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use clap::Parser;
use shader_playground::ShaderPlayground;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::Key,
    window::WindowBuilder,
};

async fn run<P, T>(shader: P, textures: T)
where
    P: AsRef<Path>,
    T: IntoIterator<Item = P>,
{
    let event_loop = EventLoop::new().expect("could not create even loop");
    let window = WindowBuilder::new()
        .with_title("Shader Playground")
        .build(&event_loop)
        .expect("could not create window");

    let mut playground = ShaderPlayground::new(&window, shader, textures)
        .await
        .expect("could not create shader playground");

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(|event, elwt| {
            let ti = Instant::now();
            match event {
                Event::WindowEvent {
                    window_id: _,
                    event,
                } => match event {
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Character(s),
                                ..
                            },
                        ..
                    } if s == "r" => playground
                        .update_render_pipeline()
                        .expect("could not update render pipeline"),
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Character(s),
                                ..
                            },
                        ..
                    } if s == "t" => {
                        playground.set_start_time(Instant::now());
                        window.request_redraw();
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        playground.set_mouse_coords(position.x as f32, position.y as f32);
                        window.request_redraw();
                    }
                    WindowEvent::Resized(new_size) => playground.resize(new_size),
                    WindowEvent::RedrawRequested => {
                        playground.render().expect("could not render frame");
                        thread::sleep(
                            Duration::from_secs_f64(1. / 60.).saturating_sub(ti.elapsed()),
                        );
                    }
                    WindowEvent::CloseRequested => elwt.exit(),
                    _ => {}
                },
                Event::AboutToWait => window.request_redraw(),
                _ => {}
            }
        })
        .unwrap();
}

#[derive(Parser)]
struct Args {
    /// Path of the shader file.
    shader: PathBuf,

    /// Path of the texture shared with the shader.
    #[arg(id = "texture", short, long)]
    textures: Vec<PathBuf>,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    pollster::block_on(run(&args.shader, &args.textures));
}
