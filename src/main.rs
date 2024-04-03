use std::{env, path::Path, process};

use shader_playground::ShaderPlayground;
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::Key,
    window::WindowBuilder,
};

async fn run<P: AsRef<Path>>(shader_path: P) {
    let event_loop = EventLoop::new().expect("could not create even loop");
    let builder = WindowBuilder::new();
    let window = builder.build(&event_loop).expect("could not create window");

    let mut playground = ShaderPlayground::new(&window, shader_path)
        .await
        .expect("could not create shader playground");

    event_loop.set_control_flow(ControlFlow::Poll);

    playground.update_render_pipeline();
    event_loop
        .run(|event, elwt| match event {
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
                } if s == "r" => {
                    playground.update_render_pipeline();
                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    playground.set_mouse_coords(position.x as f32, position.y as f32);
                    window.request_redraw();
                }
                WindowEvent::Resized(new_size) => playground.resize(new_size),
                WindowEvent::RedrawRequested => {
                    playground.render().expect("could not render frame")
                }
                WindowEvent::CloseRequested => elwt.exit(),
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        })
        .unwrap();
}

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: shader_playground shader_file");
        process::exit(2);
    }
    let shader_path = &args[1];

    pollster::block_on(run(shader_path));
}
