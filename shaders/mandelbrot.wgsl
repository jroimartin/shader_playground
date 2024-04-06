struct Uniforms {
    mouse: vec2<f32>,
    size: vec2<f32>,
    time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.);
    out.color = in.color;
    return out;
}

const MAX_ITERATIONS = 1000;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = uniforms.time;
    let uv = in.position.xy / uniforms.size.xy;
    let scale = 5.5/(10. + exp(10*abs(sin(t*0.2))));
    let offset = vec2<f32>(-.402-1./exp(scale), 0.);
    let x0 = (uv.x * 2.47 - 2.0) * scale + offset.x;
    let y0 = (uv.y * 2.24 - 1.12) * scale + offset.y;
    var x = 0.0;
    var y = 0.0;
    var i = 0;
    for (; x*x + y*y <= 2*2 && i < MAX_ITERATIONS; i++) {
        let xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
    }
    let color = f32(i)/f32(MAX_ITERATIONS);
    return vec4<f32>(abs(sin(0.66*t*color)), 0., 0., 1.0);
}
