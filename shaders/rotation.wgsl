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

const ROTATION_SPEED = 3.;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = uniforms.time;
    let uv = in.position.xy / uniforms.size.xy;

    let sin_theta = sin(ROTATION_SPEED * t);
    let cos_theta = cos(ROTATION_SPEED * t);
    let rotation = mat2x2<f32>(cos_theta, -sin_theta, sin_theta, cos_theta);
    let offset = vec2<f32>(0.5, 0.5);
    let uvr = (uv - offset) * rotation + offset;

    var color = vec4<f32>(1.);
    if (uvr.x > 0.3 && uvr.x < 0.7 && uvr.y > 0.3 && uvr.y < 0.7) {
        color = vec4<f32>(1., 0., 0., 1.);
    }

    return color;
}
