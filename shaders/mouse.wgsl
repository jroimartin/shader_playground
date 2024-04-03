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
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec4<f32>(0.1, 0.1, 0.1, 1.);

    let d = length(in.clip_position.xy - uniforms.mouse.xy);
    let r = min(uniforms.size.x, uniforms.size.y) / 4. * abs(cos(uniforms.time));

    if d < r {
        color = vec4<f32>(1.);
    }

    return color;
}
