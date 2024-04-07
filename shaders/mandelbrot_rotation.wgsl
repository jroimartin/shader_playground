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

const MAX_ITERATIONS = 1500;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = uniforms.time;
    let uv = in.position.xy/uniforms.size.xy;

    let scale = 25./(10. + exp(10*abs(sin(t*0.2))));

    let rot_speed = 1/exp(scale/(25./10. + exp(10.))*100.);
    let sin_theta = sin(rot_speed*t);
    let cos_theta = cos(rot_speed*t);
    let rotation = mat2x2<f32>(cos_theta, -sin_theta, sin_theta, cos_theta);
    let rot_point = vec2<f32>(0.5, 0.5);
    let uvr = (uv - rot_point)*rotation + rot_point;

    let offset = vec2<f32>(-2.5 + 20./(17.2 + exp(scale)), -scale/2.);
    var uv0 = uvr*scale + offset;
    var x = 0.0;
    var y = 0.0;
    var i = 0;
    for (; x*x + y*y <= 2*2 && i < MAX_ITERATIONS; i++) {
        let xtemp = x*x - y*y + uv0.x;
        y = 2*x*y + uv0.y;
        x = xtemp;
    }
    let color = f32(i)/f32(MAX_ITERATIONS);
    return vec4<f32>(abs(sin(t*color)), 0., 0., 1.0);
}
