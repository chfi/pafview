struct VertConfig {
  line_width: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@group(0) @binding(0) var<uniform> projection: mat4x4f;
@group(0) @binding(1) var<uniform> config: VertConfig;

@group(1) @binding(0) var<uniform> model: mat4x4f;


@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) p0: vec2f,
    @location(1) p1: vec2f,
    @location(2) z: f32,
    @location(3) color_packed: u32,
) -> VertexOut {
    var result: VertexOut;

    let i = vertex_index % 6u;

    var pos: vec2f;

    switch i {
        case 0u: {
        pos = vec2(0.0, -0.5);
        }
        case 1u: {
        pos = vec2(1.0, -0.5);
        }
        case 2u: {
        pos = vec2(1.0, 0.5);
        }
        case 3u: {
        pos = vec2(0.0, -0.5);
        }
        case 4u: {
        pos = vec2(1.0, 0.5);
        }
        default: {
        pos = vec2(0.0, 0.5);
        }
    }

    let view_width = 2.0 * projection[0][0];

    let s0 = model * vec4(p0, 0.0, 1.0);
    let s1 = model * vec4(p1, 0.0, 1.0);

    let x_basis = s1.xy - s0.xy;
    let y_basis = normalize(vec2(-x_basis.y, x_basis.x));

    let sp = s0.xy + x_basis * pos.x + y_basis * (config.line_width / view_width) * pos.y;
    let transform = projection;

    result.position = projection * vec4(sp, 0.0, 1.0);
    result.position.z = z;

    // let color_ix = color % 5;
    result.color = unpack4x8unorm(color_packed);

    /*
    switch color_ix {
        case 0u: {
        result.color = color_scheme.m_bg;
        result.position.z = 0.5;
        }
        case 1u: {
        result.color = color_scheme.eq_bg;
        result.position.z = 0.6;
        }
        case 2u: {
        result.color = color_scheme.x_bg;
        result.position.z = 0.7;
        }
        case 3u: {
        result.color = color_scheme.i_bg;
        result.position.z = 0.1;
        }
        case 4u: {
        result.color = color_scheme.d_bg;
        result.position.z = 0.1;
        }
        default: {
        result.color = color_scheme.m_bg;
        result.position.z = 0.5;
        }
    }
    */

    return result;
}


@fragment
fn fs_main(
    @location(0) color: vec4f,
) -> @location(0) vec4f {
    return color;
}
