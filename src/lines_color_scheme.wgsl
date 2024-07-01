struct VertConfig {
  line_width: f32,
  brightness: f32,
  pad1: f32,
  pad2: f32,
}

struct ColorScheme {
 m_bg: vec4f,
 eq_bg: vec4f,
 x_bg: vec4f,
 i_bg: vec4f,
 d_bg: vec4f,
}

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@group(0) @binding(0) var<uniform> projection: mat4x4f;
@group(0) @binding(1) var<uniform> config: VertConfig;

@group(1) @binding(0) var<uniform> color_scheme: ColorScheme;

@group(2) @binding(0) var<uniform> model: mat4x4f;

@vertex
fn vs_main(
           @builtin(vertex_index) vertex_index: u32,
           @location(0) p0: vec2f,
           @location(1) p1: vec2f,
           @location(2) color: u32,
           // @location(2) color: u32,
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

  let x_basis = p1 - p0;
  let y_basis = normalize(vec2(-x_basis.y, x_basis.x));

  let view_width = 2.0 * projection[0][0];

  let pp = p0 + x_basis * pos.x + y_basis * (config.line_width / view_width) * pos.y;

  let transform = projection * model;

  result.position = transform * vec4(pp, 0.0, 1.0);
  result.position.z = 0.5;

  switch color % 5 {
      case 0u: {
        result.color = color_scheme.m_bg;
      }
      case 1u: {
        result.color = color_scheme.eq_bg;
      }
      case 2u: {
        result.color = color_scheme.x_bg;
      }
      case 3u: {
        result.color = color_scheme.i_bg;
      }
      case 4u: {
        result.color = color_scheme.d_bg;
      }
      default: {
        result.color = color_scheme.m_bg;
      }
  }

  return result;
}


struct FragmentOut {
  @location(0) color: vec4f,
}

@fragment
fn fs_main(
           @location(0) color: vec4f,
) -> FragmentOut {
  var result: FragmentOut;
  result.color = color;
  return result;
}
