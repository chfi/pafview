struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> u_projection: mat4x4f;

@vertex
fn vs_main(
           @builtin(vertex_index) vertex_index: u32,
           ) -> VertexOut {

  var uv: vec2f;
  var pos: vec2f;
  switch i {
      case 0u: {
        pos = vec2(0.0, 0.0);
        uv = vec2(0.0, 0.0);
      }
      case 1u: {
        pos = vec2(1.0, 0.0);
        uv = vec2(1.0, 0.0);
      }
      case 2u: {
        pos = vec2(1.0, 1.0);
        uv = vec2(1.0, 1.0);
      }
      case 3u: {
        pos = vec2(0.0, 0.0);
        uv = vec2(0.0, 0.0);
      }
      case 4u: {
        pos = vec2(1.0, 1.0);
        uv = vec2(1.0, 1.0);
      }
      default: {
        pos = vec2(0.0, 1.0);
        uv = vec2(0.0, 1.0);
      }
  }

  result.position = u_projection * vec4(pos, 0.5, 1.0);
  result.uv = uv;

}


@group(1) @binding(0)
var u_texture: texture_2d<vec4f>;

@group(1) @binding(1)
var u_sampler: sampler;

struct FragmentOut {
  @location(0) color: vec4f,
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let color = textureSample(u_color, u_texture, uv);
  return color;
}
