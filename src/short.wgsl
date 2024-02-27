struct Config {
  // in NDC
  frag_width: f32,
  frag_height: f32,
  pad0: f32,
  pad1: f32,
}

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
}

@group(0) @binding(0) var<uniform> projection: mat4x4f;
@group(0) @binding(1) var<uniform> config: Config;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32,
           @location(0) p0: vec2f,
           @location(1) p1: vec2f,
           ) -> VertexOut {
  
  // p0 & p1 are known to be "close" in the active view;
  // otherwise the other renderer would be used

  var result: VertexOut;

  let i = vertex_index % 6u;

  let xs = config.frag_width;
  let ys = config.frag_height;

  var pos: vec2f;

  switch i {
      case 0u: {
        pos = vec2(-xs, -ys);
      }
      case 1u: {
        pos = vec2(xs, -ys);
      }
      case 2u: {
        pos = vec2(xs, ys);
      }
      case 3u: {
        pos = vec2(-xs, -ys);
      }
      case 4u: {
        pos = vec2(xs, ys);
      }
      default: {
        pos = vec2(-xs, ys);
      }
    }
  
  var center = vec4(p0 + 0.5 * (p1 - p0), 0.0, 1.0);
  var delta = vec4(pos, 0.0, 0.0);
  
  result.position = delta + projection * center;
  result.position.z = 0.5;
  
  // TODO compute color based on length compared to fragment size
  // let color_u = (vec4u(config.color) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(255u);
  // result.color = vec4f(color_u) / vec4f(255.0);
  result.color = vec4f(0.0, 0.0, 0.0, 1.0);
  
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
