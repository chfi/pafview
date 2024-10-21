#import bevy_sprite::{
    mesh2d_functions,
    mesh2d_vertex_output::VertexOutput,
}

#import bevy_sprite::mesh2d_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities: u32;
@group(2) @binding(3) var<uniform> border_width_px: f32;


@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_px);
    let t = min(min(ts.x, ts.y), min(ts.z, ts.w));
    let color = bordered_rect_util::compute_color(fill_color, border_color, t);
    return color;
}
