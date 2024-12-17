#import bevy_sprite::{
    mesh2d_functions,
    mesh2d_vertex_output::VertexOutput,
}

#import bevy_sprite::mesh2d_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color_u: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color_u: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities_u: u32;
@group(2) @binding(3) var<uniform> border_width_px_u: f32;
@group(2) @binding(4) var<uniform> border_width_modifiers_u: u32;





@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u);
    let borders = unpack4x8unorm(border_opacities_u);

    let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv);
    let ats = (vec4f(1.0) - ts) * borders;
    let t = max(max(ats.x, ats.y), max(ats.z, ats.w));

    let border_color = vec4f(border_color_u.rgb, t);
    return bordered_rect_util::over(border_color, fill_color_u);
}
