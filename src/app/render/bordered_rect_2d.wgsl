#import bevy_sprite::{
    mesh2d_functions,
    mesh2d_vertex_output::VertexOutput,
}

#import bevy_sprite::mesh2d_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color_u: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color_u: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities_u: u32;
@group(2) @binding(3) var<uniform> border_width_px_u: f32;


@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {

    let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u);
    let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv);

    let border_alpha = bordered_rect_util::compute_alpha_for_border(
        mesh.uv,
        unpack4x8unorm(border_opacities_u),
        border_width_uv
    );
    // let border_alpha = 1.0;
    let border_color = vec4f(border_color_u.rgb, border_alpha);
    // let border_color = vec4f(border_color_u.rgba);

    let t = min(min(ts.x, ts.y), min(ts.z, ts.w));
    let color = bordered_rect_util::compute_color(fill_color_u, border_color, t);
    return color;
}
