#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

#import bevy_pbr::mesh_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color_u: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color_u: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities_u: u32;
@group(2) @binding(3) var<uniform> border_width_px_u: f32;
@group(2) @binding(4) var<uniform> border_width_modifiers_u: u32;

@vertex
// fn vertex(@builtin(vertex_index) vi: u32, vertex: Vertex) -> VOut {
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // This is how bevy computes the world position
    // The vertex.instance_index is very important. Esepecially if you are using batching and gpu preprocessing
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);

    return out;
}


fn compute_border_uv_width(
    uv: vec2f,
    border_width_px: f32,
    modifiers: vec4f,
) -> vec4f {
    let d_uv_dx = abs(dpdx(uv));
    let d_uv_dy = abs(dpdy(uv));

    let uv_screen_x = 1.0 / d_uv_dx.x;
    let uv_screen_y = 1.0 / d_uv_dy.y;



    let uv_width_x = border_width_px / uv_screen_x;
    let uv_width_y = border_width_px / uv_screen_y;

    let uvs = vec4f(vec2f(uv_width_x), vec2f(uv_width_y)) * modifiers;
    return uvs;

    // return vec2f(border_uv_width_x, border_uv_width_y);
}

fn compute_border_ts(
    uv: vec2f,
    border_width_uvs: vec4f,
    // border_width_uv: vec2f,
) -> vec4f {

    // let widths = border_width_uvs;
    // let widths = vec4f(
    //     border_width_uv.x,
    //     border_width_uv.x,
    //     border_width_uv.y,
    //     border_width_uv.y,
    // );

    // let dists = vec4f(
    //     uv.x - border_width_uv.x,
    //     abs((1.0 - uv.x)) - border_width_uv.x,
    //     uv.y - border_width_uv.y,
    //     abs((1.0 - uv.y)) - border_width_uv.y,
    // );
    let dists = vec4f(
        uv.x - border_width_uvs.x,
        abs((1.0 - uv.x)) - border_width_uvs.y,
        uv.y - border_width_uvs.z,
        abs((1.0 - uv.y)) - border_width_uvs.w,
    );

    // let normed = dists / vec4f(border_width_uv.xx, border_width_uv.yy);
    let normed = dists / border_width_uvs;
    let ts = clamp(normed, vec4f(0.0), vec4f(1.0));
    // let ts = smoothstep(vec4f(0.0), widths, dists);

    return ts;
}



@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let width_modifiers = unpack4x8unorm(border_width_modifiers_u);

    let border_width_uv = compute_border_uv_width(mesh.uv, border_width_px_u, width_modifiers);
    // let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u);
    let borders = unpack4x8unorm(border_opacities_u);

    let ts = compute_border_ts(mesh.uv, border_width_uv);
    // let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv);
    let ats = (vec4f(1.0) - ts) * borders;
    let t = max(max(ats.x, ats.y), max(ats.z, ats.w));

    let border_color = vec4f(border_color_u.rgb, t);
    return bordered_rect_util::over(border_color, fill_color_u);
}
