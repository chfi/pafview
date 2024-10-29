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

fn over(
    a: vec4f,
    b: vec4f,
) -> vec4f {
    let c_a = a.rgb;
    let a_a = a.a;
    let c_b = b.rgb;
    let a_b = b.a;

    let c_o = c_a + c_b * (1.0 - a_a);
    let a_o = a_a + a_b * (1.0 - a_a);

    return vec4f(c_o, a_o);
}

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    // let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_px);
    // let t = min(min(ts.x, ts.y), min(ts.z, ts.w));
    // let color = bordered_rect_util::compute_color(fill_color, border_color, t);
    // return color;

    let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u);

    var color = vec4f(1.0);

    let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv);
    let t_ = min(min(ts.x, ts.y), min(ts.z, ts.w));
    let t = 1.0 - t_;

    color = over(vec4f(vec3f(0.0), t), vec4f(1.0));

    /*
    let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv);

    // let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u * 10.0);
    // let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv * 0.5);

    // let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u * 10.0);
    // let ts = bordered_rect_util::compute_border_ts(mesh.uv, border_width_uv * 0.5);

    let border_alpha = bordered_rect_util::compute_alpha_for_border(
        mesh.uv,
        unpack4x8unorm(border_opacities_u),
        border_width_uv
    );
    let border_color = vec4f(border_color_u.rgb, border_alpha);
    // let border_color = vec4f(border_color_u.rgba);

    let t_ = min(min(ts.x, ts.y), min(ts.z, ts.w));
    let t = border_alpha * t_;

    let color = bordered_rect_util::compute_color(fill_color_u, border_color, t);
    */
    return color;
}
