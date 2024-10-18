#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

#import bevy_pbr::mesh_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities: u32;
@group(2) @binding(3) var<uniform> border_width_px: f32;

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

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let screen_dims = view.viewport.zw;
    let screen_width = view.viewport.z;
    let screen_height = view.viewport.w;

    let fc_px = mesh.position;

    let fw = fwidth(mesh.uv);

    let uv = mesh.uv;
    let min_x = min(uv.x, 1.0 - uv.x);
    let min_y = min(uv.y, 1.0 - uv.y);

    let border = fw * border_width_px;
    let border_px = min(border.x, border.y);
    let min_px = min(min_x, min_y);

    let t = smoothstep(0.0, border_px, min_px);

    let c_a = fill_color.rgb;
    let a_a = fill_color.a;
    let c_b = border_color.rgb;
    let a_b = border_color.a;

    let c_o = c_a + c_b * (1.0 - t);
    let a_o = a_a + a_b * (1.0 - t);

    let color = vec4f(c_o, a_o);

    return color;

}
