#import bevy_pbr::{
    mesh_functions,
    forward_io::{Vertex, VertexOutput},
    view_transformations::position_world_to_clip,
}

// i *think* i can use this to get the screen size in pixels
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

    // We just use the raw vertex color
    // out.color = vertex.color.rgb;

    // let vmod = vi % 3;
    // var vout: VOut;
    // vout.vertex = out;
    // vout.bary = vec3f((vmod == 0) as f32, (vmod == 1) as f32, (vmod == 2) as f32);



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
    // let min_x = min(uv.x * xydx.x, 1.0 - uv.x * xydx.x);
    // let min_y = min(uv.y * xydy.y, 1.0 - uv.y * xydy.y);

    let border = fw * border_width_px;

    // let border = min(fw.x, fw.y) * border_width_px;

    var color = fill_color;
    if min_x < border.x || min_y < border.y {
        color = border_color;
    }

    return color;

}
