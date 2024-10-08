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
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // This is how bevy computes the world position
    // The vertex.instance_index is very important. Esepecially if you are using batching and gpu preprocessing
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);

    // We just use the raw vertex color
    // out.color = vertex.color.rgb;

    return out;
}

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let screen_width = view.viewport.z;
    let screen_height = view.viewport.w;

    let fc_px = mesh.position;
    // let fc_px = mesh.uv;

    let r = fc_px.x / screen_width;
    let b = fc_px.y / screen_height;
    // let r = fc_px.x;
    // let b = fc_px.y;

    // let v_dist =
    return vec4f(r, 0.0, b, 1.0);
    // let edge_dist = min



}
