#import bevy_sprite::{
    mesh2d_functions,
    mesh2d_vertex_output::VertexOutput,
}

#import bevy_sprite::mesh2d_view_bindings::view;

@group(2) @binding(0) var<uniform> fill_color_u: vec4<f32>;
@group(2) @binding(1) var<uniform> border_color_u: vec4<f32>;
@group(2) @binding(2) var<uniform> border_opacities_u: u32;
@group(2) @binding(3) var<uniform> border_width_px_u: f32;


fn compute_border_uv_width(
    uv: vec2f,
    border_width_px: f32,
) -> vec2f {
    let d_uv_dx = abs(dpdx(uv));
    let d_uv_dy = abs(dpdy(uv));

    let uv_screen_x = 1.0 / d_uv_dx.x;
    let uv_screen_y = 1.0 / d_uv_dy.y;

    let border_uv_width_x = border_width_px / uv_screen_x;
    let border_uv_width_y = border_width_px / uv_screen_y;

    return vec2f(border_uv_width_x, border_uv_width_y);
}


fn compute_border_ts(
    uv: vec2f,
    border_width_uv: vec2f,
    // border_alphas: vec4f,
) -> vec4f {

    let widths = vec4f(
        border_width_uv.x,
        border_width_uv.x,
        border_width_uv.y,
        border_width_uv.y,
    );

    let dists = vec4f(
        uv.x - border_width_uv.x,
        abs((1.0 - uv.x)) - border_width_uv.x,
        uv.y - border_width_uv.y,
        abs((1.0 - uv.y)) - border_width_uv.y,
    );

    let normed = dists / vec4f(border_width_uv.xx, border_width_uv.yy);
    let ts = clamp(normed, vec4f(0.0), vec4f(1.0));
    // let ts = smoothstep(vec4f(0.0), widths, dists);

    return ts;
    // return ts * border_alphas;
}


fn compute_alpha_for_border(
    uv: vec2f,
    border_alphas: vec4f,
    border_width_uv: vec2f,
) -> f32 {
    var alpha = 0.0;

    if uv.x < border_width_uv.x {
        alpha = max(alpha, border_alphas.w);
    }
    if uv.x > 1.0 - border_width_uv.x {
        alpha = max(alpha, border_alphas.y);
    }

    if uv.y < border_width_uv.y {
        alpha = max(alpha, border_alphas.x);
    }
    if uv.y > 1.0 - border_width_uv.y {
        alpha = max(alpha, border_alphas.z);
    }

    // return alpha;
    return clamp(alpha, 0.0, 1.0);
}

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {

    // let border_width_px = border_width_px_u * 10.0;
    let border_width_px = 10.0;

    let border_width_uv = compute_border_uv_width(mesh.uv, border_width_px);
    let ts = compute_border_ts(mesh.uv, border_width_uv);

    let borders = unpack4x8unorm(border_opacities_u);

    let on_left = mesh.uv.x < 0.5;
    let on_top = mesh.uv.y < 0.5;

    // var border_alpha = 0.0;


    let h_t = select(ts.w, ts.y, on_left);
    let v_t = select(ts.x, ts.z, on_top);

    let h_alpha = select(borders.w, borders.y, on_left);
    let v_alpha = select(borders.x, borders.z, on_top);

    let border_alpha = max(h_alpha, v_alpha);
    // let t = max(h_t, v_t);
    let t = min(min(ts.x, ts.y), min(ts.z, ts.w));

    // let border_alpha =
    // let border_alpha = compute_alpha_for_border(
    //     mesh.uv,
    //     unpack4x8unorm(border_opacities_u),
    //     border_width_uv
    // );

    // let border_color = vec4f(border_color_u.rgb, border_alpha);

    // let t = min(min(ts.x, ts.y), min(ts.z, ts.w));
    // let color = compute_color(fill_color_u, border_color, t);
    // return color;
    return vec4(t, 0.0, 1.0 - t, 1.0);
    // return vec4(border_alpha, 0.0, 1.0 - border_alpha, 1.0);
}

/*
// @fragment
fn fragment_old(
    mesh: VertexOutput,
) -> /*@location(0)*/ vec4<f32> {

    // let border_width_uv = bordered_rect_util::compute_border_uv_width(mesh.uv, border_width_px_u);
    let border_width_uv = compute_border_uv_width(mesh.uv, 10.0);

    let border_opacities = unpack4x8unorm(border_opacities_u);

    // let ts = compute_border_ts(mesh.uv, border_width_uv);
    let ts = compute_border_ts(mesh.uv, border_width_uv, border_opacities);


    let on_left = mesh.uv.x < 0.5;
    let on_top = mesh.uv.y < 0.5;

    // let on_left = mesh.uv.x < border_width_uv.x;
    // let on_top = mesh.uv.y < border_width_uv.y;





    var border_alpha = 1.0;

    // if mesh.uv.x < 1.0 - mesh.uv.x {
    //     if mesh.uv.y < 1.0 - mesh.uv.y {
    //         use

    //     } else {

    //     }
    // } else {

    // }

    if on_left {
        if on_top {
            if mesh.uv.y < mesh.uv.x {
                border_alpha = ts.y;
            } else {
                border_alpha = ts.w;
            }
        } else {
            if mesh.uv.y < mesh.uv.x {
                border_alpha = ts.x;
            } else {
                border_alpha = ts.z;
            }
        }
    } else {
        if on_top {
            if mesh.uv.y > mesh.uv.x {
                border_alpha = ts.y;
            } else {
                border_alpha = ts.w;
            }
        } else {
            if mesh.uv.y < mesh.uv.x {
                border_alpha = ts.x;
            } else {
                border_alpha = ts.z;
            }
        }
    }

    // let closest_h = select(ts.w, ts.y, on_left);
    // let closest_v = select(ts.x, ts.z, on_top);

    // let ba1 = select(ts.xy, ts.zw, on_left);
    // let ba2 = select(ba1.x, ba1.y, on_top);

    // let border_alpha = ba2;


    /*
    if mesh.uv.x < border_width_uv.x {
        if mesh.uv.y < border_width_uv.y {
            // border_opacities.x * ts.x
            border_alpha = ts.x;
        } else {
            // border_opacities.y * ts.y
            border_alpha = ts.y;
        }
    } else {
        if mesh.uv.y < border_width_uv.y {
            // border_opacities.z * ts.z
            border_alpha = ts.z;

        } else {
            // border_opacities.w * ts.w
            border_alpha = ts.w;
        }
    };
    */
    // let border_alpha = min(min(ts.x, ts.y), min(ts.z, ts.w));

    // let border_alpha = compute_alpha_for_border(
    //     mesh.uv,
    //     unpack4x8unorm(border_opacities_u),
    //     border_width_uv
    // );
    return vec4(border_alpha, 0.0, 1.0 - border_alpha, 1.0);
    /*
    // let border_alpha = 1.0;
    let border_color = vec4f(border_color_u.rgb, border_alpha);
    // let border_color = vec4f(border_color_u.rgba);

    let t = min(min(ts.x, ts.y), min(ts.z, ts.w));
    let color = bordered_rect_util::compute_color(fill_color_u, border_color, t);
    return color;
     */
}
*/
