#define_import_path bordered_rect_util


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

    let ts = smoothstep(vec4f(0.0), widths, dists);

    return ts;
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

    return alpha;
}


fn compute_color(
    fill_color: vec4f,
    border_color: vec4f,
    t: f32,
) -> vec4f {
    let c_a = fill_color.rgb;
    let a_a = fill_color.a;
    let c_b = border_color.rgb;
    let a_b = border_color.a;

    let c_o = c_a + c_b * (1.0 - t);
    let a_o = a_a + a_b * (1.0 - t);

    return vec4f(c_o, a_o);
}
