#define_import_path bordered_rect_util


fn compute_border_ts(
    uv: vec2f,
    border_width_px: f32,
) -> vec4f {
    let d_uv_dx = abs(dpdx(uv));
    let d_uv_dy = abs(dpdy(uv));

    let uv_screen_x = 1.0 / d_uv_dx.x;
    let uv_screen_y = 1.0 / d_uv_dy.y;

    let border_uv_width_x = border_width_px / uv_screen_x;
    let border_uv_width_y = border_width_px / uv_screen_y;

    let widths = vec4f(
        border_uv_width_x,
        border_uv_width_x,
        border_uv_width_y,
        border_uv_width_y,
    );

    let dists = vec4f(
        abs(uv.x - border_uv_width_x),
        abs((1.0 - uv.x) - border_uv_width_x),
        abs(uv.y - border_uv_width_y),
        abs((1.0 - uv.y) - border_uv_width_y),
    );

    let ts = smoothstep(vec4f(0.0), widths, dists);

    return ts;
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
