// utilities for drawing & interacting with regions in the match grid

use ultraviolet::DVec2;

pub fn draw_rect_region(
    painter: &egui::Painter,
    window_dims: [u32; 2],
    view: &crate::view::View,
    x_range: std::ops::RangeInclusive<f64>,
    y_range: std::ops::RangeInclusive<f64>,
) {
    let x0 = *x_range.start();
    let y0 = *y_range.start();

    let x1 = *x_range.end();
    let y1 = *y_range.end();

    let p0 = DVec2::new(x0, y0);
    let p1 = DVec2::new(x1, y1);

    let s0: [f32; 2] = view.map_world_to_screen(window_dims, p0).into();
    let s1: [f32; 2] = view.map_world_to_screen(window_dims, p1).into();

    let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);

    let rect = egui::Rect::from_two_pos([100.0, 100.0].into(), [400.0, 400.0].into());
    // let rect = egui::Rect::from_two_pos(s0.into(), s1.into());

    log::warn!("drawing rect {rect:?}");

    painter.rect_filled(rect, 0.0, color);
}
