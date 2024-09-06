// utilities for drawing & interacting with regions in the match grid

use ultraviolet::DVec2;

pub(crate) fn region_to_screen_rect(
    screen_size: impl Into<[f32; 2]>,
    view: &crate::view::View,
    x_range: std::ops::RangeInclusive<f64>,
    y_range: std::ops::RangeInclusive<f64>,
) -> egui::Rect {
    let screen_size = screen_size.into();
    let x0 = *x_range.start();
    let y0 = *y_range.start();

    let x1 = *x_range.end();
    let y1 = *y_range.end();

    let p0 = DVec2::new(x0, y0);
    let p1 = DVec2::new(x1, y1);

    let s0: [f32; 2] = view.map_world_to_screen(screen_size, p0).into();
    let s1: [f32; 2] = view.map_world_to_screen(screen_size, p1).into();

    egui::Rect::from_two_pos(s0.into(), s1.into())
}

pub(crate) fn draw_rect_region(
    painter: &egui::Painter,
    screen_size: impl Into<[f32; 2]>,
    color: impl Into<egui::Color32>,
    view: &crate::view::View,
    x_range: std::ops::RangeInclusive<f64>,
    y_range: std::ops::RangeInclusive<f64>,
) -> egui::Rect {
    let rect = region_to_screen_rect(screen_size, view, x_range, y_range);
    painter.rect_filled(rect, 0.0, color);
    rect
}

/*
pub fn paf_line_debug_aabbs(
    input: &crate::PafInput,
    ctx: &egui::Context,
    view: &crate::view::View,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Background,
        "line-aabb".into(),
    ));

    let screen_size = ctx.screen_rect().size();

    let cursor = ctx.input(|i| i.pointer.latest_pos());
    let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);

    let mut draw_text = None;

    for line in &input.processed_lines {
        let aabb_min = line.aabb_min;
        let aabb_max = line.aabb_max;
        let x_range = aabb_min.x..=aabb_max.x;
        let y_range = aabb_min.y..=aabb_max.y;
        let rect = draw_rect_region(&painter, screen_size, color, view, x_range, y_range);

        if let Some(pos) = cursor {
            if rect.contains(pos) {
                let tgt = &input.targets[line.target_id.0].name;
                let qry = &input.queries[line.query_id.0].name;
                draw_text = Some((rect.left_bottom(), format!("{qry}/{tgt}")));
            }
        }
    }

    if let Some((pos, text)) = draw_text {
        painter.text(
            pos,
            egui::Align2::LEFT_BOTTOM,
            text,
            egui::FontId::monospace(12.0),
            egui::Color32::BLACK,
        );
    }
}
*/
