// utilities for drawing & interacting with regions in the match grid

use std::sync::Arc;

use egui::mutex::Mutex;
use ultraviolet::DVec2;

pub fn region_to_screen_rect(
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

pub fn draw_rect_region(
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

#[derive(Default)]
pub struct SelectionHandler {
    right_click_w_pos: Option<DVec2>,

    pub(super) selection_target: Option<SelectionTarget>,
}

// poorly named, but the target of a selection "requested" by another system
// -- when the SelectionHandler has "completed" a selection, the result is placed
// here for the "caller" to make use of
#[derive(Default, Clone)]
pub struct SelectionTarget {
    pub result: Arc<Mutex<Option<crate::view::View>>>,
}

impl SelectionHandler {
    pub fn has_active_selection_request(&self) -> bool {
        self.selection_target.is_some()
    }

    pub fn run(&mut self, ctx: &egui::Context, view: &mut crate::view::View) {
        let (right_pressed, right_released, cur_pos) = ctx.input(|i| {
            // let left_pressed = i.pointer.button_pressed(egui::PointerButton::Primary);
            // let left_rel = i.pointer.button_released(egui::PointerButton::Primary);
            let right_pressed = i.pointer.button_pressed(egui::PointerButton::Secondary);
            let right_rel = i.pointer.button_released(egui::PointerButton::Secondary);
            let cur_pos = i.pointer.latest_pos();
            (right_pressed, right_rel, cur_pos)
        });

        let Some(p1) = cur_pos else {
            return;
        };

        let screen_size = ctx.screen_rect().size();

        let sp1: [f32; 2] = p1.into();
        let wp1 = view.map_screen_to_world(screen_size, sp1);

        if right_pressed && self.right_click_w_pos.is_none() {
            self.right_click_w_pos = Some(wp1);
        }

        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            "selection".into(),
        ));

        if let Some(wp0) = self.right_click_w_pos {
            let l = wp0.x.min(wp1.x);
            let r = wp0.x.max(wp1.x);
            let u = wp0.y.min(wp1.y);
            let d = wp0.y.max(wp1.y);

            let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
            draw_rect_region(&painter, screen_size, color, view, l..=r, u..=d);

            let l = l.floor() as u64;
            let r = r.ceil() as u64;
            let u = u.floor() as u64;
            let d = d.ceil() as u64;

            if right_released {
                // TODO should use left button here; factor out & improve
                let new_view = view.fit_ranges_in_view_with_aspect(
                    screen_size.x as f64 / screen_size.y as f64,
                    Some(l..r),
                    Some(u..d),
                );

                if let Some(target) = self.selection_target.take() {
                    let mut result = target.result.lock();
                    *result = Some(new_view);
                } else {
                    *view = new_view;
                }
                self.right_click_w_pos.take();
            }
        }
    }
}
