// utilities for drawing & interacting with regions in the match grid

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
) {
    let rect = region_to_screen_rect(screen_size, view, x_range, y_range);
    painter.rect_filled(rect, 0.0, color);
}

#[derive(Default)]
pub struct SelectionHandler {
    right_click_w_pos: Option<DVec2>,
}

impl SelectionHandler {
    pub fn run(&mut self, ctx: &egui::Context, view: &mut crate::view::View) {
        let (right_pressed, right_released, cur_pos) = ctx.input(|i| {
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
            let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
            draw_rect_region(
                &painter,
                screen_size,
                color,
                view,
                wp0.x..=wp1.x,
                wp0.y..=wp1.y,
            );
        }

        if right_released {
            if let Some(wp0) = self.right_click_w_pos.take() {
                let l = wp0.x.floor() as usize;
                let r = wp1.x.ceil() as usize;
                let u = wp0.y.floor() as usize;
                let d = wp1.y.ceil() as usize;

                *view = view.fit_ranges_in_view(
                    screen_size.x as f64 / screen_size.y as f64,
                    Some(l..r),
                    Some(u..d),
                );
            }
        }
    }
}
