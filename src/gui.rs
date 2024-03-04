use egui::{util::IdTypeMap, DragValue, Ui};
use ultraviolet::{Mat4, Vec2};

use crate::{view::View, PafInput};

pub(crate) fn view_controls(input: &PafInput, view: &mut View, ctx: &egui::Context) {
    egui::Window::new("View")
        // .resizable(true)
        // .vscroll(true)
        // .default_open(false)
        .show(&ctx, |mut ui| {
            let x_limit = input.target_len as f64;
            let y_limit = input.query_len as f64;

            let v = *view;

            let x_min =
                DragValue::new(&mut view.x_min).clamp_range(0f64..=(v.x_max - 1.0).max(0.0));
            let x_max = DragValue::new(&mut view.x_max).clamp_range((v.x_min + 1.0)..=x_limit);

            ui.horizontal(|ui| {
                ui.add(x_min);
                ui.add(x_max);
            });

            let y_min =
                DragValue::new(&mut view.y_min).clamp_range(0f64..=(v.y_max - 1.0).max(0.0));
            let y_max = DragValue::new(&mut view.y_max).clamp_range((v.y_min + 1.0)..=y_limit);

            ui.horizontal(|ui| {
                ui.add(y_min);
                ui.add(y_max);
            });
            //

            let (cursor_pos, screen_dims) = ctx.input(|i| {
                let pos = i.pointer.hover_pos();
                let dims = i.screen_rect.size();
                (pos, [dims.x as u32, dims.y as u32])
            });

            // ui.max_rect()

            if let Some(p) = cursor_pos {
                let sp: [f32; 2] = p.to_vec2().into();
                ui.label(format!("Cursor (screen) {p:?}"));

                let view_pos = view.map_screen_to_view(screen_dims, sp);
                let world_pos = view.map_screen_to_world(screen_dims, sp);

                ui.label(format!("Cursor (view): {view_pos:?}"));
                ui.label(format!("Cursor (world): {world_pos:?}"));
            }
        });
}
