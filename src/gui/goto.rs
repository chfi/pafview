use bimap::BiMap;
use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::grid::AxisRange;
use crate::math_conv::*;
use crate::{annotations::AnnotationStore, sequences::SeqId, view::View, AlignedSeq, PafViewerApp};

pub fn goto_region_window(
    ctx: &egui::Context,
    open: &mut bool,
    alignment_grid: &crate::grid::AlignmentGrid,
    view: &mut View,
) {
    egui::Window::new("Go to region")
        .open(open)
        .show(ctx, |ui| {
            let target_id = ui.id().with("target-range");
            let query_id = ui.id().with("query-range");

            let (mut target_buf, mut query_buf) = ui
                .data(|data| {
                    let t = data.get_temp::<String>(target_id)?;
                    let q = data.get_temp::<String>(query_id)?;
                    Some((t, q))
                })
                .unwrap_or_default();

            let (pressed_enter, window_dims, aspect_ratio) = ui.input(|i| {
                let pressed = i.key_pressed(egui::Key::Enter);
                let rect = i.screen_rect();
                let dims = [rect.width() as u32, rect.height() as u32];
                let aspect = rect.width() as f64 / rect.height() as f64;
                (pressed, dims, aspect)
            });

            // Target/X

            let goto_btn = ui.button("Go to range");

            let mut goto = goto_btn.clicked();

            ui.horizontal(|ui| {
                ui.label("Target");
                let target_text = ui.text_edit_singleline(&mut target_buf);
                goto |= target_text.lost_focus() && pressed_enter;
            });

            // Query/Y

            ui.horizontal(|ui| {
                ui.label("Query");
                let query_text = ui.text_edit_singleline(&mut query_buf);
                goto |= query_text.lost_focus() && pressed_enter;
            });

            let x_range = crate::grid::parse_axis_range_into_global(
                &alignment_grid.sequence_names,
                &alignment_grid.x_axis,
                &target_buf,
            )
            .and_then(AxisRange::to_global);
            let y_range = crate::grid::parse_axis_range_into_global(
                &alignment_grid.sequence_names,
                &alignment_grid.y_axis,
                &query_buf,
            )
            .and_then(AxisRange::to_global);

            let layer =
                egui::LayerId::new(egui::Order::Background, egui::Id::new("region-painter"));
            let painter = ui.ctx().layer_painter(layer);

            let screen_size = ui.ctx().screen_rect().size();

            // if goto_btn.hovered() {
            if let Some((x, y)) = x_range.as_ref().zip(y_range.as_ref()) {
                // let x = (*x.start())..=(*x.end());
                // let y = (*y.start())..=(*y.end());
                // let x = (x.start as f64)..=(x.end as f64);
                // let y = (y.start as f64)..=(y.end as f64);

                let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
                crate::regions::draw_rect_region(
                    &painter,
                    screen_size,
                    color,
                    view,
                    x.clone(),
                    y.clone(),
                );
            }
            // }

            if goto {
                let new_view =
                    view.fit_ranges_in_view_with_aspect_f64(aspect_ratio, x_range, y_range);
                *view = new_view;
            }

            ui.data_mut(|data| {
                data.insert_temp(target_id, target_buf);
                data.insert_temp(query_id, query_buf);
            });
        });
    // .
}

/*
pub fn goto_range_controls(
    alignment_grid: &crate::grid::AlignmentGrid,
    seq_names: &BiMap<String, SeqId>,
    input: &PafInput,
    view: &mut View,
    ui: &mut Ui,
) {

    let parse_range = |names: &BiMap<String, SeqId>,
                       axis: &crate::grid::GridAxis,
                       txt: &str|
     -> Option<std::ops::Range<u64>> {
        let mut split = txt.split(':');
        let name = split.next()?;
        let id = *names.get_by_left(name)?;

        let offset = axis.sequence_offset(id)?;
        // let offset = seqs[id].offset;

        let mut range = split
            .next()?
            .split('-')
            .filter_map(|s| s.parse::<u64>().ok());
        let start = range.next()? + offset;
        let end = range.next()? + offset;

        Some(start..end)
    };


}

*/
