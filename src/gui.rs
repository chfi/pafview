use egui::{util::IdTypeMap, DragValue, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{view::View, AlignedSeq, NameCache, PafInput};

pub(crate) fn goto_range_controls(
    name_cache: &NameCache,
    input: &PafInput,
    view: &mut View,
    ui: &mut Ui,
) {
    let target_id = ui.id().with("target-range");
    let query_id = ui.id().with("query-range");

    let (mut target_buf, mut query_buf) = ui
        .data(|data| {
            let t = data.get_temp::<String>(target_id)?;
            let q = data.get_temp::<String>(query_id)?;
            Some((t, q))
        })
        .unwrap_or_default();

    let parse_range = |names: &FxHashMap<String, usize>,
                       seqs: &[AlignedSeq],
                       txt: &str|
     -> Option<std::ops::Range<usize>> {
        let mut split = txt.split(':');
        let name = split.next()?;
        let id = *names.get(name)?;

        let offset = seqs[id].offset;

        let mut range = split
            .next()?
            .split('-')
            .filter_map(|s| s.parse::<usize>().ok());
        let start = range.next()? + offset;
        let end = range.next()? + offset;

        Some(start..end)
    };

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
        let target_text = ui.text_edit_singleline(&mut target_buf);
        // let go = ui.button("Go to target");

        goto |= target_text.lost_focus() && pressed_enter;

        // if go.clicked() || (target_text.lost_focus() && pressed_enter) {
        //     if let Some(range) = parse_range(&name_cache.target_names, &input.targets, &target_buf)
        //     {
        //         let new_view = view.fit_ranges_in_view(aspect_ratio, Some(range), None);
        //         *view = new_view;
        //     }
        // }
    });

    // Query/Y

    ui.horizontal(|ui| {
        let query_text = ui.text_edit_singleline(&mut query_buf);
        // let go = ui.button("Go to query");

        goto |= query_text.lost_focus() && pressed_enter;

        // if go.clicked() || (query_text.lost_focus() && pressed_enter) {
        //     if let Some(range) = parse_range(&name_cache.query_names, &input.queries, &query_buf) {
        //         let new_view = view.fit_ranges_in_view(aspect_ratio, None, Some(range));
        //         *view = new_view;
        //     }
        // }
    });

    let x_range = parse_range(&name_cache.target_names, &input.targets, &target_buf);
    let y_range = parse_range(&name_cache.query_names, &input.queries, &query_buf);

    let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("region-painter"));
    let painter = ui.ctx().layer_painter(layer);

    if goto_btn.hovered() {
        if let Some((x, y)) = x_range.as_ref().zip(y_range.as_ref()) {
            let x = (x.start as f64)..=(x.end as f64);
            let y = (y.start as f64)..=(y.end as f64);

            crate::regions::draw_rect_region(&painter, window_dims, view, x, y);
        }
    }

    if goto {
        let new_view = view.fit_ranges_in_view(aspect_ratio, x_range, y_range);
        *view = new_view;
    }

    ui.data_mut(|data| {
        data.insert_temp(target_id, target_buf);
        data.insert_temp(query_id, query_buf);
    });
}

pub(crate) fn view_controls(
    name_cache: &NameCache,
    input: &PafInput,
    view: &mut View,
    ctx: &egui::Context,
) {
    egui::Window::new("View")
        // .resizable(true)
        // .vscroll(true)
        // .default_open(false)
        .show(&ctx, |mut ui| {
            let x_limit = input.target_len as f64;
            let y_limit = input.query_len as f64;

            goto_range_controls(name_cache, input, view, ui);

            // let x_min =
            //     DragValue::new(&mut view.x_min).clamp_range(0f64..=(v.x_max - 1.0).max(0.0));
            // let x_max = DragValue::new(&mut view.x_max).clamp_range((v.x_min + 1.0)..=x_limit);
            let x_min = egui::Label::new(format!("{}", view.x_min));
            let x_max = egui::Label::new(format!("{}", view.x_max));
            let y_min = egui::Label::new(format!("{}", view.y_min));
            let y_max = egui::Label::new(format!("{}", view.y_max));

            ui.horizontal(|ui| {
                ui.add(x_min);
                ui.add(x_max);
            });

            // let y_min =
            //     DragValue::new(&mut view.y_min).clamp_range(0f64..=(v.y_max - 1.0).max(0.0));
            // let y_max = DragValue::new(&mut view.y_max).clamp_range((v.y_min + 1.0)..=y_limit);

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
