use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{view::View, AlignedSeq, NameCache, PafInput};

pub fn goto_range_controls(name_cache: &NameCache, input: &PafInput, view: &mut View, ui: &mut Ui) {
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
        goto |= target_text.lost_focus() && pressed_enter;
    });

    // Query/Y

    ui.horizontal(|ui| {
        let query_text = ui.text_edit_singleline(&mut query_buf);
        goto |= query_text.lost_focus() && pressed_enter;
    });

    let x_range = parse_range(&name_cache.target_names, &input.targets, &target_buf);
    let y_range = parse_range(&name_cache.query_names, &input.queries, &query_buf);

    let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("region-painter"));
    let painter = ui.ctx().layer_painter(layer);

    let screen_size = ui.ctx().screen_rect().size();

    // if goto_btn.hovered() {
    if let Some((x, y)) = x_range.as_ref().zip(y_range.as_ref()) {
        let x = (x.start as f64)..=(x.end as f64);
        let y = (y.start as f64)..=(y.end as f64);

        let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
        crate::regions::draw_rect_region(&painter, screen_size, color, view, x, y);
    }
    // }

    if goto {
        let new_view = view.fit_ranges_in_view(aspect_ratio, x_range, y_range);
        *view = new_view;
    }

    ui.data_mut(|data| {
        data.insert_temp(target_id, target_buf);
        data.insert_temp(query_id, query_buf);
    });
}

pub fn view_controls(
    name_cache: &NameCache,
    input: &PafInput,
    view: &mut View,
    ctx: &egui::Context,
) {
    egui::Window::new("View")
        // .resizable(true)
        // .vscroll(true)
        // .default_open(false)
        .show(&ctx, |ui| {
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
                (pos, dims)
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

pub fn draw_ruler_h(
    painter: &egui::Painter,
    window_dims: [f32; 2],
    screen_y: f32,
    text: &str,
    //
) {
    let [w, _h] = window_dims;
    let p0 = egui::pos2(0.0, screen_y);
    let p1 = egui::pos2(w as f32, screen_y);
    painter.line_segment(
        [p0, p1],
        egui::Stroke {
            width: 1.0,
            color: egui::Rgba::from_gray(0.5).into(),
        },
    );

    let mut job = egui::text::LayoutJob::default();
    job.halign = egui::Align::LEFT;

    job.append(
        text,
        0.0,
        egui::text::TextFormat {
            font_id: FontId::monospace(12.0),
            color: Color32::PLACEHOLDER,
            valign: egui::Align::TOP,
            ..Default::default()
        },
    );

    let galley = painter.layout_job(job);

    painter.galley(egui::pos2(16.0, screen_y), galley, Color32::BLACK);
}

pub fn draw_ruler_v(painter: &egui::Painter, window_dims: [f32; 2], screen_x: f32, text: &str) {
    let [w, h] = window_dims;
    let p0 = egui::pos2(screen_x, 0.0);
    let p1 = egui::pos2(screen_x, h as f32);
    painter.line_segment(
        [p0, p1],
        egui::Stroke {
            width: 1.0,
            color: egui::Rgba::from_gray(0.5).into(),
        },
    );

    let mut job = egui::text::LayoutJob::default();
    job.halign = egui::Align::LEFT;

    job.append(
        text,
        0.0,
        egui::text::TextFormat {
            font_id: FontId::monospace(12.0),
            color: Color32::PLACEHOLDER,
            // valign: egui::Align::TOP,
            ..Default::default()
        },
    );

    let galley = painter.layout_job(job);

    if screen_x + galley.size().x > w as f32 {
        painter.galley(
            egui::pos2(screen_x - 8.0 - galley.size().x, 16.0),
            galley,
            Color32::BLACK,
        );
    } else {
        painter.galley(egui::pos2(screen_x + 8.0, 16.0), galley, Color32::BLACK);
    }
}

pub fn draw_cursor_position_rulers(
    input: &PafInput,
    ctx: &egui::Context,
    view: &crate::view::View,
) {
    let Some(cursor_pos) = ctx.input(|i| i.pointer.latest_pos()) else {
        return;
    };

    let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("ruler-painter"));
    let painter = ctx.layer_painter(layer);

    let screen_size = ctx.screen_rect().size();

    draw_cursor_position_rulers_impl(input, &painter, screen_size, view, cursor_pos)
}

pub fn draw_cursor_position_rulers_impl(
    input: &PafInput,
    painter: &egui::Painter,
    window_dims: impl Into<[f32; 2]>,
    view: &crate::view::View,
    cursor_pos: impl Into<[f32; 2]>,
) {
    let window_dims = window_dims.into();
    let cursor_pos = cursor_pos.into();
    let pos = Vec2::from(cursor_pos);

    let world_pt = view.map_screen_to_world(window_dims, pos);

    // get target sequence by doing a binary search on the targets' offsets
    if world_pt.x > 0.0 && world_pt.x < input.target_len as f64 {
        let tgt_offset = world_pt.x as usize;

        let tgt_ix = input.targets.partition_point(|seq| seq.offset < tgt_offset);
        if let Some(target) = input.targets.get(tgt_ix) {
            let offset_in_tgt = target.offset - tgt_offset;
            let label = format!("{}:{offset_in_tgt}", target.name);
            draw_ruler_v(painter, window_dims, pos.x, &label);
        }
    }

    // & the same for query and the Y coordinate
    if world_pt.y > 0.0 && world_pt.y < input.query_len as f64 {
        let qry_offset = world_pt.y as usize;

        let qry_ix = input.queries.partition_point(|seq| seq.offset < qry_offset);
        if let Some(query) = input.queries.get(qry_ix) {
            let offset_in_qry = query.offset - qry_offset;
            let label = format!("{}:{offset_in_qry}", query.name);
            draw_ruler_h(painter, window_dims, pos.y, &label);
        }
    }
}
