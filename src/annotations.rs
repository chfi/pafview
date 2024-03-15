use std::sync::Arc;

use ultraviolet::{DVec2, Vec2};

use crate::{NameCache, PafInput};

pub fn find_matches_for_target_range(
    names: &NameCache,
    input: &PafInput,
    target_name: &str,
    target_range: std::ops::Range<u64>,
) -> Vec<(usize, Vec<[DVec2; 2]>)> {
    let Some(&target_id) = names.target_names.get(target_name) else {
        return Vec::new();
    };
    let target = &input.targets[target_id];
    let global_tgt_offset = target.offset as u64 + target_range.start;

    let mut output = Vec::new();

    for (cid, cigar) in input.processed_lines.iter().enumerate() {
        if cigar.target_id != target_id {
            continue;
        }

        let start_ix = cigar
            .match_offsets
            .partition_point(|[tgt_ofs, _]| *tgt_ofs < target_range.start + global_tgt_offset);

        if start_ix == cigar.match_offsets.len() {
            continue;
        }

        let end_ix = cigar.match_offsets[start_ix..]
            .partition_point(|[tgt_ofs, _]| *tgt_ofs < target_range.end + global_tgt_offset);

        let match_range = start_ix..(start_ix + end_ix);

        let mut segments = Vec::new();

        for (match_i, &[start, end]) in cigar.match_edges[match_range].iter().enumerate() {
            // TODO only include the part of the line that's actually covered by the range?
            segments.push([start, end]);
        }

        if !segments.is_empty() {
            output.push((cid, segments));
        }
    }

    output
}

#[derive(Default, Clone)]
struct AnnotTestState {
    key: Option<(String, [u64; 2])>,
    match_output: Arc<egui::mutex::Mutex<Vec<(usize, Vec<[DVec2; 2]>)>>>,
}

pub fn draw_annotation_test_window(
    names: &NameCache,
    input: &PafInput,
    ctx: &egui::Context,
    view: &crate::view::View,
) {
    let id_str = "annotation-test-window";
    let id = egui::Id::new(id_str);
    let (mut range_text, mut label_text, mut annot_state) = ctx.data(|d| {
        let (range_text, label_text) = d.get_temp::<(String, String)>(id).unwrap_or_default();
        let state = d.get_temp::<AnnotTestState>(id).unwrap_or_default();
        (range_text, label_text, state)
    });

    egui::Window::new("Annotation Test").show(&ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Range");
            ui.text_edit_singleline(&mut range_text);
        });
        ui.horizontal(|ui| {
            ui.label("Label");
            ui.text_edit_singleline(&mut label_text);
        });
    });

    let parse_range = |text: &str| {
        let mut splits = text.split(':');
        let name = splits.next()?;
        let mut range_comp = splits.next()?.split('-');
        let start = range_comp.next().and_then(|s| s.parse::<u64>().ok())?;
        let end = range_comp.next().and_then(|s| s.parse::<u64>().ok())?;
        Some((name.to_string(), [start, end]))
    };

    if let key @ Some((name, [start, end])) = parse_range(&range_text).as_ref() {
        if key != annot_state.key.as_ref() {
            let new_outputs = find_matches_for_target_range(names, input, &name, *start..*end);
            annot_state.key = key.cloned();
            let mut outputs = annot_state.match_output.lock();
            *outputs = new_outputs;
        }
    } else {
        annot_state.key = None;
        let mut outputs = annot_state.match_output.lock();
        outputs.clear();
    }

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, id));

    let dims = ctx.screen_rect().size();

    let stroke = egui::Stroke {
        width: 4.0,
        color: egui::Color32::RED,
    };

    {
        let matches = annot_state.match_output.lock();
        let mut aabb_min = Vec2::broadcast(std::f32::MAX);
        let mut aabb_max = Vec2::broadcast(std::f32::MIN);
        for (cigar_i, matches) in matches.iter() {
            for (match_i, &[from, to]) in matches.iter().enumerate() {
                let s0 = view.map_world_to_screen(dims, from);
                let s1 = view.map_world_to_screen(dims, to);

                aabb_min = aabb_min.min_by_component(s0).min_by_component(s1);
                aabb_max = aabb_max.max_by_component(s0).max_by_component(s1);

                let p0: [f32; 2] = s0.into();
                let p1: [f32; 2] = s1.into();

                painter.line_segment([p0.into(), p1.into()], stroke);

                if match_i == matches.len() / 2 {
                    let d = s1 - s0;
                    let angle = d.y.atan2(d.x) as f32;

                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        &label_text,
                        0.0,
                        egui::text::TextFormat {
                            font_id: egui::FontId::monospace(12.0),
                            color: egui::Color32::PLACEHOLDER,
                            ..Default::default()
                        },
                    );

                    let galley = painter.layout_job(job);

                    let mut text_shape =
                        egui::epaint::TextShape::new(p0.into(), galley, egui::Color32::BLACK);
                    text_shape.angle = angle;

                    painter.add(text_shape);
                }
            }
        }

        // let rect = egui::Rect::from_x_y_ranges(aabb_min.x..=aabb_max.x, aabb_min.y..=aabb_max.y);
        // let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
        // painter.rect_filled(rect, 0.0, color);
    }

    ctx.data_mut(|d| {
        d.insert_temp(id, (range_text, label_text));
        d.insert_temp(id, annot_state);
    });
}
