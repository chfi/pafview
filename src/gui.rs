use bimap::BiMap;
use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{annotations::AnnotationStore, view::View, AlignedSeq, PafInput, PafViewerApp};

pub mod regions;

pub fn goto_range_controls(
    alignment_grid: &crate::grid::AlignmentGrid,
    seq_names: &BiMap<String, usize>,
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

    let parse_range = |names: &BiMap<String, usize>,
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

    let x_range = parse_range(&seq_names, &alignment_grid.x_axis, &target_buf);
    let y_range = parse_range(&seq_names, &alignment_grid.y_axis, &query_buf);

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
    ctx: &egui::Context,
    alignment_grid: &crate::grid::AlignmentGrid,
    seq_names: &BiMap<String, usize>,
    input: &PafInput,
    view: &mut View,
    window_states: &mut AppWindowStates,
) {
    egui::Window::new("View")
        .open(&mut window_states.goto_region_open)
        // .resizable(true)
        // .vscroll(true)
        // .default_open(false)
        .show(&ctx, |ui| {
            goto_range_controls(alignment_grid, seq_names, input, view, ui);

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
            egui::pos2(screen_x - 8.0 - galley.size().x, 32.0),
            galley,
            Color32::BLACK,
        );
    } else {
        painter.galley(egui::pos2(screen_x + 8.0, 32.0), galley, Color32::BLACK);
    }
}

pub fn draw_cursor_position_rulers(
    grid: &crate::grid::AlignmentGrid,
    seq_names: &bimap::BiMap<String, usize>,
    // input: &PafInput,
    ctx: &egui::Context,
    view: &crate::view::View,
) {
    let Some(cursor_pos) = ctx.input(|i| i.pointer.latest_pos()) else {
        return;
    };

    let layer = egui::LayerId::new(egui::Order::Background, egui::Id::new("ruler-painter"));
    let painter = ctx.layer_painter(layer);

    let screen_size = ctx.screen_rect().size();

    draw_cursor_position_rulers_impl(grid, seq_names, &painter, screen_size, view, cursor_pos)
}

pub fn draw_cursor_position_rulers_impl(
    grid: &crate::grid::AlignmentGrid,
    seq_names: &bimap::BiMap<String, usize>,
    // input: &PafInput,
    painter: &egui::Painter,
    window_dims: impl Into<[f32; 2]>,
    view: &crate::view::View,
    cursor_pos: impl Into<[f32; 2]>,
) {
    let window_dims = window_dims.into();
    let cursor_pos = cursor_pos.into();
    let pos = Vec2::from(cursor_pos);

    let world_pt = view.map_screen_to_world(window_dims, pos);

    if let Some((tgt_id, loc_offset)) = grid.x_axis.global_to_axis_exact(world_pt.x as u64) {
        let name = seq_names.get_by_right(&tgt_id).unwrap();
        let label = format!("{}:{loc_offset}", name);
        draw_ruler_v(painter, window_dims, pos.x, &label);
    }

    if let Some((qry_id, loc_offset)) = grid.y_axis.global_to_axis_exact(world_pt.y as u64) {
        let name = seq_names.get_by_right(&qry_id).unwrap();
        let label = format!("{}:{loc_offset}", name);
        draw_ruler_h(painter, window_dims, pos.y, &label);
    }
}

pub struct MenuBar;

impl MenuBar {
    pub fn show(
        // &self,
        ctx: &egui::Context,
        app: &PafViewerApp,
        window_states: &mut AppWindowStates,
    ) {
        egui::TopBottomPanel::top("menu_panel").show(ctx, |ui| {
            // show/hide goto range window
            ui.horizontal(|ui| {
                if ui.button("Go to range").clicked() {
                    window_states.goto_region_open = !window_states.goto_region_open;
                }

                // show/hide annotations list window
                if let Some(open) = window_states.annotation_list_open.as_mut() {
                    if ui.button("Annotations").clicked() {
                        *open = !*open;
                    }
                }
            })
        });
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AppWindowStates {
    pub annotation_list_open: Option<bool>,
    pub goto_region_open: bool,

    pub regions_of_interest_open: bool,
}

impl AppWindowStates {
    pub fn new(annotations: &AnnotationStore) -> Self {
        let annotation_list_open = (!annotations.is_empty()).then_some(false);

        AppWindowStates {
            annotation_list_open,
            goto_region_open: false,
            regions_of_interest_open: false,
        }
    }
}
