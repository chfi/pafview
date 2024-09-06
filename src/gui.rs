use egui::{Color32, FontId};
use ultraviolet::Vec2;

use crate::{annotations::AnnotationStore, sequences::SeqId};

pub mod config;
// pub mod debug;
pub mod goto;

/*
pub fn view_controls(
    ctx: &egui::Context,
    alignment_grid: &crate::grid::AlignmentGrid,
    seq_names: &BiMap<String, SeqId>,
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
*/

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
    seq_names: &bimap::BiMap<String, SeqId>,
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
    seq_names: &bimap::BiMap<String, SeqId>,
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
        let mut label = format!("{}:{loc_offset}", name);
        #[cfg(debug_assertions)]
        {
            label = format!("({}) {label}", tgt_id.0);
        }
        draw_ruler_v(painter, window_dims, pos.x, &label);
    }

    if let Some((qry_id, loc_offset)) = grid.y_axis.global_to_axis_exact(world_pt.y as u64) {
        let name = seq_names.get_by_right(&qry_id).unwrap();
        let mut label = format!("{}:{loc_offset}", name);
        #[cfg(debug_assertions)]
        {
            label = format!("({}) {label}", qry_id.0);
        }
        draw_ruler_h(painter, window_dims, pos.y, &label);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AppWindowStates {
    pub annotation_list_open: Option<bool>,
    pub goto_region_open: bool,

    pub regions_of_interest_open: bool,

    pub config_open: bool,

    #[cfg(debug_assertions)]
    pub label_physics_debug_open: bool,
}

impl std::default::Default for AppWindowStates {
    fn default() -> Self {
        Self {
            annotation_list_open: None,
            goto_region_open: false,
            regions_of_interest_open: false,
            config_open: false,

            #[cfg(debug_assertions)]
            label_physics_debug_open: false,
        }
    }
}

impl AppWindowStates {
    pub fn new(annotations: &AnnotationStore) -> Self {
        let annotation_list_open = (!annotations.is_empty()).then_some(false);

        AppWindowStates {
            annotation_list_open,
            goto_region_open: false,
            regions_of_interest_open: false,

            config_open: false,

            #[cfg(debug_assertions)]
            label_physics_debug_open: false,
            // label_physics_debug_open: true,
        }
    }
}
