use crate::math_conv::*;
use crate::view::Viewport;

use super::LabelPhysics;

#[derive(Clone)]
struct DebugWindowState {
    draw_anchor_links: bool,
}

impl std::default::Default for DebugWindowState {
    fn default() -> Self {
        Self {
            draw_anchor_links: true,
        }
    }
}

const BASE_ID: &'static str = "LabelPhysicsDebugWindow";

pub fn label_physics_debug_window(
    ctx: &egui::Context,
    open: &mut bool,
    physics: &LabelPhysics,
    viewport: &Viewport,
    // alignment_grid: &crate::grid::AlignmentGrid,
    // view: &mut View,
) {
    let eid = egui::Id::new(BASE_ID);
    //
    let mut dbg_state = ctx.data(|data| data.get_temp::<DebugWindowState>(eid).unwrap_or_default());

    let painter = ctx.layer_painter(egui::LayerId::debug());

    if dbg_state.draw_anchor_links {
        let anchor_stroke = (1.0, egui::Color32::BLUE);
        for (_ann_id, ann_data) in physics.annotations.iter() {
            let tgt_handle = ann_data.target_label_ix;

            let Some(anchor_pos) = physics.target_labels.anchor_screen_pos[tgt_handle] else {
                continue;
            };

            painter.circle_stroke(anchor_pos.as_epos2(), 5.0, anchor_stroke);

            let Some(rigid_body) = physics.target_labels.label_rigid_body[tgt_handle]
                .and_then(|handle| physics.rigid_body_set.get(handle))
            else {
                continue;
            };

            let rb_pos = rigid_body.position().translation.as_epos2();
            painter.line_segment([anchor_pos.as_epos2(), rb_pos], anchor_stroke);
            //
        }
    }

    egui::Window::new("Label Physics Debugger")
        .open(open)
        .show(ctx, |ui| {
            ui.toggle_value(&mut dbg_state.draw_anchor_links, "Show label anchors");
            //
        });

    ctx.data_mut(|data| data.insert_temp(eid, dbg_state));
}

// pub fn draw_anchor_links
