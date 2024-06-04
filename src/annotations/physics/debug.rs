use crate::math_conv::*;
use crate::view::Viewport;

use super::LabelPhysics;

#[derive(Clone)]
struct DebugWindowState {
    draw_anchor_links: bool,
    draw_projected_anchor_from_cursor: bool,

    debug_grid_tile_aabbs: bool,
    debug_grid_tile_at_point_cursor: bool,

    debug_optimal_tile_label_at_cursor: bool,
}

impl std::default::Default for DebugWindowState {
    fn default() -> Self {
        Self {
            draw_anchor_links: false,
            draw_projected_anchor_from_cursor: true,

            debug_grid_tile_aabbs: false,
            debug_grid_tile_at_point_cursor: false,
            debug_optimal_tile_label_at_cursor: true,
        }
    }
}

const BASE_ID: &'static str = "LabelPhysicsDebugWindow";

fn draw_projected_anchor(
    app: &crate::PafViewerApp,
    physics: &LabelPhysics,
    viewport: &Viewport,
    painter: &egui::Painter,
    screen_pos: egui::Pos2,
) {
    let screen_world = viewport.screen_world_mat3();
    let world_screen = viewport.world_screen_mat3();
    let world_pos = screen_world.transform_point2(screen_pos.as_uv());

    let stroke = (1.0, egui::Color32::RED);

    painter.circle_stroke(screen_pos, 3.0, stroke);

    if let Some(proj_y) =
        physics
            .heightfields
            .project_screen_from_top(&app.alignment_grid, viewport, screen_pos.x)
    {
        painter.line_segment(
            [screen_pos, [screen_pos.x, proj_y].as_epos2()],
            (2.0, stroke.1),
        );
    }

    if let Some((_, pos)) = app.alignment_grid.cast_ray(world_pos, [0.0, -1.0], true) {
        let pos = world_screen.transform_point2(pos.to_f32()).as_epos2();
        painter.line_segment([screen_pos, pos], (2.0, egui::Color32::LIGHT_BLUE));
    }
}

// fn draw_tile_aabbs

pub fn label_physics_debug_window(
    ctx: &egui::Context,
    open: &mut bool,
    app: &crate::PafViewerApp,
    physics: &LabelPhysics,
    viewport: &Viewport,
    // alignment_grid: &crate::grid::AlignmentGrid,
    // view: &mut View,
) {
    let eid = egui::Id::new(BASE_ID);
    //
    let mut dbg_state = ctx.data(|data| data.get_temp::<DebugWindowState>(eid).unwrap_or_default());

    let painter = ctx.layer_painter(egui::LayerId::debug());

    // let draw_x_at = |pos: egui::Pos2, size: f32| {
    // painter.line_
    // };

    if dbg_state.draw_projected_anchor_from_cursor {
        if let Some(pos) = ctx.pointer_latest_pos() {
            draw_projected_anchor(app, physics, viewport, &painter, pos);
        }
    }

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

            // ann_data.s
            let rb_rect = egui::Rect::from_center_size(rb_pos, ann_data.size.as_evec2());
            painter.rect_stroke(rb_rect, 0.0, anchor_stroke);
            // let rb_size = {
            // };
            // let rb_size =

            // let rb_rect =
            painter.line_segment([anchor_pos.as_epos2(), rb_pos], anchor_stroke);
            //
        }
    }

    if dbg_state.debug_grid_tile_aabbs {
        app.alignment_grid.draw_tile_aabbs(&painter, viewport);
    }

    if dbg_state.debug_optimal_tile_label_at_cursor {
        if let Some(pos) = ctx.pointer_latest_pos() {
            // draw_projected_anchor(app, physics, viewport, &painter, pos);

            let world_pos = viewport
                .screen_world_dmat3()
                .transform_point2(pos.as_uv().to_f64());

            let world_target = world_pos.x;
            let grid = &app.alignment_grid;

            // top_heightfield_in_visible_column start
            // let top = (viewport.view_center.y + viewport.view_size.y * 0.5) as f32;
            let top = (viewport.view_center.y + viewport.view_size.y * 0.45) as f32;
            // let top = grid.y_axis.total_len as f32;

            let origin = [world_target as f32, top];

            let origin_screen = viewport
                .world_screen_mat3()
                .transform_point2(origin.as_uv());

            // println!("casting ray from {origin:?}");
            if let Some((tile, hit_world)) = grid.cast_ray(origin, [0.0, -1.0], false) {
                let hit_screen = viewport
                    .world_screen_dmat3()
                    .transform_point2(hit_world)
                    .to_f32();
                //

                let hit = hit_screen.as_epos2();

                let stroke = (3.0, egui::Color32::GREEN);
                painter.line_segment([origin_screen.as_epos2(), hit], stroke);
                painter.circle_stroke(hit, 5.0, stroke);
            }

            if let Some((tile, hit_world)) = grid.cast_ray(origin, [0.0, -1.0], true) {
                let hit_screen = viewport
                    .world_screen_dmat3()
                    .transform_point2(hit_world)
                    .to_f32();
                //

                let hit = hit_screen.as_epos2();

                let stroke = (2.0, egui::Color32::RED);
                painter.line_segment([origin_screen.as_epos2(), hit], stroke);
                painter.circle_stroke(hit, 5.0, stroke);
            }

            // top_heightfield_in_visible_column end
            // if let Some((qry, hfield)) = physics.heightfields.top_heightfield_in_visible_column(
            //     &app.alignment_grid,
            //     viewport,
            //     world_pos.x,
            // ) {
            //     //
            // }
        }
    }

    egui::Window::new("Label Physics Debugger")
        .open(open)
        .show(ctx, |ui| {
            egui::Grid::new(ui.id().with("toggles")).show(ui, |ui| {
                ui.toggle_value(&mut dbg_state.draw_anchor_links, "Show label anchors");

                ui.toggle_value(
                    &mut dbg_state.draw_projected_anchor_from_cursor,
                    "Test anchor projection",
                );

                ui.end_row();

                ui.toggle_value(&mut dbg_state.debug_grid_tile_aabbs, "Show grid AABBs");
                ui.end_row();
            });

            ui.separator();

            if let Some(pos) = ctx.pointer_latest_pos() {
                let world_pos = viewport
                    .screen_world_dmat3()
                    .transform_point2(pos.as_uv().to_f64());

                if let Some(tgt) = app.alignment_grid.x_axis.global_to_axis_local(world_pos.x) {
                    ui.label(format!(
                        "target under cursor (world {}): {tgt:?}",
                        world_pos.x
                    ));
                } else {
                    ui.label(format!("no target under cursor (world {})", world_pos.x));
                }

                if dbg_state.debug_grid_tile_at_point_cursor {
                    if let Some(pair @ (tgt, qry)) =
                        app.alignment_grid.tile_at_world_point(world_pos)
                    {
                        ui.label("Hovering alignment {pair:?}");
                    }
                }
                //
            }
        });

    ctx.data_mut(|data| data.insert_temp(eid, dbg_state));
}

// pub fn draw_anchor_links
