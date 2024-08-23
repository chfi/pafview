use std::sync::Arc;

use bevy::{
    math::{vec2, DVec2},
    prelude::*,
    utils::{HashMap, HashSet},
};
use bevy_egui::{EguiContexts, EguiUserTextures};

use crate::{grid::AxisRange, sequences::SeqId};

use super::{
    render::{
        AlignmentLayoutMaterials, AlignmentPolylineMaterial, AlignmentRenderTarget,
        AlignmentViewer, AlignmentViewerImages,
    },
    selection::SelectionActionTrait,
    view::AlignmentViewport,
};

pub struct FigureExportPlugin;

impl Plugin for FigureExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ChangeExportTileRegion>()
            .add_event::<UpdateExportAlignmentLayout>()
            .init_resource::<FigureExportWindowOpen>()
            .init_resource::<FigureRegionSelectionMode>()
            .add_systems(PreUpdate, handle_selection_user_input)
            .add_systems(
                Update,
                (
                    super::selection::selection_action_input_system::<FigureRegionSelection>,
                    send_region_change_event,
                )
                    .chain(),
            )
            .add_systems(
                Startup,
                setup_figure_export_window
                    .after(super::render::prepare_alignment_grid_layout_materials),
            )
            .add_systems(
                PreUpdate,
                (update_egui_textures, show_figure_export_window)
                    .chain()
                    .after(bevy_egui::EguiSet::BeginFrame),
            )
            .add_systems(
                PreUpdate,
                swap_egui_textures.before(super::render::swap_rendered_alignment_viewer_images),
            )
            .add_systems(
                Update,
                (
                    update_exported_tiles.after(send_region_change_event),
                    update_figure_export_alignment_layout,
                    update_figure_export_layout_children,
                )
                    .chain(),
            );

        // app.add_systems(
        //     Update,
        //     |query: Query<
        //         (Entity, &AlignmentRenderTarget),
        //         (With<FigureExportImage>, Added<AlignmentRenderTarget>),
        //     >| {
        //         for (ent, tgt) in query.iter() {
        //             println!("render target added for {ent:?}:\t{tgt:?}");
        //         }
        //     },
        // );
    }
}

#[derive(Debug, Default, Resource, Reflect)]
pub struct FigureExportWindowOpen {
    pub is_open: bool,
}

// initializes the figure export window object and relevant assets (image for rendering etc.)
fn setup_figure_export_window(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    grid_layout: Res<super::render::AlignmentGridLayout>,
) {
    let mut viewer =
        super::render::spawn_alignment_viewer_grid_layout(&mut commands, &mut images, &grid_layout);
    let viewer = viewer
        .insert((
            FigureExportImage,
            AlignmentViewer::default().with_bg_color(Color::WHITE),
        ))
        .id();

    commands.insert_resource(FigureExportWindow {
        display_img: viewer,
        egui_textures: None,
        export_layout_size: None,
        export_layouts: None,
    });
}

#[derive(Component)]
struct FigureExportImage;

#[derive(Resource)]
struct FigureExportWindow {
    display_img: Entity,
    egui_textures: Option<[egui::TextureId; 2]>,

    export_layout_size: Option<DVec2>,
    export_layouts: Option<FigureExportLayouts>,
}

fn swap_egui_textures(
    mut fig_export: ResMut<FigureExportWindow>,
    sprites: Query<&AlignmentRenderTarget, With<FigureExportImage>>,
) {
    let Ok(tgt) = sprites.get(fig_export.display_img) else {
        return;
    };

    if tgt.is_ready.load() {
        if let Some(txs) = fig_export.egui_textures.as_mut() {
            txs.swap(0, 1);
        }
    }
}

fn update_egui_textures(
    mut egui_textures: ResMut<EguiUserTextures>,

    mut fig_export: ResMut<FigureExportWindow>,
    img_query: Query<(&AlignmentViewer, &AlignmentViewerImages)>,
) {
    // the egui_textures field should be set to None when the images change (e.g. due to resize)
    // (and/or maybe this should be redesigned)
    if fig_export.egui_textures.is_some() {
        return;
    }

    let Ok((_disp_image, images)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let front = egui_textures.add_image(images.front.clone_weak());
    let back = egui_textures.add_image(images.back.clone_weak());

    fig_export.egui_textures = Some([front, back]);
}

fn show_figure_export_window(
    mut commands: Commands,
    mut contexts: EguiContexts,

    mut window_open: ResMut<FigureExportWindowOpen>,
    // mut is_rendering: Local<bool>,
    alignment_viewport: Res<AlignmentViewport>,
    alignment_grid: Res<crate::AlignmentGrid>,

    active_renders: Query<&AlignmentRenderTarget>,
    mut img_query: Query<
        (Entity, &mut AlignmentViewer, &AlignmentViewerImages),
        With<FigureExportImage>,
    >,
    mut fig_export: ResMut<FigureExportWindow>,
    mut region_select_mode: ResMut<FigureRegionSelectionMode>,
    keys: Res<ButtonInput<KeyCode>>,

    mut region_events: EventWriter<ChangeExportTileRegion>,
    // mut
) {
    // println!(
    //     "user is selecting??: {}",
    //     region_select_mode.user_is_selecting
    // );

    let Ok((viewer_ent, mut viewer, viewer_images)) = img_query.get_mut(fig_export.display_img)
    else {
        return;
    };

    let is_rendering = active_renders.contains(viewer_ent);

    if region_select_mode.user_is_selecting {
        if keys.just_pressed(KeyCode::Escape) {
            region_select_mode.user_is_selecting = false;
        } else {
            return;
        }
    }

    let ctx = contexts.ctx_mut();

    egui::Window::new("Figure Exporter")
        .default_width(600.0)
        .resizable(true)
        .open(&mut window_open.is_open)
        .show(&ctx, |ui| {
            //

            if ui.button("Select tiles").clicked() {
                region_select_mode.user_is_selecting = true;
            }

            if let Some([front, _]) = fig_export.egui_textures.as_ref() {
                ui.image((*front, egui::vec2(500.0, 500.0)));
            }

            let mut button = |text: &str| ui.add_enabled(!is_rendering, egui::Button::new(text));

            if button("render current view").clicked() {
                viewer.next_view = Some(alignment_viewport.view.clone());
            }

            if button("render initial view").clicked() {
                let view = &alignment_viewport.initial_view;

                let new_view = alignment_viewport
                    .view
                    .fit_ranges_in_view_f64(Some(view.x_range()), Some(view.y_range()));

                viewer.next_view = Some(new_view);
            }

            let target_id = ui.id().with("target-range");
            let query_id = ui.id().with("query-range");

            ui.vertical(|ui| {
                let (mut target_buf, mut query_buf) = ui
                    .data(|data| {
                        let t = data.get_temp::<String>(target_id)?;
                        let q = data.get_temp::<String>(query_id)?;
                        Some((t, q))
                    })
                    .unwrap_or_default();

                ui.horizontal(|ui| {
                    ui.label("Target");
                    ui.text_edit_singleline(&mut target_buf);
                });

                ui.horizontal(|ui| {
                    ui.label("Query");
                    ui.text_edit_singleline(&mut query_buf);
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

                let ranges = x_range.zip(y_range);
                if ui
                    .add_enabled(ranges.is_some(), egui::Button::new("Export tile subset"))
                    .clicked()
                {
                    if let Some((x_range, y_range)) = ranges {
                        let p0 = DVec2::new(*x_range.start(), *y_range.start());
                        let p1 = DVec2::new(*x_range.end(), *y_range.end());
                        region_events.send(ChangeExportTileRegion {
                            world_region: [p0, p1],
                        });
                    }
                }

                ui.data_mut(|data| {
                    data.insert_temp(target_id, target_buf);
                    data.insert_temp(query_id, query_buf);
                });
            });

            ui.vertical(|ui| {
                let (mut tgt_len, mut query_len) = ui.data(|data| {
                    let t = data
                        .get_temp::<f64>(target_id)
                        .unwrap_or(alignment_grid.x_axis.total_len as f64);
                    let q = data
                        .get_temp::<f64>(query_id)
                        .unwrap_or(alignment_grid.y_axis.total_len as f64);
                    (t, q)
                });

                ui.label("Custom layout");

                ui.horizontal(|ui| {
                    ui.label("X Size");
                    ui.add(egui::DragValue::new(&mut tgt_len));
                });
                ui.horizontal(|ui| {
                    ui.label("Y Size");
                    ui.add(egui::DragValue::new(&mut query_len));
                });

                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        fig_export.export_layout_size = Some(DVec2::new(tgt_len, query_len));
                    }
                    if ui.button("Clear").clicked() {
                        fig_export.export_layout_size = None;
                    }
                });

                ui.data_mut(|data| {
                    data.insert_temp(target_id, tgt_len);
                    data.insert_temp(query_id, query_len);
                });
            })
        });
}

#[derive(Event, Debug, Clone, Copy, Reflect)]
struct ChangeExportTileRegion {
    world_region: [DVec2; 2],
}

fn update_exported_tiles(
    mut commands: Commands,
    mut last_tile_region: Local<Option<[std::ops::RangeInclusive<usize>; 2]>>,

    mut export_region_events: EventReader<ChangeExportTileRegion>,
    mut layout_events: EventWriter<UpdateExportAlignmentLayout>,

    figure_export_window: Res<FigureExportWindow>,

    alignment_grid: Res<crate::AlignmentGrid>,
    alignments: Res<crate::Alignments>,
) {
    let Some(new_region) = export_region_events.read().last() else {
        return;
    };
    println!("updating region to export: {new_region:?}");

    let [p0, p1] = new_region.world_region;
    let min = p0.min(p1);
    let max = p0.max(p1);

    let x_axis = &alignment_grid.x_axis;
    let y_axis = &alignment_grid.y_axis;

    let x_tiles = x_axis.tiles_covered_by_range(min.x..=max.x);
    let y_tiles = y_axis.tiles_covered_by_range(min.y..=max.y);

    let Some((x_tiles, y_tiles)) = x_tiles.zip(y_tiles) else {
        return;
    };
    let x_tiles = x_tiles.collect::<Vec<_>>();
    let y_tiles = y_tiles.collect::<Vec<_>>();

    let map_ix = |s: Option<&SeqId>, a: &crate::grid::GridAxis| {
        s.and_then(|s| a.seq_index_map.get(s).copied())
    };

    let x_ixs = map_ix(x_tiles.first(), x_axis).zip(map_ix(x_tiles.last(), x_axis));
    let y_ixs = map_ix(y_tiles.first(), y_axis).zip(map_ix(y_tiles.last(), y_axis));
    // let p1_ixs = map_ix(x_tiles.first(), x_axis).zip(map_ix(x_tiles.last(), x_axis));
    // let p1_ixs = map_ix(x_tiles.first(), &x_axis).zip(x_tiles.last(), &x_axis);
    // let x0_ix = x_tiles
    //     .first()
    //     .and_then(|s| x_axis.seq_index_map.get(s))
    //     .unwrap();

    let Some(((x0, x1), (y0, y1))) = x_ixs.zip(y_ixs) else {
        return;
    };
    let x_range = x0..=x1;
    let y_range = y0..=y1;

    let region_changed = if let Some(prev_ranges) = last_tile_region.as_ref() {
        let [prev_xr, prev_yr] = prev_ranges;
        prev_xr != &x_range || prev_yr != &y_range
    } else {
        false
    };

    if region_changed {
        *last_tile_region = Some([x_range.clone(), y_range.clone()]);
        // create layout material & update viewer entity
        let mut alignment_set = HashSet::default();
        let layout_size = figure_export_window.export_layout_size;

        for &query in &x_tiles {
            for &target in &y_tiles {
                let key = (target, query);

                if let Some(als) = alignments.pairs.get(&key) {
                    for (ix, _al) in als.iter().enumerate() {
                        alignment_set.insert(super::alignments::AlignmentIndex {
                            query,
                            target,
                            pair_index: ix,
                        });
                    }
                }
            }
        }
        layout_events.send(UpdateExportAlignmentLayout {
            alignment_set,
            layout_size,
        });
    }
}

#[derive(Event, Debug, Reflect)]
struct UpdateExportAlignmentLayout {
    alignment_set: HashSet<super::alignments::AlignmentIndex>,
    layout_size: Option<DVec2>,
}

fn update_figure_export_alignment_layout(
    mut update_events: EventReader<UpdateExportAlignmentLayout>,

    mut alignment_mats: ResMut<Assets<AlignmentPolylineMaterial>>,
    mut layout_mats: ResMut<Assets<AlignmentLayoutMaterials>>,

    mut export_window: ResMut<FigureExportWindow>,
    //
    alignment_store: Res<crate::Alignments>,
    alignment_grid: Res<crate::AlignmentGrid>,
    color_schemes: Res<super::AlignmentColorSchemes>,
    vertex_buffer_index: Res<super::render::AlignmentVerticesIndex>,
) {
    let Some(UpdateExportAlignmentLayout {
        alignment_set,
        layout_size,
    }) = update_events.read().last()
    else {
        return;
    };

    if alignment_set.is_empty() {
        export_window.export_layouts = None;
        return;
    }

    let mut seq_tiles_to_place = HashSet::default();

    // step through the entire `alignment_set`
    for alignment in alignment_set {
        let Some(vx) = vertex_buffer_index.vertices.get(alignment) else {
            continue;
        };
        let color = color_schemes.get(alignment);

        seq_tiles_to_place.insert((alignment.target, alignment.query));
    }

    let mut x_min = std::u64::MAX;
    let mut x_max = std::u64::MIN;
    let mut y_min = std::u64::MAX;
    let mut y_max = std::u64::MIN;

    let mut seq_tile_positions = HashMap::default();

    for pair @ &(target, query) in seq_tiles_to_place.iter() {
        //
        let Some([tgt_range, qry_range]) = alignment_grid.seq_pair_ranges(target, query) else {
            continue;
        };

        x_min = tgt_range.start.min(x_min);
        x_max = tgt_range.end.min(x_max);
        y_min = qry_range.start.min(y_min);
        y_max = qry_range.end.min(y_max);

        seq_tile_positions.insert(*pair, [tgt_range.start as f64, qry_range.start as f64]);
    }

    let width = (x_max - x_min) as f64;
    let height = (y_max - y_min) as f64;

    let layout_size = layout_size.unwrap_or(DVec2::new(width, height));

    let x_scale = width / layout_size.x;
    let y_scale = height / layout_size.y;

    for (pair, offsets) in seq_tile_positions.iter_mut() {
        offsets[0] -= x_min as f64;
        offsets[1] -= y_min as f64;

        offsets[0] *= x_scale;
        offsets[1] *= y_scale;
    }

    let mut line_only_pos = Vec::new();
    let mut with_base_level_pos = Vec::new();

    for alignment in alignment_set {
        let Some(&[x, y]) = seq_tile_positions.get(&(alignment.target, alignment.query)) else {
            continue;
        };
        let pos = vec2(x as f32, y as f32);

        let Some(al_data) = alignment_store.get(*alignment) else {
            continue;
        };

        if al_data.cigar.is_empty() {
            line_only_pos.push((*alignment, pos));
        } else {
            with_base_level_pos.push((*alignment, pos));
        }
    }

    let line_only = layout_mats.add(AlignmentLayoutMaterials::from_positions_iter(
        &mut alignment_mats,
        &vertex_buffer_index,
        &color_schemes,
        line_only_pos,
    ));

    let with_base_level = layout_mats.add(AlignmentLayoutMaterials::from_positions_iter(
        &mut alignment_mats,
        &vertex_buffer_index,
        &color_schemes,
        with_base_level_pos,
    ));

    println!("Setting figure exporter custom layout");
    export_window.export_layouts = Some(FigureExportLayouts {
        line_only,
        with_base_level,
    });
}

fn update_figure_export_layout_children(
    mut commands: Commands,

    mut update_events: EventReader<UpdateExportAlignmentLayout>,

    fig_export: Res<FigureExportWindow>,
    grid_layout: Res<super::render::AlignmentGridLayout>,
) {
    if update_events.is_empty() {
        return;
    }
    update_events.clear();
    println!("updating figure export layout children");

    let mut viewer = commands.entity(fig_export.display_img);
    viewer.despawn_descendants();

    viewer.with_children(|parent| {
        if let Some(layouts) = fig_export.export_layouts.as_ref() {
            println!("using custom layout");
            parent.spawn(layouts.with_base_level.clone());
            parent.spawn((layouts.line_only.clone(), super::render::LineOnlyAlignment));
        } else {
            println!("using default layout");
            parent.spawn(grid_layout.with_base_level.clone());
            parent.spawn((
                grid_layout.line_only.clone(),
                super::render::LineOnlyAlignment,
            ));
        }
    });
}

#[derive(Debug, Reflect, Clone)]
struct FigureExportLayouts {
    line_only: Handle<AlignmentLayoutMaterials>,
    with_base_level: Handle<AlignmentLayoutMaterials>,
}

#[derive(Component, Default, Clone, Copy, Reflect)]
struct FigureRegionSelection;

impl SelectionActionTrait for FigureRegionSelection {
    fn action() -> super::selection::SelectionAction {
        super::selection::SelectionAction::RegionSelection
    }
}

// must run after selection_action_input_system::<FigureRegionSelection>
fn send_region_change_event(
    mut region_events: EventWriter<ChangeExportTileRegion>,
    mut select_mode: ResMut<FigureRegionSelectionMode>,
    selections: Query<
        &super::selection::Selection,
        (
            With<FigureRegionSelection>,
            Added<super::selection::SelectionComplete>,
        ),
    >,
) {
    for selection in selections.iter() {
        let ev = ChangeExportTileRegion {
            world_region: [selection.start_world, selection.end_world],
        };
        region_events.send(ev);
        select_mode.user_is_selecting = false;
    }
}

#[derive(Resource, Default, Clone, Copy, Reflect)]
pub struct FigureRegionSelectionMode {
    pub user_is_selecting: bool,
}

use leafwing_input_manager::prelude::*;

// NB: must run in PreUpdate (selection_action_input_systems should run in Update)
fn handle_selection_user_input(
    mut selection_actions: ResMut<ActionState<super::selection::SelectionAction>>,

    mut selection_mode: ResMut<FigureRegionSelectionMode>,
) {
    let select = super::selection::SelectionAction::RegionSelection;
    let release = super::selection::SelectionAction::SelectionRelease;

    if !selection_mode.user_is_selecting {
        selection_actions.consume(&select);
        return;
    }

    let Some(data) = selection_actions.action_data(&select) else {
        return;
    };
    let data = data.clone();

    if let Some(rel_data) = selection_actions.action_data_mut(&release) {
        *rel_data = data;
    }
}
