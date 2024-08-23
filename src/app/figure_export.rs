use std::sync::Arc;

use bevy::{math::DVec2, prelude::*, utils::HashMap};
use bevy_egui::{EguiContexts, EguiUserTextures};

use crate::{grid::AxisRange, paf::AlignmentIndex, sequences::SeqId};

use super::{
    render::{AlignmentDisplayBackImage, AlignmentRenderTarget, AlignmentViewer},
    selection::SelectionActionTrait,
    view::AlignmentViewport,
};

pub struct FigureExportPlugin;

impl Plugin for FigureExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ChangeExportTiles>()
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
            .add_systems(Startup, setup_figure_export_window)
            .add_systems(
                PreUpdate,
                (update_egui_textures, show_figure_export_window)
                    .chain()
                    .after(bevy_egui::EguiSet::BeginFrame),
            )
            .add_systems(
                PreUpdate,
                swap_egui_textures.before(super::render::swap_rendered_alignment_viewer_images),
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
fn setup_figure_export_window(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 1,
    };

    let mut image = Image {
        texture_descriptor: wgpu::TextureDescriptor {
            label: None,
            size,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    image.resize(size);

    let back_image = image.clone();

    let img_handle = images.add(image);
    let back_img_handle = images.add(back_image);

    let display_img = commands
        .spawn((
            FigureExportImage,
            AlignmentViewer::default().with_bg_color(Color::WHITE),
            img_handle,
            AlignmentDisplayBackImage {
                image: back_img_handle,
            },
        ))
        .id();

    // println!("figure export display sprite: {display_img:?}");

    commands.insert_resource(FigureExportWindow {
        display_img,
        egui_textures: None,
    });
}

#[derive(Component)]
struct FigureExportImage;

#[derive(Resource)]
struct FigureExportWindow {
    display_img: Entity,
    egui_textures: Option<[egui::TextureId; 2]>,
    // front_egui_img: Option<egui::TextureId>,
    // back_egui_img: Option<egui::TextureId>,
}

fn swap_egui_textures(
    mut fig_export: ResMut<FigureExportWindow>,
    // mut egui_textures: ResMut<EguiUserTextures>,
    sprites: Query<(
        // Entity,
        &Handle<Image>,
        &AlignmentDisplayBackImage,
        &AlignmentRenderTarget,
    )>,
) {
    let Ok((front, back, tgt)) = sprites.get(fig_export.display_img) else {
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
    img_query: Query<(&AlignmentViewer, &Handle<Image>, &AlignmentDisplayBackImage)>,
) {
    // the egui_textures field should be set to None when the images change (e.g. due to resize)
    // (and/or maybe this should be redesigned)
    if fig_export.egui_textures.is_some() {
        return;
    }

    let Ok((_disp_image, image, back_img)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let front = egui_textures.add_image(image.clone_weak());
    let back = egui_textures.add_image(back_img.image.clone_weak());

    fig_export.egui_textures = Some([front, back]);
}

fn show_figure_export_window(
    mut commands: Commands,
    mut contexts: EguiContexts,

    mut window_open: ResMut<FigureExportWindowOpen>,
    // mut is_rendering: Local<bool>,
    alignment_viewport: Res<AlignmentViewport>,
    alignment_grid: Res<crate::AlignmentGrid>,

    img_query: Query<(Entity, &AlignmentViewer, &AlignmentDisplayBackImage)>,
    mut fig_export: ResMut<FigureExportWindow>,
) {
    let Ok((_disp_ent, _disp_image, back_image)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let image = &back_image.image;

    // let is_rendering = rendering_query.get(disp_ent).is_ok();
    let is_rendering = false;

    let ctx = contexts.ctx_mut();

    egui::Window::new("Figure Exporter")
        .default_width(600.0)
        .resizable(true)
        .open(&mut window_open.is_open)
        .show(&ctx, |ui| {
            //

            if let Some([front, _]) = fig_export.egui_textures.as_ref() {
                ui.image((*front, egui::vec2(500.0, 500.0)));
            }

            let mut button = |text: &str| ui.add_enabled(!is_rendering, egui::Button::new(text));

            if button("render current view").clicked() {
                commands
                    .entity(fig_export.display_img)
                    .insert(AlignmentRenderTarget {
                        alignment_view: alignment_viewport.view.clone(),
                        canvas_size: [500, 500].into(),
                        image: image.clone(),
                        is_ready: Arc::new(false.into()),
                    });
            }

            if button("render initial view").clicked() {
                let view = &alignment_viewport.initial_view;

                let new_view = alignment_viewport
                    .view
                    .fit_ranges_in_view_f64(Some(view.x_range()), Some(view.y_range()));

                commands
                    .entity(fig_export.display_img)
                    .insert(AlignmentRenderTarget {
                        alignment_view: new_view,
                        canvas_size: [500, 500].into(),
                        image: image.clone(),
                        is_ready: Arc::new(false.into()),
                    });
            }

            ui.vertical(|ui| {
                let target_id = ui.id().with("target-range");
                let query_id = ui.id().with("query-range");

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

                // if ui.button("")
                if let Some((x_range, y_range)) = x_range.zip(y_range) {
                    // TODO
                }

                ui.data_mut(|data| {
                    data.insert_temp(target_id, target_buf);
                    data.insert_temp(query_id, query_buf);
                });
            });
        });
}

#[derive(Event, Debug, Clone, Copy, Reflect)]
struct ChangeExportTiles {
    world_region: [DVec2; 2],
}

fn update_exported_tiles(
    mut commands: Commands,
    mut last_tile_region: Local<Option<[std::ops::RangeInclusive<usize>; 2]>>,
    mut export_region_events: EventReader<ChangeExportTiles>,

    alignment_grid: Res<crate::AlignmentGrid>,
) {
    let Some(new_region) = export_region_events.read().last() else {
        return;
    };

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
        // create layout material & update viewer entity
    }

    todo!();
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
    mut region_events: EventWriter<ChangeExportTiles>,
    selections: Query<
        &super::selection::Selection,
        (
            With<FigureRegionSelection>,
            Added<super::selection::SelectionComplete>,
        ),
    >,
) {
    for selection in selections.iter() {
        let ev = ChangeExportTiles {
            world_region: [selection.start_world, selection.end_world],
        };
        region_events.send(ev);
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
    if !selection_mode.user_is_selecting {
        selection_actions.consume(&super::selection::SelectionAction::RegionSelection);
        return;
    }

    //
}
