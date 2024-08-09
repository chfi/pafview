use bevy::{prelude::*, render::view::RenderLayers, sprite::MaterialMesh2dBundle, utils::HashMap};

use crate::{
    annotations::{AnnotationId, RecordEntryId, RecordListId},
    grid::AxisRange,
};

use super::view::AlignmentViewport;

pub(super) struct AnnotationsPlugin;

pub mod gui;
mod material;

/*

*/

impl Plugin for AnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabelPhysics>()
            .init_resource::<AnnotationPainter>()
            .init_resource::<Annotations>()
            .init_resource::<AnnotationEntityMap>()
            .add_event::<LoadAnnotationFile>()
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, load_annotation_file.pipe(prepare_annotations))
            .add_systems(
                Update,
                (update_annotation_regions, update_annotation_labels),
            );
        // .add_systems(
        //     Update,
        //     (update_annotation_labels, draw_annotations)
        //         .chain()
        //         .after(super::gui::menubar_system),
        // );
    }
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct Annotations(pub crate::annotations::AnnotationStore);

#[derive(Debug, Clone, Copy, Component)]
pub struct Annotation {
    pub record_list: RecordListId,
    pub list_index: RecordEntryId,
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct AnnotationEntityMap(HashMap<AnnotationId, Entity>);

#[derive(Component)]
pub struct DisplayEntities {
    query_region: Entity,
    query_label: Entity,

    target_region: Entity,
    target_label: Entity,
    // query_region: Option<Entity>,
    // query_label: Option<Entity>,

    // target_region: Option<Entity>,
    // target_label: Option<Entity>,
}

#[derive(Event)]
pub struct LoadAnnotationFile {
    pub path: std::path::PathBuf,
}

#[derive(Event)]
pub enum AnnotationEvent {
    ChangeVisibility { annot_id: Annotation, visible: bool },
}

#[derive(Resource, Default)]
struct LabelPhysics(crate::annotations::physics::LabelPhysics);

// NB: probably want to replace the egui painter-based annotation drawing
// with something cleaner & more integrated into bevy
#[derive(Resource, Default)]
pub struct AnnotationPainter(pub crate::annotations::draw::AnnotationPainter);

#[derive(Resource)]
struct DisplayHandles {
    mesh: bevy::sprite::Mesh2dHandle,
    // material: Handle<ColorMaterial>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    // mut materials: ResMut<Assets<ColorMaterial>>,
    alignments: Res<crate::Alignments>,
    mut load_events: EventWriter<LoadAnnotationFile>,
    mut label_physics: ResMut<LabelPhysics>,
) {
    let mesh = meshes.add(Mesh::from(Rectangle::default()));
    commands.insert_resource(DisplayHandles { mesh: mesh.into() });

    label_physics.0.heightfields =
        crate::annotations::physics::AlignmentHeightFields::from_alignments(&alignments);

    use clap::Parser;
    let args = crate::cli::Cli::parse();

    if let Some(path) = args.bed {
        load_events.send(LoadAnnotationFile { path });
    }
}

fn load_annotation_file(
    frame_count: Res<bevy::core::FrameCount>,
    sequences: Res<crate::Sequences>,
    mut annotations: ResMut<Annotations>,
    // mut annotation_painter: ResMut<AnnotationPainter>,
    // mut viewer: ResMut<super::PafViewer>,
    mut load_events: EventReader<LoadAnnotationFile>,
) -> Vec<crate::annotations::AnnotationId> {
    let mut labels_to_prepare = Vec::new();
    if frame_count.0 == 0 {
        return labels_to_prepare;
    }

    for LoadAnnotationFile { path } in load_events.read() {
        match annotations.load_bed_file(&sequences.sequence_names, &path) {
            Ok(list_id) => {
                let annot_ids = annotations
                    .list_by_id(list_id)
                    .into_iter()
                    .flat_map(|list| {
                        list.records
                            .iter()
                            .enumerate()
                            .map(|(record_id, _)| (list_id, record_id))
                    });

                labels_to_prepare.extend(annot_ids);

                log::info!("Loaded BED file `{path:?}`");
            }
            Err(err) => {
                log::error!("Error loading BED file at path `{path:?}`: {err:?}")
            }
        }
    }

    labels_to_prepare
}

fn prepare_annotations(
    In(labels_to_prepare): In<Vec<crate::annotations::AnnotationId>>,
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,

    annotations: Res<Annotations>,
    mut annot_entity_map: ResMut<AnnotationEntityMap>,

    display_handles: Res<DisplayHandles>,
) {
    for annot_id @ (list_id, entry_id) in labels_to_prepare {
        // TODO color from annotation/name

        let record = &annotations.list_by_id(list_id).unwrap().records[entry_id];

        let color = record.color;
        let annot_color =
            Color::srgba_u8(color.r(), color.g(), color.b(), color.a()).with_alpha(0.4);
        let color_mat = materials.add(ColorMaterial::from_color(annot_color));
        // let color_mat = materials.add(ColorMaterial::from_color(Color::srgb(0.8, 0.0, 0.0)));

        let query_region = commands
            .spawn((
                RenderLayers::layer(1),
                MaterialMesh2dBundle {
                    mesh: display_handles.mesh.clone(),
                    material: color_mat.clone(),
                    ..default()
                },
            ))
            // .insert(SpatialBundle::HIDDEN_IDENTITY)
            .insert(SpatialBundle::INHERITED_IDENTITY)
            .id();
        let target_region = commands
            .spawn((
                RenderLayers::layer(1),
                MaterialMesh2dBundle {
                    mesh: display_handles.mesh.clone(),
                    material: color_mat.clone(),
                    ..default()
                },
            ))
            // .insert(SpatialBundle::HIDDEN_IDENTITY)
            .insert(SpatialBundle::INHERITED_IDENTITY)
            .id();

        // TODO labels
        let query_label = commands.spawn(()).id();
        let target_label = commands.spawn(()).id();

        let mut annot_ent = commands.spawn(Annotation {
            record_list: list_id,
            list_index: entry_id,
        });

        annot_ent.insert(DisplayEntities {
            query_region,
            query_label,
            target_region,
            target_label,
        });

        let annot_ent = annot_ent.id();

        annot_entity_map.insert(annot_id, annot_ent);
    }
    //
}

fn update_annotation_regions(
    alignment_grid: Res<crate::AlignmentGrid>,
    annotations: Res<Annotations>,

    alignment_view: Res<AlignmentViewport>,

    windows: Query<&Window>,

    display_ents: Query<(&Annotation, &DisplayEntities)>,
    mut transforms: Query<&mut Transform>,
) {
    let x_axis = &alignment_grid.x_axis;
    let y_axis = &alignment_grid.y_axis;

    let screen_dims = windows.single().size();

    for (annot_id, entities) in display_ents.iter() {
        let list = annotations.list_by_id(annot_id.record_list).unwrap();
        let record = &list.records[annot_id.list_index];

        let axis_range = AxisRange::Seq {
            seq_id: record.seq_id,
            range: record.seq_range.clone(),
        };
        let world_x_range = x_axis.axis_range_into_global(&axis_range);
        let world_y_range = y_axis.axis_range_into_global(&axis_range);

        let Some((world_x_range, world_y_range)) = world_x_range.zip(world_y_range) else {
            continue;
        };

        // update region transforms (screenspace) based on current view

        let s0 = alignment_view.view.map_world_to_screen(
            screen_dims,
            [*world_x_range.start(), *world_y_range.start()],
        );
        let s1 = alignment_view
            .view
            .map_world_to_screen(screen_dims, [*world_x_range.end(), *world_y_range.end()]);

        let s0 = Vec2::new(
            s0.x - screen_dims.x * 0.5,
            screen_dims.y - s0.y - screen_dims.y * 0.5,
        );
        let s1 = Vec2::new(
            s1.x - screen_dims.x * 0.5,
            screen_dims.y - s1.y - screen_dims.y * 0.5,
        );

        let mid = (s0 + s1) * 0.5;

        // hacky fix to avoid z-fighting
        let z = -1.0 - (annot_id.list_index as f32) / 1_000_000.0;

        if let Ok(mut transform) = transforms.get_mut(entities.query_region) {
            transform.translation = Vec3::new(0.0, mid.y, z);

            let width = (s0.y - s1.y).abs().max(0.5);
            transform.scale = Vec3::new(screen_dims.x, width, 1.0);
        }

        if let Ok(mut transform) = transforms.get_mut(entities.target_region) {
            transform.translation = Vec3::new(mid.x, 0.0, z - 1.0);

            let width = (s0.x - s1.x).abs().max(0.5);
            transform.scale = Vec3::new(width, screen_dims.y, 1.0);
        }
    }
}

fn update_annotation_labels(
    //
    time: Res<Time>,
    // viewer: Res<super::PafViewer>,
    app_view: Res<AlignmentViewport>,
    windows: Query<&Window>,

    mut label_physics: ResMut<LabelPhysics>,
    // mut annotation_painter: ResMut<AnnotationPainter>,
) {

    //
}

/*
fn prepare_annotations_old(
    // while we're using egui's fonts, which won't be available the first frame
    In(labels_to_prepare): In<Vec<crate::annotations::AnnotationId>>,

    frame_count: Res<bevy::core::FrameCount>,
    viewer: Res<super::PafViewer>,

    mut contexts: bevy_egui::EguiContexts,
    mut annotation_painter: ResMut<AnnotationPainter>,
    mut label_physics: ResMut<LabelPhysics>,
) {
    if labels_to_prepare.is_empty() || frame_count.0 == 0 {
        return;
    }

    let ctx = contexts.ctx_mut();

    let app = &viewer.app;

    ctx.fonts(|fonts| {
        label_physics.0.prepare_annotations(
            &app.alignment_grid,
            &app.annotations,
            labels_to_prepare.into_iter(),
            fonts,
            &mut annotation_painter.0,
        );
    });
}

fn update_annotation_labels_old(
    //
    time: Res<Time>,
    viewer: Res<super::PafViewer>,
    app_view: Res<AlignmentViewport>,
    windows: Query<&Window>,

    mut label_physics: ResMut<LabelPhysics>,
    mut annotation_painter: ResMut<AnnotationPainter>,

    // just for debug painter (temporary)
    mut contexts: bevy_egui::EguiContexts,
) {
    let window = windows.single();
    let res = &window.resolution;

    let view = &app_view.view;
    let viewport = crate::view::Viewport {
        view_center: view.center(),
        view_size: view.size(),
        canvas_offset: [0.0, 0.0].into(),
        canvas_size: [res.width(), res.height()].into(),
    };

    let grid = &viewer.app.alignment_grid;

    let debug_painter = contexts.ctx_mut().debug_painter();

    label_physics
        .0
        .update_anchors(&debug_painter, grid, &viewport);

    label_physics.0.update_labels_new(
        &debug_painter,
        grid,
        &viewer.app.annotations,
        &mut annotation_painter.0,
        &viewport,
    );

    let dt = time.delta_seconds();

    label_physics.0.step(grid, dt, &viewport);
}
*/

/*
fn draw_annotations(
    mut contexts: bevy_egui::EguiContexts,
    mut annotation_painter: ResMut<AnnotationPainter>,
    viewer: Res<super::PafViewer>,
    menubar_size: Res<super::gui::MenubarSize>,
    app_view: Res<AlignmentViewport>,
) {
    let ctx = contexts.ctx_mut();
    let mut clip_rect = ctx.screen_rect();
    clip_rect.set_top(menubar_size.height);

    annotation_painter.0.draw(
        &viewer.app.app_config.annotation_draw_config,
        ctx,
        clip_rect,
        &app_view.view,
    );
}
*/

/*
fn load_annotation_file_old(
    frame_count: Res<bevy::core::FrameCount>,
    // mut annotation_painter: ResMut<AnnotationPainter>,
    mut viewer: ResMut<super::PafViewer>,
    mut load_events: EventReader<LoadAnnotationFile>,
) -> Vec<crate::annotations::AnnotationId> {
    let mut labels_to_prepare = Vec::new();
    if frame_count.0 == 0 {
        return labels_to_prepare;
    }

    for LoadAnnotationFile { path } in load_events.read() {
        let app = &mut viewer.app;
        let grid = &app.alignment_grid;
        let annots = &mut app.annotations;

        match annots.load_bed_file(grid, &path) {
            Ok(list_id) => {
                let annot_ids = viewer
                    .app
                    .annotations
                    .list_by_id(list_id)
                    .into_iter()
                    .flat_map(|list| {
                        list.records
                            .iter()
                            .enumerate()
                            .map(|(record_id, _)| (list_id, record_id))
                    });

                labels_to_prepare.extend(annot_ids);

                log::info!("Loaded BED file `{path:?}`");
            }
            Err(err) => {
                log::error!("Error loading BED file at path `{path:?}`: {err:?}")
            }
        }
    }

    labels_to_prepare
}
*/
