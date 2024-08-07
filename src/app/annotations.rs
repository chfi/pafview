use bevy::{prelude::*, render::view::RenderLayers, sprite::MaterialMesh2dBundle};

use crate::annotations::RecordListId;

use super::view::AlignmentViewport;

pub(super) struct AnnotationsPlugin;

/*

*/

impl Plugin for AnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabelPhysics>()
            .init_resource::<AnnotationPainter>()
            .init_resource::<Annotations>()
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

#[derive(Component)]
pub struct Annotation {
    pub record_list: RecordListId,
    pub list_index: usize,
}

#[derive(Component)]
struct DisplayEntities {
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
    // let material = materials.add(ColorMaterial::default());
    commands.insert_resource(DisplayHandles {
        mesh: mesh.into(),
        // material,
    });

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

    display_handles: Res<DisplayHandles>,
) {
    for (list_id, entry_id) in labels_to_prepare {
        // TODO color from annotation/name
        let color_mat = materials.add(ColorMaterial::default());

        let query_region = commands
            .spawn((
                RenderLayers::layer(1),
                MaterialMesh2dBundle {
                    mesh: display_handles.mesh.clone(),
                    material: color_mat.clone(),
                    ..default()
                },
            ))
            .insert(SpatialBundle::HIDDEN_IDENTITY)
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
            .insert(SpatialBundle::HIDDEN_IDENTITY)
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
    }
    //
}

fn update_annotation_regions(
    alignment_view: Res<AlignmentViewport>,
    annotations: Res<Annotations>,

    display_ents: Query<(&Annotation, &DisplayEntities)>,
    mut transforms: Query<&mut Transform>,
) {
    for (annot, entities) in display_ents.iter() {
        // update region transforms (screenspace) based on current view
        if let Ok(cmds) = transforms.get_mut(entities.query_region) {
            // TODO
        }

        // TODO target region too
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
