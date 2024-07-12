use bevy::prelude::*;

use crate::annotations::RecordListId;

use super::view::AlignmentViewport;

pub(super) struct AnnotationsPlugin;

/*

*/

impl Plugin for AnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LabelPhysics>()
            .init_resource::<AnnotationPainter>()
            .add_event::<LoadAnnotationFile>()
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, load_annotation_file.pipe(prepare_annotations))
            .add_systems(
                Update,
                (update_annotation_labels, draw_annotations)
                    .chain()
                    .after(super::gui::menubar_system),
            );
    }
}

// #[derive(Component)]
// struct Annotation {
//     record_list: RecordListId,
//     list_index: usize,
// }

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

fn setup(
    // mut commands: Commands,
    viewer: Res<super::PafViewer>,
    mut load_events: EventWriter<LoadAnnotationFile>,
    mut label_physics: ResMut<LabelPhysics>,
) {
    label_physics.0.heightfields =
        crate::annotations::physics::AlignmentHeightFields::from_alignments(&viewer.app.alignments);

    use clap::Parser;
    let args = crate::cli::Cli::parse();

    if let Some(path) = args.bed {
        load_events.send(LoadAnnotationFile { path });
    }
}

fn update_annotation_labels(
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

fn draw_annotations(
    mut contexts: bevy_egui::EguiContexts,
    mut annotation_painter: ResMut<AnnotationPainter>,
    viewer: Res<super::PafViewer>,
    app_view: Res<AlignmentViewport>,
) {
    // menubar clip rect would be nice... how should i get that?
    // store it on the menubar after its render system runs,
    // & then just schedule this system after?

    let ctx = contexts.ctx_mut();
    let clip_rect = ctx.screen_rect();

    annotation_painter.0.draw(
        &viewer.app.app_config.annotation_draw_config,
        ctx,
        clip_rect,
        &app_view.view,
    );
}

fn load_annotation_file(
    frame_count: Res<bevy::core::FrameCount>,
    mut annotation_painter: ResMut<AnnotationPainter>,
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

        match annots.load_bed_file(grid, &mut annotation_painter.0, &path) {
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

fn prepare_annotations(
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
