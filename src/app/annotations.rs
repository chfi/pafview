use bevy::{
    math::{DVec2, U64Vec2},
    prelude::*,
    render::view::RenderLayers,
    sprite::{Anchor, MaterialMesh2dBundle},
    utils::HashMap,
};
use bevy_mod_picking::picking_core::Pickable;

use crate::{
    annotations::{AnnotationId, RecordEntryId, RecordListId},
    grid::AxisRange,
};

use super::{
    alignments::{layout::SeqPairLayout, AlignmentLayoutQuery},
    render::bordered_rect::BorderedRectMaterial2d,
    view::AlignmentViewport,
};

pub(super) struct AnnotationsPlugin;

pub mod gui;

/*

*/

impl Plugin for AnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app //.init_resource::<LabelPhysics>()
            .init_resource::<AnnotationPainter>()
            .init_resource::<Annotations>()
            .init_resource::<AnnotationEntityMap>()
            .register_type::<AnnotationEntityMap>()
            // .add_plugins(bevy_inspector_egui::quick::ResourceInspectorPlugin::<
            //     AnnotationEntityMap,
            // >::default())
            .add_event::<LoadAnnotationFile>()
            .register_type::<Annotation>()
            .register_type::<DisplayEntities>()
            .add_systems(Startup, setup)
            .add_systems(PreUpdate, load_annotation_file.pipe(prepare_annotations))
            .add_systems(
                PreUpdate,
                update_annotation_regions.after(super::view::enforce_alignment_viewport_limits),
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

#[derive(Debug, Clone, Copy, Component, Reflect)]
pub struct Annotation {
    pub record_list: RecordListId,
    pub list_index: RecordEntryId,
}

#[derive(Default, Resource, Deref, DerefMut, Reflect, Debug)]
pub struct AnnotationEntityMap(HashMap<AnnotationId, Entity>);

#[derive(Component, Reflect)]
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

// #[derive(Resource, Default)]
// struct LabelPhysics(crate::annotations::physics::LabelPhysics);

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
    // mut label_physics: ResMut<LabelPhysics>,
) {
    let mesh = meshes.add(Mesh::from(Rectangle::default()));
    commands.insert_resource(DisplayHandles { mesh: mesh.into() });

    // label_physics.0.heightfields =
    //     crate::annotations::physics::AlignmentHeightFields::from_alignments(&alignments);

    use clap::Parser;
    let args = crate::cli::Cli::parse();

    if let Some(path) = args.bed {
        load_events.send(LoadAnnotationFile { path });
    }

    if let Some(path) = args.bedpe {
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
        let Some(ext) = path.extension() else {
            continue;
        };

        let result = if ext.eq_ignore_ascii_case("bed") {
            annotations.load_bed_file(&sequences.sequence_names, &path)
        } else if ext.eq_ignore_ascii_case("bedpe") {
            annotations.load_bedpe_file(&sequences.sequence_names, &path)
        } else {
            continue;
        };

        match result {
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

#[derive(Component, Clone)]
struct AnnotationLabel;

fn prepare_annotations(
    In(labels_to_prepare): In<Vec<crate::annotations::AnnotationId>>,
    mut commands: Commands,
    mut materials: ResMut<Assets<BorderedRectMaterial2d>>,

    annotations: Res<Annotations>,
    mut annot_entity_map: ResMut<AnnotationEntityMap>,

    display_handles: Res<DisplayHandles>,
) {
    for annot_id @ (list_id, entry_id) in labels_to_prepare {
        // TODO color from annotation/name

        let record = &annotations.list_by_id(list_id).unwrap().records[entry_id];

        let color = record.color;
        let annot_color = Color::srgba_u8(color.r(), color.g(), color.b(), color.a());

        let fill_color = LinearRgba::from(annot_color.with_alpha(0.4));
        let border_color = LinearRgba::from(annot_color);

        let mat = BorderedRectMaterial2d {
            fill_color,
            border_color,
            border_width_px: 1.0,
            ..default()
        };

        let tgt_mat = materials.add(BorderedRectMaterial2d {
            border_opacities: 0x0000FFFF,
            ..mat.clone()
        });
        let qry_mat = materials.add(BorderedRectMaterial2d {
            border_opacities: 0xFFFF0000,
            ..mat
        });

        let query_region = commands
            .spawn((
                RenderLayers::layer(1),
                MaterialMesh2dBundle {
                    mesh: display_handles.mesh.clone(),
                    material: qry_mat,
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
                    material: tgt_mat,
                    ..default()
                },
            ))
            // .insert(SpatialBundle::HIDDEN_IDENTITY)
            .insert(SpatialBundle::INHERITED_IDENTITY)
            .id();

        let text_color = Color::BLACK;

        // TODO labels
        let label_bundle = (
            AnnotationLabel,
            RenderLayers::layer(1),
            Text2dBundle {
                text: Text::from_section(
                    &record.label,
                    TextStyle {
                        color: text_color.into(),
                        ..default()
                    },
                ),
                text_anchor: Anchor::TopLeft,
                visibility: Visibility::Visible,
                ..default()
            },
        );
        let query_label = commands
            .spawn(label_bundle.clone())
            .insert((Pickable::IGNORE, Anchor::TopLeft))
            .id();
        let target_label = commands
            .spawn(label_bundle)
            .insert((Pickable::IGNORE, Anchor::TopLeft))
            .id();

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
    annotations: Res<Annotations>,

    layouts: AlignmentLayoutQuery,

    alignment_view: Res<AlignmentViewport>,

    windows: Query<&Window>,

    display_ents: Query<(&Annotation, &DisplayEntities)>,
    mut transforms: Query<&mut Transform, Without<Handle<SeqPairLayout>>>,
) {
    // TODO actually use layout roots, not just the default layout asset w/o transform
    let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };
    let screen_dims = window.size();

    for (annot_id, entities) in display_ents.iter() {
        let list = annotations.list_by_id(annot_id.record_list).unwrap();
        let record = &list.records[annot_id.list_index];

        let tgt_seq_offset = layout.target_offsets.get(&record.tgt_id);
        let qry_seq_offset = layout.query_offsets.get(&record.qry_id);

        let Some((tgt_seq_offset, qry_seq_offset)) = tgt_seq_offset.zip(qry_seq_offset) else {
            return;
        };

        let seq_offsets = DVec2::new(*tgt_seq_offset, *qry_seq_offset);

        let local_p0 = U64Vec2::new(record.tgt_range.start, record.qry_range.start);
        let local_p1 = U64Vec2::new(record.tgt_range.end, record.qry_range.end);

        let p0 = seq_offsets + local_p0.as_dvec2();
        let p1 = seq_offsets + local_p1.as_dvec2();

        let s0 = alignment_view.view.map_world_to_screen(screen_dims, p0);
        let s1 = alignment_view.view.map_world_to_screen(screen_dims, p1);

        let mut mid = (s0 + s1) * 0.5;
        mid.y = screen_dims.y - mid.y;

        // hacky fix to avoid z-fighting
        let z = -1.0 - (annot_id.list_index as f32) / 1_000_000.0;

        if let Ok(mut transform) = transforms.get_mut(entities.query_region) {
            transform.translation = Vec3::new(screen_dims.x * 0.5, mid.y, z);

            let width = (s0.y - s1.y).abs().max(0.5);
            transform.scale = Vec3::new(screen_dims.x, width, 1.0);
        }

        if let Ok(mut transform) = transforms.get_mut(entities.target_region) {
            transform.translation = Vec3::new(mid.x, screen_dims.y * 0.5, z - 1.0);

            let width = (s0.x - s1.x).abs().max(0.5);
            transform.scale = Vec3::new(width, screen_dims.y, 1.0);
        }

        if let Ok(mut transform) = transforms.get_mut(entities.query_label) {
            transform.translation = Vec3::new(10.0, mid.y, z + 1.0);
        }

        if let Ok(mut transform) = transforms.get_mut(entities.target_label) {
            transform.translation = Vec3::new(mid.x, screen_dims.y - 30.0, z + 1.0);
        }
    }
}
