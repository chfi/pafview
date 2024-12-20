use avian2d::parry::bounding_volume::Aabb;
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
    alignments::{
        layout::{AabbQbvh, DefaultLayout, SeqPairLayout},
        AlignmentLayoutQuery,
    },
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
            .init_resource::<LabelQbvh>()
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

#[derive(Debug, Clone, Copy, Component, Reflect, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
struct LoadAnnotationFile {
    pub path: std::path::PathBuf,
}

#[derive(Event)]
enum AnnotationEvent {
    ChangeVisibility { annot_id: Annotation, visible: bool },
}

// #[derive(Resource, Default)]
// struct LabelPhysics(crate::annotations::physics::LabelPhysics);

// NB: probably want to replace the egui painter-based annotation drawing
// with something cleaner & more integrated into bevy
#[derive(Resource, Default)]
struct AnnotationPainter(pub crate::annotations::draw::AnnotationPainter);

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
                text_anchor: Anchor::Center,
                visibility: Visibility::Visible,
                ..default()
            },
        );
        let query_label = commands
            .spawn(label_bundle.clone())
            .insert(Pickable::IGNORE)
            .id();
        let target_label = commands.spawn(label_bundle).insert(Pickable::IGNORE).id();

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
    mut visibilities: Query<&mut Visibility>,
    label_sizes: Query<&bevy::text::TextLayoutInfo>,

    mut label_qbvh: ResMut<LabelQbvh>,
    // mut relevant_label_annots: Local<HashSet<Annotation>>,
) {
    // TODO actually use layout roots, not just the default layout asset w/o transform
    let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };
    let screen_dims = window.size();

    label_qbvh.qbvh = default();
    label_qbvh.workspace = default();
    label_qbvh.annot_qbvh_map.clear();

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

        let tgt_x = s0.x;
        let qry_y = s0.y;

        let Ok(label_size) = label_sizes.get(entities.target_label) else {
            continue;
        };

        let label_size = label_size.logical_size.as_dvec2();

        // target label
        let mut candidate_position = DVec2::new(tgt_x as f64, 40.0) + label_size * 0.5;
        let mut final_position: Option<DVec2> = None;

        let half_extents = label_size * 0.5;

        for _attempt in 0..10 {
            let mut cand = candidate_position;

            let mut is_colliding = false;
            label_qbvh
                .qbvh
                .aabbs_in_rect_callback(cand, half_extents, |_, aabb| {
                    is_colliding = true;
                    if aabb.mins.y < cand.y {
                        cand.y += label_size.y + 2.0;
                        // cand.y = aabb.mins.y - aabb.extents().y - half_extents.y;
                    }
                    false
                });

            if !is_colliding {
                final_position = Some(cand);
                break;
            }

            candidate_position = cand;
        }

        if let Some(pos) = final_position {
            label_qbvh.insert_or_update(*annot_id, false, pos, label_size);

            if let Ok(mut vis) = visibilities.get_mut(entities.target_label) {
                *vis = Visibility::Inherited;
            }
            if let Ok(mut transform) = transforms.get_mut(entities.target_label) {
                transform.translation =
                    Vec3::new(pos.x as f32, screen_dims.y - pos.y as f32, z + 1.0);
                // transform.translation = Vec3::new(tgt_x, screen_dims.y - 30.0, z + 1.0);
            }
        } else {
            if let Ok(mut vis) = visibilities.get_mut(entities.target_label) {
                *vis = Visibility::Hidden;
            }
        }

        // temporary query label until i get things tested & cleaned up
        //
        //

        // query label
        let mut candidate_position = DVec2::new(40.0, qry_y as f64) + label_size * 0.5;
        let mut final_position: Option<DVec2> = None;

        let half_extents = label_size * 0.5;

        for _attempt in 0..4 {
            let mut cand = candidate_position;

            let mut is_colliding = false;
            label_qbvh
                .qbvh
                .aabbs_in_rect_callback(cand, half_extents, |_, aabb| {
                    is_colliding = true;
                    if aabb.maxs.x > cand.x {
                        cand.x += label_size.x + 16.0;
                    }
                    false
                });

            if !is_colliding {
                final_position = Some(cand);
                break;
            }

            candidate_position = cand;
        }

        if let Some(pos) = final_position {
            label_qbvh.insert_or_update(*annot_id, true, pos, label_size);

            if let Ok(mut vis) = visibilities.get_mut(entities.query_label) {
                *vis = Visibility::Inherited;
            }
            if let Ok(mut transform) = transforms.get_mut(entities.query_label) {
                transform.translation =
                    Vec3::new(pos.x as f32, screen_dims.y - pos.y as f32, z + 1.0);
            }
        } else {
            if let Ok(mut vis) = visibilities.get_mut(entities.query_label) {
                *vis = Visibility::Hidden;
            }
        }

        /*

        if let Ok(mut transform) = transforms.get_mut(entities.target_label) {
            transform.translation = Vec3::new(mid.x, screen_dims.y - 30.0, z + 1.0);
        }
        */
    }
}

#[derive(Component)]
struct AnchorEntity {
    world_anchor: DVec2,
    anchor_bounds_min: DVec2,
    anchor_bounds_max: DVec2,
}

impl Default for AnchorEntity {
    fn default() -> Self {
        Self {
            world_anchor: DVec2::ZERO,
            anchor_bounds_min: DVec2::NEG_INFINITY,
            anchor_bounds_max: DVec2::INFINITY,
        }
    }
}

#[derive(Component)]
struct LabelAnchor {
    anchor: Entity,
}

#[derive(Resource, Default)]
struct LabelQbvh {
    qbvh: AabbQbvh<usize>,

    // TODO need to support at least two labels per annotation id if i want
    // separate labels for target & query
    annot_qbvh_map: HashMap<(Annotation, bool), usize>,

    workspace: avian2d::parry::partitioning::QbvhUpdateWorkspace,
}

impl LabelQbvh {
    // fn remove(&mut self, id: Annotation) {
    //     //
    // }

    fn insert_or_update(&mut self, id: Annotation, is_query: bool, screen_pos: DVec2, size: DVec2) {
        let aabb =
            Aabb::from_half_extents(screen_pos.to_array().into(), (size * 0.5).to_array().into());

        let key = (id, is_query);

        if let Some(qbvh_index) = self.annot_qbvh_map.get(&key).copied() {
            self.qbvh.qbvh.pre_update_or_insert(qbvh_index);
            self.qbvh.aabbs[qbvh_index] = aabb;
        } else {
            let qbvh_index = self.qbvh.data.len();
            self.qbvh.qbvh.pre_update_or_insert(qbvh_index);
            self.qbvh.data.push(qbvh_index);
            self.qbvh.aabbs.push(aabb);
            self.annot_qbvh_map.insert(key, qbvh_index);
        }

        self.qbvh
            .qbvh
            .refit(1.0, &mut self.workspace, |qbvh_index| {
                self.qbvh.aabbs[*qbvh_index]
            });
    }
}

/*
fn update_annotation_labels(
    mut commands: Commands,

    view: Res<AlignmentViewport>,

    // TODO handle layouts properly etc. etc.
    layouts: Res<Assets<SeqPairLayout>>,
    default_layout: Res<DefaultLayout>,

    annotations: Query<(Entity, &Annotation, &DisplayEntities)>,
    mut labels: Query<(Entity, &mut Transform), With<AnnotationLabel>>,

    mut label_qbvh: ResMut<LabelQbvh>,

    windows: Query<&Window>,
) {
    let Some(layout) = layouts.get(&default_layout.layout) else {
        return;
    };

    let Ok(screen_dims) = windows.get_single().map(|win| win.size()) else {
        return;
    };

    for (annot_ent, annot_id, display_ents) in annotations.iter() {
        // get/compute bounds for annotation region

        //
        let list = annotations.list_by_id(annot_id.record_list).unwrap();
        let record = &list.records[annot_id.list_index];

        let tgt_seq_offset = layout.target_offsets.get(&record.tgt_id);
        let qry_seq_offset = layout.query_offsets.get(&record.qry_id);

        if let Ok((label_ent, anchor)) = labels.get(display_ents.target_label) {
            //
        }

        // for (label_ent, anchor) in labels.iter() {

        //
        // }
    }
}
*/
