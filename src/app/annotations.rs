use avian2d::{
    parry::{
        self,
        bounding_volume::{Aabb as ParryAabb, BoundingVolume},
        query::{PointQuery, PointQueryWithLocation},
    },
    prelude::*,
};

use bevy::{
    math::{DVec2, U64Vec2},
    prelude::*,
    render::view::RenderLayers,
    sprite::{Anchor, MaterialMesh2dBundle, Mesh2d, Mesh2dHandle},
    text::TextLayoutInfo,
    utils::HashMap,
};
use bevy_mod_picking::picking_core::Pickable;

use crate::annotations::{AnnotationId, RecordEntryId, RecordListId};

use super::{
    alignments::{
        layout::{AabbQbvh, DefaultLayout, SeqPairLayout},
        AlignmentAabbs, AlignmentAxis, AlignmentLayoutQuery, DefaultLayoutRoot,
    },
    render::{
        bordered_rect::BorderedRectMaterial2d,
        sampled_lines::{AlignmentCollisionLines, AlignmentSamplingParams},
        MainAlignmentView,
    },
    view::AlignmentViewport,
    AlignmentIndex, SequencePairTile,
};

pub(super) struct AnnotationsPlugin;

pub mod gui;

impl Plugin for AnnotationsPlugin {
    fn build(&self, app: &mut App) {
        app //.init_resource::<LabelPhysics>()
            // .init_resource::<AnnotationPainter>()
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
            .add_systems(
                PreUpdate,
                (
                    load_annotation_file.pipe(prepare_annotations),
                    add_label_physics,
                )
                    .chain(),
            )
            .add_systems(
                PreUpdate,
                update_annotation_regions.after(super::view::enforce_alignment_viewport_limits),
            )
            .add_systems(
                PreUpdate,
                (
                    set_label_anchors,
                    update_annotation_labels,
                    label_anchor_constraints,
                )
                    .chain()
                    .after(update_annotation_regions),
            );
        // .add_systems(
        //     Update,
        //     (update_annotation_labels, draw_annotations)
        //         .chain()
        //         .after(super::gui::menubar_system),
        // );

        app.insert_gizmo_config(
            AnchorGizmos,
            GizmoConfig {
                render_layers: RenderLayers::layer(1),
                ..default()
            },
        )
        // .add_systems(Startup, |mut cfg: ResMut<GizmoConfigStore>| {
        //     todo!();
        // })
        .add_systems(PreUpdate, anchor_debug_gizmos);

        fn anchor_debug_gizmos(
            mut gizmos: Gizmos<AnchorGizmos>,
            view: Res<AlignmentViewport>,
            windows: Query<&Window>,
            labels: Query<(&Position, &LabelAnchor)>,
        ) {
            let Ok(win_size) = windows.get_single().map(|w| w.size()) else {
                return;
            };

            let color = Color::hsl(270.0, 0.8, 0.5);

            for (label_pos, anchor) in labels.iter() {
                let mut anchor_pos = Vec2::from(
                    *view
                        .view
                        .map_world_to_screen(win_size, anchor.world_point)
                        .as_array(),
                );
                anchor_pos.y = win_size.y - anchor_pos.y;
                let mut label_pos = label_pos.0.as_vec2();
                // label_pos.y = win_size.y - label_pos.y;

                gizmos.circle_2d(anchor_pos, 5.0, color);
                gizmos.line_2d(anchor_pos, label_pos, color);
                // gizmos.line_2d(anchor_pos, label_pos.0.as_vec2(), color);
                // gizmos.circle_2d(ancho)
                //
            }
        }

        // #[derive(Component)]
        // struct TestBox;

        // fn debug_box(boxes: Query<(Entity, &Transform), With<TestBox>>) {
        //     for (box_, transform) in boxes.iter() {
        //         let pos = transform.translation;
        //         println!("box {box_:?} - {pos:?}");
        //     }
        // }

        // fn drop_box(
        //     mut commands: Commands,

        //     mut meshes: ResMut<Assets<Mesh>>,
        //     // mut mats: ResMut<Assets<BorderedRectMaterial2d>>,
        //     mut mats: ResMut<Assets<BorderedRectMaterial2d>>,
        //     mut assets: Local<Option<(Mesh2dHandle, Handle<BorderedRectMaterial2d>)>>,

        //     button: Res<ButtonInput<MouseButton>>,
        //     window: Query<&Window>,
        // ) {
        //     if assets.is_none() {
        //         let mesh = meshes.add(Rectangle::from_size([100.0, 100.0].into()));
        //         let color = Color::hsl(240.0, 1.0, 0.5).to_linear();
        //         let mat = mats.add(BorderedRectMaterial2d {
        //             fill_color: color,
        //             border_color: color,
        //             ..default()
        //         });
        //         *assets = Some((Mesh2dHandle(mesh), mat));
        //     }
        //     let Some((mesh, mat)) = assets.as_ref() else {
        //         return;
        //     };

        //     let Some((cursor, dims)) = window
        //         .get_single()
        //         .ok()
        //         .and_then(|w| Some((w.cursor_position()?, w.size())))
        //     else {
        //         return;
        //     };

        //     if button.just_pressed(MouseButton::Left) {
        //         println!("spawning box at {cursor:?}");
        //         commands.spawn((
        //             TestBox,
        //             SpatialBundle::from_transform(Transform::from_xyz(
        //                 cursor.x,
        //                 dims.y - cursor.y,
        //                 0.0,
        //             )),
        //             mesh.clone(),
        //             mat.clone(),
        //             RenderLayers::layer(1),
        //             Mass(10.0),
        //             Inertia(5.0),
        //             LinearVelocity(DVec2::new(0.0, -100.0)),
        //             Collider::rectangle(100.0, 100.0),
        //             RigidBody::Dynamic,
        //         ));
        //     }
        // }
        // app.add_systems(PreUpdate, debug_box);
        // app.add_systems(PostUpdate, drop_box);

        // fn print_collisions(query: Query<(Entity, &CollidingEntities)>) {
        //     for (entity, colliding_entities) in &query {
        //         println!(
        //             "{:?} is colliding with the following entities: {:?}",
        //             entity, colliding_entities
        //         );
        //     }
        // }
        // app.add_systems(PostUpdate, print_collisions);
    }
}

#[derive(Default, Reflect, GizmoConfigGroup)]
struct AnchorGizmos;

#[derive(Resource, Default, Deref, DerefMut)]
pub struct Annotations(pub crate::annotations::AnnotationStore);

impl Annotations {
    pub fn get_record(&self, annot: &Annotation) -> Option<&crate::annotations::Record> {
        let list = self.list_by_id(annot.record_list)?;
        list.records.get(annot.list_index)
    }
}

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

#[derive(PhysicsLayer, Clone, Copy)]
pub enum LabelPhysicsLayers {
    ActiveLabel,
    InactiveLabel,
}

#[derive(Component, Clone, Copy)]
struct AnnotationLabel {
    annotation: Entity,
    axis: AlignmentAxis,
    is_active: bool,
}

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

        let annot_ent = commands
            .spawn(Annotation {
                record_list: list_id,
                list_index: entry_id,
            })
            .id();

        let label_bundle = (
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

        // let query_label = commands
        //     .spawn(label_bundle.clone())
        //     .insert((
        //         Pickable::IGNORE,
        //         AnnotationLabel {
        //             annotation: annot_ent,
        //             axis: AlignmentAxis::Query,
        //             is_active: false,
        //         },
        //     ))
        //     .id();
        let target_label = commands
            .spawn(label_bundle)
            .insert((
                Pickable::IGNORE,
                AnnotationLabel {
                    annotation: annot_ent,
                    axis: AlignmentAxis::Target,
                    is_active: false,
                },
            ))
            .id();

        commands.entity(annot_ent).insert(DisplayEntities {
            query_region,
            // query_label,
            query_label: Entity::PLACEHOLDER,
            target_region,
            target_label,
        });

        annot_entity_map.insert(annot_id, annot_ent);
    }
    //
}

fn add_label_physics(
    mut commands: Commands,

    annot_labels: Query<(Entity, &TextLayoutInfo), (With<AnnotationLabel>, Without<Collider>)>,
) {
    for (label, text_info) in annot_labels.iter() {
        let label_size = text_info.logical_size.as_dvec2();
        if label_size.length_squared() == 0.0 {
            continue;
        }
        commands.entity(label).insert((
            RigidBody::Dynamic,
            Collider::rectangle(label_size.x, label_size.y),
            CollisionLayers::NONE,
            Mass(100.0),
            Inertia(1.0),
            LinearDamping(0.1),
            LockedAxes::ROTATION_LOCKED,
        ));
    }
}

/*
fn toggle_label_physics_activity(
    // mut labels:
    // mut labels: Query<(Entity, &TextLayoutInfo), (With<AnnotationLabel>, Without<Collider>)>,
    mut labels: Query<(&mut RigidBody, &mut CollisionLayers),
) {

}
*/

fn update_annotation_regions(
    annotations: Res<Annotations>,

    layouts: AlignmentLayoutQuery,

    alignment_view: Res<AlignmentViewport>,

    windows: Query<&Window>,

    // alignment_collision: Query<&AlignmentCollisionLines>,
    display_ents: Query<(&Annotation, &DisplayEntities)>,
    mut transforms: Query<&mut Transform, Without<Handle<SeqPairLayout>>>,
    // mut visibilities: Query<&mut Visibility>,
    label_sizes: Query<&bevy::text::TextLayoutInfo>,
    // mut label_qbvh: ResMut<LabelQbvh>,
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

    // label_qbvh.qbvh = default();
    // label_qbvh.workspace = default();
    // label_qbvh.annot_qbvh_map.clear();

    for (annot_id, entities) in display_ents.iter() {
        let list = annotations.list_by_id(annot_id.record_list).unwrap();
        let record = &list.records[annot_id.list_index];

        let Some([s0, s1]) = layout.map_local_region_to_screen(
            &alignment_view.view,
            screen_dims,
            (record.tgt_id, record.tgt_range.clone()),
            (record.qry_id, record.qry_range.clone()),
        ) else {
            continue;
        };

        // let tgt_seq_offset = layout.target_offsets.get(&record.tgt_id);
        // let qry_seq_offset = layout.query_offsets.get(&record.qry_id);

        // let Some((tgt_seq_offset, qry_seq_offset)) = tgt_seq_offset.zip(qry_seq_offset) else {
        //     return;
        // };

        // let seq_offsets = DVec2::new(*tgt_seq_offset, *qry_seq_offset);

        // let local_p0 = U64Vec2::new(record.tgt_range.start, record.qry_range.start);
        // let local_p1 = U64Vec2::new(record.tgt_range.end, record.qry_range.end);

        // let p0 = seq_offsets + local_p0.as_dvec2();
        // let p1 = seq_offsets + local_p1.as_dvec2();

        // let s0 = alignment_view.view.map_world_to_screen(screen_dims, p0);
        // let s1 = alignment_view.view.map_world_to_screen(screen_dims, p1);

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
        /*
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
        */
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
    // anchor_bounds_min: DVec2,
    // anchor_bounds_max: DVec2,
}

impl Default for AnchorEntity {
    fn default() -> Self {
        Self {
            world_anchor: DVec2::ZERO,
            // anchor_bounds_min: DVec2::NEG_INFINITY,
            // anchor_bounds_max: DVec2::INFINITY,
        }
    }
}

#[derive(Component, Debug)]
struct LabelAnchor {
    world_point: DVec2,
    normal: DVec2,
    anchor_alignment: AlignmentIndex,
    // valid_region: ParryAabb,
}

fn set_label_anchors(
    mut commands: Commands,

    default_layout_root: Res<DefaultLayoutRoot>,
    layout_query: AlignmentLayoutQuery,

    alignment_aabbs: Res<AlignmentAabbs>,

    viewport: Res<AlignmentViewport>,

    annotations: Res<Annotations>,
    annotation_query: Query<(Entity, &Annotation)>,

    alignment_samplers: Query<(&AlignmentCollisionLines, &AlignmentSamplingParams)>,

    labels: Query<(Entity, &AnnotationLabel, Option<&LabelAnchor>)>,

    mut last_sampled: Local<Option<AlignmentSamplingParams>>,
    // time: Res<Time>,
    mut gizmos: Gizmos<AnchorGizmos>,
) {
    // if time.elapsed_seconds() < 3.0 {
    //     return;
    // }

    let layout = layout_query
        .layout_roots
        .get(default_layout_root.0)
        .ok()
        .and_then(|(_, _transform, layout_handle, _tile_entities)| {
            layout_query.layout_assets.get(layout_handle)
        });

    let Some(layout) = layout else {
        return;
    };

    let Ok((alignment_lines, sampling_params)) = alignment_samplers.get_single() else {
        return;
    };

    let screen_dims = sampling_params.canvas_size;

    let resample = last_sampled
        .as_ref()
        .map(|s| s != sampling_params)
        .unwrap_or(true);

    if resample {
        *last_sampled = Some(*sampling_params);
    }

    let mut updated_anchors = 0;
    let mut anchor_count = 0;
    let mut anchor_outside_region = 0;
    let mut skipped = 0;

    // dbg!(&alignment_lines.polylines);

    let view = viewport.view;

    for (label_ent, label_annot, old_anchor) in labels.iter() {
        anchor_count += 1;

        // recreate if prev anchor point is out of view bounds

        let Some(record) = annotation_query
            .get(label_annot.annotation)
            .ok()
            .and_then(|(_, annot)| annotations.get_record(annot))
        else {
            continue;
        };

        let Some(tile_offset) = layout
            .target_offsets
            .get(&record.tgt_id)
            .zip(layout.query_offsets.get(&record.qry_id))
            .map(|(&x, &y)| DVec2::new(x, y))
        else {
            continue;
        };

        // if recreate_anchor {
        // find intersection of annotated region with view
        // but this should be in screenspace/pixels, since we're
        // working with the screen-sampled alignments
        let intersecting_region: ParryAabb = {
            let (x_min, x_max, y_min, y_max) = match label_annot.axis {
                AlignmentAxis::Target => {
                    let y_min = view.y_min;
                    let y_max = view.y_max;

                    let xs = record.tgt_range_f64();
                    let x_min = (xs.start() + tile_offset.x).clamp(view.x_min, view.x_max);
                    let x_max = (xs.end() + tile_offset.x).clamp(view.x_min, view.x_max);

                    // let x_min = xs.start().clamp(view.x_min, view.x_max);
                    // let x_max = xs.end().clamp(view.x_min, view.x_max);

                    (x_min, x_max, y_min, y_max)
                }
                #[allow(unreachable_code)]
                AlignmentAxis::Query => {
                    continue;
                    let x_min = view.x_min;
                    let x_max = view.x_max;

                    let ys = record.qry_range_f64();
                    let y_min = (ys.start() + tile_offset.y).clamp(view.y_min, view.y_max);
                    let y_max = (ys.end() + tile_offset.y).clamp(view.y_min, view.y_max);

                    (x_min, x_max, y_min, y_max)
                }
            };

            let mins = DVec2::new(x_min, y_min);
            let maxs = DVec2::new(x_max, y_max);

            let vw = view.width();
            let vh = view.height();

            let s_size = sampling_params.canvas_size.as_dvec2();
            let mut mins = view.map_world_to_screen(sampling_params.canvas_size, mins);
            let mut maxs = view.map_world_to_screen(sampling_params.canvas_size, maxs);

            mins.y = sampling_params.canvas_size.y - mins.y;
            maxs.y = sampling_params.canvas_size.y - maxs.y;

            // let x_min = s_size.x * (x_min - view.x_min) / vw;
            // let x_max = s_size.x * (x_max - view.x_min) / vw;
            // let y_min = s_size.y * (y_min - view.y_min) / vh;
            // let y_max = s_size.y * (y_max - view.y_min) / vh;

            // ParryAabb::new([x_min, y_min].into(), [x_max, y_max].into())
            ParryAabb::new(
                [mins.x as f64, mins.y as f64].into(),
                [maxs.x as f64, maxs.y as f64].into(),
            )
        };

        let mut recreate_anchor = false;

        if let Some(prev) = old_anchor {
            let mut sp = view.map_world_to_screen(sampling_params.canvas_size, prev.world_point);
            sp.y = sampling_params.canvas_size.y - sp.y;
            if !intersecting_region
                .contains_local_point(&nalgebra::Point2::new(sp.x as f64, sp.y as f64))
            {
                // anchor_outside_region += 1;
                recreate_anchor = true;
                commands.entity(label_ent).remove::<LabelAnchor>();
            }
        } else {
            recreate_anchor = true;
        }

        // let recreate_anchor = old_anchor
        //     .map(|prev| {})
        //     // .map(|prev| !view.contains_point(prev.world_point))
        //     .unwrap_or(true)
        //     || resample;

        // use the `intersecting_region` AABB to query the QBVH built from the AABBs
        // of the sampled screenspace alignment lines

        // let mut best_alignment: Option<(AlignmentIndex, &parry::shape::Polyline, DVec2)> = None;
        let mut best_alignment: Option<(AlignmentIndex, DVec2, DVec2)> = None;
        let region_pt = intersecting_region.center();
        let pt = DVec2::from(region_pt.coords.data.0[0]);

        let size: DVec2 = intersecting_region.extents().data.0[0].into();
        gizmos.rect_2d(
            pt.as_vec2(),
            0.0,
            size.as_vec2(),
            Color::hsl(150.0, 0.9, 0.5).with_alpha(0.3),
        );

        if recreate_anchor {
            let mut get_color = {
                let mut i = 0;
                move || {
                    let hue = i;
                    let d = 27;
                    i = (i + d) % 360;
                    Color::hsl(hue as f32, 0.8, 0.5)
                }
            };
            alignment_lines.qbvh.aabbs_in_rect_callback(
                intersecting_region.center(),
                intersecting_region.half_extents(),
                // intersecting_region.half_extents() * 1_000.0,
                |key @ (_layout_root, al_index), polyline_aabb| {

                    {
                        use bevy::math::vec2;
                        let c = polyline_aabb.center();
                        let s = polyline_aabb.extents();
                        let c = vec2(c.x as f32, screen_dims.y - c.y as f32);
                        let s = vec2(s.x as f32, s.y as f32);
                        gizmos.rect_2d(
                            c,
                            0.0,
                            s,
                            Color::hsl(90.0, 0.9, 0.2).with_alpha(0.3),
                        );
                        gizmos.rect_2d(
                            c,
                            0.0,
                            // Rot2::degrees(90.0),
                            s * 0.9,
                            Color::hsl(90.0, 0.9, 0.2).with_alpha(0.3),
                        );
                    }

                    // check if polyline is actually inside region...?

                    // if !polyline_aabb.intersects(&intersecting_region) {
                    // dbg!();
                    if !polyline_aabb.intersects(&intersecting_region) {
                        println!("polyline AABB does not intersect view: {polyline_aabb:?} vs {intersecting_region:?}");
                        return true;
                    }

                    let Some(polyline) = alignment_lines.polylines.get(&key) else {
                        // dbg!();
                        return true;
                    };


                    let (mut closest_point, (seg_id, location)) =
                        polyline.project_local_point_and_get_location(&region_pt, true);

                    if !intersecting_region
                        .contains_local_point(&nalgebra::Point2::new(closest_point.point.x, closest_point.point.y))
                    {
                        let closest = alignment_lines.closest_point_to_aabb_sides(AlignmentAxis::Target, &intersecting_region);

                        if let [Some((_, a)), Some((_, b))] = closest {
                            let mins = a.min(b);
                            let maxs = a.max(b);
                            let top = mins.y;
                            let left = DVec2::new(mins.x, top);
                            // let right = DVec2::new(maxs.x, top);
                            closest_point.point.x = mins.x;
                            closest_point.point.y = mins.y;


                        } else {
                            skipped += 1;
                            return true;
                        }
                        // anchor_outside_region += 1;
                        // commands.entity(label_ent).remove::<LabelAnchor>();
                        // continue;
                        // recreate_anchor = true;
                    }

                    let closest = DVec2::from(closest_point.point.coords.data.0[0]);
                    {
                    let mut p = closest.as_vec2();
                    p.y = screen_dims.y - p.y;
                    gizmos.rect_2d(
                        p,
                        0.0,
                        [10.0, 10.0].into(),
                        get_color(),
                        // Color::hsl(10.0, 0.9, 0.5).with_alpha(0.8),
                    );
                    }

                    // let [[cx, cy]] = polyline
                    //     .project_local_point(&region_pt, true)
                    //     .point
                    //     .coords
                    //     .data
                    //     .0;
                    // let closest = DVec2::new(cx, cy);

                    let dist = closest.distance(pt);

                    let prev_best = best_alignment
                        .as_ref()
                        .map(|(_, _, prev)| prev.distance(pt))
                        .unwrap_or(std::f64::INFINITY);

                    // dbg!(al_index, dist, closest);
                    if dist < prev_best {
                        let segment = polyline.segment(seg_id);
                        let normal = segment.normal().map(|n| DVec2::from(n.data.0[0])).unwrap_or([1.0, 0.0].into());

                        // dbg!(dist, prev_best);
                        best_alignment = Some((al_index, normal, closest));
                        // best_alignment = Some((al_index, polyline, closest));
                    }

                    true
                },
            );

            // dbg!();
            let Some((anchor_alignment, normal, closest_point)) = best_alignment else {
                continue;
            };

            {
                let p = closest_point.as_vec2();
                gizmos.rect_2d(
                    p,
                    0.0,
                    [10.0, 10.0].into(),
                    get_color(),
                    // Color::hsl(10.0, 0.9, 0.5).with_alpha(0.8),
                );
                gizmos.line_2d(p - Vec2::ONE * -5.0, p + Vec2::ONE * 5.0, get_color());
            }

            if !intersecting_region
                .contains_local_point(&nalgebra::Point2::new(closest_point.x, closest_point.y))
            {
                anchor_outside_region += 1;
                commands.entity(label_ent).remove::<LabelAnchor>();
                continue;
                // recreate_anchor = true;
            }

            let world_point =
                view.map_screen_to_world(sampling_params.canvas_size, closest_point.as_vec2());
            // let world_point = closest_point;

            // update the label with the `LabelAnchor` component
            let anchor = LabelAnchor {
                world_point: (*world_point.as_array()).into(),
                normal,
                anchor_alignment,
            };

            updated_anchors += 1;

            commands.entity(label_ent).insert(anchor);
            // dbg!(label_ent);
        }
    }

    println!(
        "{anchor_outside_region} out of {anchor_count} anchors are outside their annotation region\tskipped: {skipped}"
    );
    // if updated_anchors > 0 {
    //     println!("updated {updated_anchors} out of {anchor_count} anchors");
    // }

    //
}

fn update_annotation_labels(
    mut commands: Commands,

    annotations: Res<Annotations>,
    layouts: AlignmentLayoutQuery,
    alignment_view: Res<AlignmentViewport>,
    windows: Query<&Window>,

    main_alignment_sampler: Query<
        (&AlignmentCollisionLines, &AlignmentSamplingParams),
        With<MainAlignmentView>,
    >,

    mut labels: Query<(
        Entity,
        &Collider,
        &mut AnnotationLabel,
        Option<&LabelAnchor>,
        &mut CollisionLayers,
        &mut Visibility,
    )>,
    mut label_positions: Query<&mut Position, With<LabelAnchor>>,
) {
    /*

    */

    let Ok((alignment_lines, sampling_params)) = main_alignment_sampler.get_single() else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    use rand::prelude::*;
    let mut rng = thread_rng();

    let mut anchorless = 0;
    let mut total = 0;

    // #[allow(unreachable_code)]
    for (
        label_ent,
        label_collider,
        mut annot_label,
        anchor,
        mut collision_layers,
        mut visibility,
    ) in labels.iter_mut()
    {
        total += 1;
        if anchor.is_none() {
            anchorless += 1;
        }
        let Some(collider) = label_collider.shape().as_shape::<parry::shape::Cuboid>() else {
            continue;
        };

        if annot_label.is_active {
            // deactivate the label if the anchor does not exist...
            //
            // TODO/NB: the anchor should be removed by another system if its region
            // is out of the view bounds
            if anchor.is_none() {
                annot_label.is_active = false;
                *collision_layers = CollisionLayers::NONE;
                *visibility = Visibility::Hidden;
            }
        } else {
            if let Some(anchor) = anchor {
                *collision_layers = CollisionLayers::new(
                    [LabelPhysicsLayers::ActiveLabel],
                    [LabelPhysicsLayers::ActiveLabel],
                );
                annot_label.is_active = true;
                *visibility = Visibility::Inherited;

                // let center = [anchor.world_point.x, window.size().y as f64 * 0.5];
                let center = [
                    anchor.world_point.x - collider.half_extents.x,
                    window.size().y as f64 * 0.5,
                ];
                let half_extents = [collider.half_extents.x, window.size().y as f64 * 0.5];
                let tgt_region_aabb =
                    ParryAabb::from_half_extents(center.into(), half_extents.into());
                let edges = alignment_lines
                    .closest_point_to_aabb_sides(AlignmentAxis::Target, &tgt_region_aabb);

                if let [Some((_, p_left)), Some((_, p_right))] = edges {
                    let mins = p_left.min(p_right);

                    let s = sampling_params
                        .view
                        .map_world_to_screen(sampling_params.canvas_size, anchor.world_point);

                    let y = mins.y - 80.0;
                    // let y = intersect_aabb.center().y - i
                    if let Ok(mut pos) = label_positions.get_mut(label_ent) {
                        pos.x = s.x as f64;
                        pos.y = s.y as f64;
                        // pos.x = s.x as f64;
                        // pos.y = y;
                    }
                }
            }
        }
    }

    // println!("{anchorless} out of {total} labels have no anchor");
}

// TODO - apply/simulate forces between anchor point and screen-space label
fn label_anchor_constraints(
    // gravity: Res<Gravity>,
    //
    mut labels: Query<(
        Entity,
        &AnnotationLabel,
        &Position,
        &mut ExternalForce,
        &LabelAnchor,
    )>,

    viewport: Res<AlignmentViewport>,
    windows: Query<&Window>,
) {
    const FORCE_CONSTANT: f64 = 1_000.0;

    let Ok(screen_dims) = windows.get_single().map(|w| w.size()) else {
        return;
    };

    for (label_ent, annot_label, position, mut force, anchor) in labels.iter_mut() {
        let p0 = position.0;

        let p1 = viewport
            .view
            .map_world_to_screen(screen_dims, anchor.world_point);

        let p1 = Vec2::new(p1.x, p1.y).as_dvec2();

        let dist = p0.distance(p1);

        if let Some(dir) = (p1 - p0).try_normalize() {
            force.set_force(dir * dist.max(0.0) * FORCE_CONSTANT);
        } else {
            force.clear();
        }

        // if dist > 200.0 {
        //     if let Some(dir) = (p1 - p0).try_normalize() {
        //         position.0 += dir * dist.max(0.0);
        //     }
        // }

        //
    }
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
        let aabb = ParryAabb::from_half_extents(
            screen_pos.to_array().into(),
            (size * 0.5).to_array().into(),
        );

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
