use std::sync::{atomic::AtomicBool, Arc};

use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::*;

use super::AlignmentColorSchemes;
use crate::{sequences::SeqId, PafViewerApp};

pub mod layout;

use layout::SeqPairLayout;

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentEntityIndex>()
            .init_resource::<SequencePairEntityIndex>()
            .add_plugins(layout::AlignmentLayoutPlugin);

        app.add_systems(Startup, initialize_default_layout);

        app.add_systems(
            Startup,
            prepare_alignments.after(super::setup_screenspace_camera),
        )
        .add_systems(PreUpdate, update_seq_pair_transforms);
        //
    }
}

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct AlignmentEntityIndex(pub HashMap<AlignmentIndex, Entity>);

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct SequencePairEntityIndex(pub HashMap<SequencePairTile, Entity>);

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct SequencePairTile {
    pub target: SeqId,
    pub query: SeqId,
}

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct AlignmentIndex {
    pub query: SeqId,
    pub target: SeqId,

    pub pair_index: usize,
}

fn update_seq_pair_transforms(
    default_layout: Res<layout::DefaultLayout>,

    mut seq_pair_tiles: Query<(&mut Transform, &SequencePairTile)>,
) {
    for (mut transform, tile_key) in seq_pair_tiles.iter_mut() {
        let Some(aabb) = default_layout.layout.aabbs.get(tile_key) else {
            continue;
        };

        let center = aabb.center();
        transform.translation = Vec3::new(center.x as f32, center.y as f32, 0.0);
    }
}

// create the initial sequence pair layout from the application input data,
// and create the DefaultLayout resource
pub(super) fn initialize_default_layout(
    mut commands: Commands,

    // cli_args: Res<crate::cli::Cli>,
    sequences: Res<crate::Sequences>,
) {
    // use alignments::layout::*;

    let mut seqs = sequences
        .sequences
        .iter()
        .map(|(id, seq)| (*id, seq.len()))
        .collect::<Vec<_>>();
    seqs.sort_by_key(|(_, l)| *l);

    let targets = seqs.iter().map(|(i, _)| *i);
    let queries = targets.clone();

    let builder = layout::LayoutBuilder::from_axes(targets, queries);
    // LayoutBuilder::from_axes(targets, queries).with_vertical_offset(Some(10_000_000.0));

    let layout = builder.clone().build(&sequences);
    let default_layout = layout::DefaultLayout::new(layout, builder);

    commands.insert_resource(default_layout);
}

/*


*/

//
pub(super) fn spawn_layout_children(
    //
    mut commands: Commands,

    layouts: Res<Assets<SeqPairLayout>>,
    mut layout_events: EventReader<layout::LayoutChangedEvent>,

    // layout_roots: Query<(Entity, &Handle<SeqPairLayout>), Added<Handle<SeqPairLayout>>>,
    layout_roots: Query<(Entity, &Handle<SeqPairLayout>)>,
) {
    for layout_event in layout_events.read() {
        if !layout_event.need_respawn {
            continue;
        }

        let Ok((root, layout_handle)) = layout_roots.get(layout_event.entity) else {
            continue;
        };

        let Some(layout) = layouts.get(layout_handle) else {
            continue;
        };

        commands.entity(root).despawn_descendants();

        for (seq_pair, aabb) in layout.aabbs.iter() {
            //

            commands.spawn((
                *seq_pair,
                SpatialBundle::INHERITED_IDENTITY,
                Pickable {
                    should_block_lower: false,
                    is_hoverable: true,
                },
            ));
        }
    }
}

pub(super) fn spawn_alignments_in_tiles(
    mut commands: Commands,

    alignments: Res<crate::Alignments>,
    seq_pair_tiles: Query<(Entity, &SequencePairTile), Without<Children>>,
) {
    for (tile_ent, seq_pair) in seq_pair_tiles.iter() {
        let Some(al_indices) = alignments.indices.get(&(seq_pair.target, seq_pair.query)) else {
            continue;
        };

        let tile_als = al_indices
            .iter()
            .filter_map(|&ix| Some((ix, alignments.alignments.get(ix)?)));

        commands.entity(tile_ent).with_children(|parent| {
            for (pair_index, alignment) in tile_als {
                parent.spawn((
                    AlignmentIndex {
                        target: alignment.target_id,
                        query: alignment.query_id,
                        pair_index,
                    },
                    Pickable {
                        should_block_lower: false,
                        is_hoverable: true,
                    },
                    // On::<Pointer<Out>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                    // On::<Pointer<Over>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                ));
            }
        });
    }
}

pub(super) fn prepare_alignment_vertices(
    mut commands: Commands,

    alignments: Res<crate::Alignments>,
    alignment_query: Query<
        (Entity, &AlignmentIndex),
        Without<Handle<super::render::AlignmentVertices>>,
        // (Without<Handle<super::render::AlignmentVertices>>,),
    >,

    mut alignment_vertices_map: ResMut<super::render::AlignmentVerticesIndex>,
    mut alignment_vertices: ResMut<Assets<super::render::AlignmentVertices>>,

    // might want to do multiple alignments per task
    mut tasks: Local<HashMap<AlignmentIndex, Task<super::render::AlignmentVertices>>>,
    // mut tasks: Local<HashMap<AlignmentIndex, Task<Vec<(Vec2, Vec2, crate::CigarOp)>>>>,
    // processing: Local<HashMap<AlignmentIndex, Arc<AtomicBool>>>,
) {
    let task_pool = AsyncComputeTaskPool::get();

    for (al_ent, al_ix) in alignment_query.iter() {
        if alignment_vertices_map.vertices.contains_key(al_ix) || tasks.contains_key(al_ix) {
            continue;
        }

        let Some((location, cigar)) = alignments
            .get(*al_ix)
            .map(|al| (al.location.clone(), al.cigar.clone()))
        else {
            continue;
        };

        let task = task_pool.spawn(async move {
            super::render::AlignmentVertices::from_location_and_cigar(&location, &cigar)
        });

        tasks.insert(*al_ix, task);
    }

    let mut complete_tasks = Vec::new();

    for (&al_ix, task) in tasks.iter_mut() {
        if !task.is_finished() {
            continue;
        }

        let Some(vertices) = bevy::tasks::block_on(bevy::tasks::poll_once(task)) else {
            continue;
        };

        alignment_vertices_map
            .vertices
            .insert(al_ix, alignment_vertices.add(vertices));

        complete_tasks.push(al_ix);
    }

    for al_ix in complete_tasks {
        tasks.remove(&al_ix);
    }
}

pub(super) fn insert_alignment_polyline_materials(
    mut commands: Commands,

    // alignments: Res<crate::Alignments>,
    vertex_index: Res<super::render::AlignmentVerticesIndex>,
    mut alignment_materials: ResMut<Assets<super::render::AlignmentPolylineMaterial>>,
    color_schemes: Res<AlignmentColorSchemes>,

    cli_args: Res<crate::cli::Cli>,

    // layout_roots: Query<(Entity, &Handle<SeqPairLayout>, &Children)>,
    // seq_pair_tiles: Query<(&SequencePairTile, &Children)>,
    alignment_query: Query<
        (Entity, &AlignmentIndex),
        (
            Without<Handle<super::render::AlignmentPolylineMaterial>>,
            Without<Handle<super::render::AlignmentVertices>>,
        ),
    >,
) {
    if cli_args.low_mem {
        return;
    }

    /*
    for (layout_root, layout_handle, root_children) in layout_roots.iter() {


        for (seq_pair, tile_children) in seq_pair_tiles.iter_many(root_children) {

        }



    }
    */

    for (entity, al_ix) in alignment_query.iter() {
        let Some(vertices) = vertex_index.vertices.get(al_ix) else {
            continue;
        };

        // create the polyline material; place at origin since `update_polyline_materials` should run after
        let material = super::render::AlignmentPolylineMaterial::from_offset_and_colors(
            [0.0, 0.0],
            color_scheme.clone(),
        );
        // let material = super::render::AlignmentPolylineMaterial::from_alignment(
        //     grid,
        //     alignment,
        //     color_scheme.clone(),
        // );
        todo!();
    }
}

pub(super) fn update_alignment_polyline_materials(
    mut commands: Commands,

    mut layout_events: EventReader<layout::LayoutChangedEvent>,
    mut alignment_materials: ResMut<Assets<super::render::AlignmentPolylineMaterial>>,

    seq_pair_tiles: Query<(&SequencePairTile, &Children)>,
    alignments_query: Query<(
        Entity,
        &AlignmentIndex,
        &Handle<super::render::AlignmentPolylineMaterial>,
    )>,
) {
    for event in layout_events.read() {
        for (seq_pair, children) in seq_pair_tiles.iter() {
            let new_model: Mat4 = todo!();

            let mut child_iter = alignments_query.iter_many_mut(children);

            while let Some((_ent, al_ix, mat_handle)) = child_iter.fetch_next() {
                //
            }
        }
    }
}

// updates `Transform` component of children of layout roots
// run after spawn_layout_children
pub(super) fn update_layout_tile_positions(
    mut commands: Commands,

    layouts: Res<Assets<SeqPairLayout>>,

    mut layout_events: EventReader<layout::LayoutChangedEvent>,
    layout_roots: Query<(Entity, &Handle<SeqPairLayout>, &Children)>,

    mut seq_pair_tiles: Query<(&SequencePairTile, &mut Transform)>,
) {
    let layout_entities = layout_events.read().map(|ev| ev.entity);
    for (root, layout_handle, children) in layout_roots.iter_many(layout_entities) {
        let Some(layout) = layouts.get(layout_handle) else {
            continue;
        };
        let mut child_iter = seq_pair_tiles.iter_many_mut(children);

        while let Some((child_tile, mut transform)) = child_iter.fetch_next() {
            let Some(aabb) = layout.aabbs.get(child_tile) else {
                continue;
            };

            let mid = aabb.center();
            transform.translation.x = mid.x as f32;
            transform.translation.y = mid.y as f32;
        }
    }
}

pub(super) fn prepare_alignments(
    mut commands: Commands,
    alignments: Res<crate::Alignments>,
    sequences: Res<crate::Sequences>,
    alignment_grid: Res<crate::AlignmentGrid>,
    color_schemes: Res<AlignmentColorSchemes>,
    cli_args: Res<crate::cli::Cli>,

    mut seq_pair_entity_index: ResMut<SequencePairEntityIndex>,
    mut alignment_entity_index: ResMut<AlignmentEntityIndex>,
    mut vertex_index: ResMut<super::render::AlignmentVerticesIndex>,

    mut meshes: ResMut<Assets<Mesh>>,
    mut border_rect_materials: ResMut<Assets<super::render::bordered_rect::BorderedRectMaterial>>,
    mut alignment_materials: ResMut<Assets<super::render::AlignmentPolylineMaterial>>,
    mut alignment_vertices: ResMut<Assets<super::render::AlignmentVertices>>,

    mut infobar_writer: EventWriter<super::infobar::InfobarAlignmentEvent>,
) {
    let grid = &alignment_grid;

    let low_mem = cli_args.low_mem;

    let border_rect_mat =
        border_rect_materials.add(crate::app::render::bordered_rect::BorderedRectMaterial {
            fill_color: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
            border_color: LinearRgba::new(0.0, 0.0, 0.0, 1.0),
            border_opacities: 0xFFFFFFFF,
            border_width_px: 0.0,
            alpha_mode: AlphaMode::Blend,
        });

    let mut digit_counts = [0usize; 10];

    for ((tgt_id, qry_id), alignments) in alignments.pairs() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        let seq_pair = SequencePairTile {
            target: tgt_id,
            query: qry_id,
        };

        let tgt_len = sequences
            .get(tgt_id)
            .map(|s| s.len() as f64)
            .unwrap_or_default();
        let qry_len = sequences
            .get(qry_id)
            .map(|s| s.len() as f64)
            .unwrap_or_default();

        let x_pos = x_offset as f64 + tgt_len * 0.5;
        let y_pos = y_offset as f64 + qry_len * 0.5;
        let transform = Transform::from_translation(Vec3::new(x_pos as f32, y_pos as f32, 0.0));

        let mesh = Rectangle::from_size([tgt_len as f32, qry_len as f32].into());

        let parent = commands
            .spawn((
                meshes.add(mesh),
                border_rect_mat.clone(),
                seq_pair,
                SpatialBundle {
                    transform,
                    ..default()
                },
                Pickable {
                    should_block_lower: false,
                    is_hoverable: true,
                },
                // On::<Pointer<Over>>::run(|input: Res<ListenerInput<Pointer<Over>>>| {
                //     println!("hovering seq pair: {:?}", input.listener());
                // }),
            ))
            .with_children(|parent| {
                for (ix, alignment) in alignments.enumerate() {
                    let len = alignment.location.target_total_len as usize;
                    let pow10 = len.ilog10() as usize;
                    digit_counts[pow10.min(digit_counts.len() - 1)] += 1;

                    let al_comp = AlignmentIndex {
                        target: alignment.target_id,
                        query: alignment.query_id,
                        pair_index: ix,
                    };
                    let color_scheme = color_schemes.colors.get(&al_comp);

                    // let vertices = if cli_args.low_mem {
                    //     render::AlignmentVertices::from_alignment_ignore_cigar(alignment)
                    // } else {
                    //     render::AlignmentVertices::from_alignment(alignment)
                    // };

                    let mut al_entity = parent.spawn((
                        al_comp,
                        Pickable {
                            should_block_lower: false,
                            is_hoverable: true,
                        },
                        On::<Pointer<Out>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                        On::<Pointer<Over>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                    ));

                    if !cli_args.low_mem {
                        let material = super::render::AlignmentPolylineMaterial::from_alignment(
                            grid,
                            alignment,
                            color_scheme.clone(),
                        );
                        let vertices = super::render::AlignmentVertices::from_alignment(alignment);

                        let vx_handle = alignment_vertices.add(vertices);

                        vertex_index.vertices.insert(al_comp, vx_handle.clone());

                        al_entity.insert((alignment_materials.add(material), vx_handle));
                    }
                    // .insert(
                    //     On::<Pointer<Over>>::run(|input: Res<ListenerInput<Pointer<Over>>>, alignments: Query<&alignments::Alignment>| {
                    //         println!("hovering alignment: {:?}", input.listener());
                    //     })
                    // )

                    let al_entity = al_entity.id();
                    alignment_entity_index.insert(al_comp, al_entity);
                }
            })
            .id();

        seq_pair_entity_index.insert(seq_pair, parent);
    }

    for (exp10, count) in digit_counts.iter().enumerate() {
        println!("{exp10:10} - {count}");
    }
}
