use std::sync::{atomic::AtomicBool, Arc};

use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::*;

use super::{render::bordered_rect::BorderedRectMaterial, AlignmentColorSchemes};
use crate::{sequences::SeqId, Alignments, PafViewerApp};

pub mod layout;

use layout::{AabbQbvh, LayoutEntityIndex, SeqPairLayout};

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentEntityIndex>()
            .init_resource::<SequencePairEntityIndex>()
            .add_plugins(layout::AlignmentLayoutPlugin);

        // app.add_systems(Startup, initialize_default_layout);
        app.add_systems(Startup, initialize_grid_material);
        app.add_systems(
            Startup,
            (initialize_default_layout, spawn_default_layout_root).chain(),
        );

        app.add_systems(PostUpdate, update_grid_material_from_config)
            .add_systems(
                PreUpdate,
                (
                    spawn_alignments_in_tiles,
                    spawn_layout_children,
                    update_layout_tile_positions,
                    prepare_alignment_vertices,
                )
                    .chain(),
            )
            .add_systems(
                PreUpdate,
                (
                    insert_alignment_polyline_materials.after(spawn_layout_children),
                    update_alignment_polyline_materials,
                )
                    .chain(),
            );

        // app.add_systems(
        //     Startup,
        //     prepare_alignments.after(super::setup_screenspace_camera),
        // )
        // .add_systems(PreUpdate, update_seq_pair_transforms);
        //
    }
}

/// `SystemParam` for easy access to laid out sequence tiles and alignments
#[derive(SystemParam)]
pub struct AlignmentLayoutQuery<'w, 's> {
    pub layout_assets: Res<'w, Assets<SeqPairLayout>>,
    pub layout_roots: Query<
        'w,
        's,
        (
            Entity,
            &'static Transform,
            &'static Handle<SeqPairLayout>,
            &'static LayoutEntityIndex,
        ),
    >,
}

impl<'w, 's> AlignmentLayoutQuery<'w, 's> {
    /// returns the tile and local offset found at `world_point`.
    /// Includes the root layout entity. If multiple layouts or tiles overlap
    /// the world point, any of the tiles may be returned
    pub fn tile_and_local_offset_at_point(
        &self,
        world_point: impl Into<[f64; 2]>,
    ) -> Option<(Entity, SequencePairTile, [u64; 2])> {
        let world_point = bevy::math::DVec2::from(world_point.into());
        let mut hit: Option<(Entity, SequencePairTile, [u64; 2])> = None;

        for (root, root_transform, layout_handle, _) in self.layout_roots.iter() {
            // TODO: take layout transform into account
            let Some(layout) = self.layout_assets.get(layout_handle) else {
                continue;
            };

            let mut hit_tile: Option<SequencePairTile> = None;
            layout
                .layout_qbvh
                .aabbs_at_point_callback(world_point, |tile| {
                    hit_tile = Some(tile);
                    false
                });

            let Some(tile) = hit_tile else {
                // let tiles = layout.layout_qbvh.tiles_at_point(world_point);
                // let Some(tile) = tiles.first() else {
                continue;
            };

            let Some(aabb) = layout.aabbs.get(&tile) else {
                continue;
            };

            // TODO: transform here too
            let mins = bevy::math::DVec2::new(aabb.mins.x, aabb.mins.y);

            let local = (world_point - mins).as_u64vec2();
            hit = Some((root, tile, local.into()));
            break;
        }

        hit
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

// NB: same order/indices as a seq. pair tile's alignments in `paf::Alignments`
#[derive(Debug, Default, Component, Deref, DerefMut)]
pub struct SequencePairAlignmentEntities(pub Vec<Entity>);

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct AlignmentIndex {
    pub query: SeqId,
    pub target: SeqId,

    pub pair_index: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Resource, Reflect)]
pub struct DefaultLayoutRoot(pub Entity);

// create the initial sequence pair layout from the application input data,
// and create the DefaultLayout resource
pub(super) fn initialize_default_layout(
    mut commands: Commands,

    // cli_args: Res<crate::cli::Cli>,
    mut layouts: ResMut<Assets<SeqPairLayout>>,
    alignments: Res<crate::Alignments>,
    sequences: Res<crate::Sequences>,
) {
    let (mut targets, mut queries): (Vec<_>, Vec<_>) = alignments
        .alignments
        .iter()
        .map(|al| {
            let tgt = (al.target_id, al.location.target_total_len);
            let qry = (al.query_id, al.location.query_total_len);
            (tgt, qry)
        })
        .unzip();

    targets.sort_by_key(|(_, l)| (std::cmp::Reverse(*l)));
    targets.dedup();
    queries.sort_by_key(|(_, l)| (std::cmp::Reverse(*l)));
    queries.dedup();

    let targets = targets.iter().map(|(i, _)| *i);
    let queries = queries.iter().map(|(i, _)| *i);

    let builder = layout::LayoutBuilder::from_axes(targets, queries);
    // LayoutBuilder::from_axes(targets, queries).with_vertical_offset(Some(10_000_000.0));

    let layout = builder.clone().build(&sequences);
    let default_layout = layout::DefaultLayout::new(layouts.add(layout), builder);

    println!("inserting default layout resource");
    commands.insert_resource(default_layout);
}

pub(super) fn spawn_default_layout_root(
    mut commands: Commands,
    default_layout: Res<layout::DefaultLayout>,
    mut layout_events: EventWriter<layout::LayoutChangedEvent>,
) {
    let entity = commands
        .spawn((
            SpatialBundle::default(),
            default_layout.layout.clone(),
            layout::LayoutEntityIndex::default(),
        ))
        .id();
    layout_events.send(layout::LayoutChangedEvent {
        entity,
        need_respawn: true,
    });
    commands.insert_resource(DefaultLayoutRoot(entity));
}

// When a layout for a "layout root" (any entity with a Handle<SeqPairLayout>,
// for now) has been updated, this system spawns a tile for each sequence pair
// in the layout, as children of the root
fn spawn_layout_children(
    mut commands: Commands,

    layouts: Res<Assets<SeqPairLayout>>,
    mut layout_events: EventReader<layout::LayoutChangedEvent>,

    mut layout_roots: Query<(
        Entity,
        &Handle<SeqPairLayout>,
        &mut layout::LayoutEntityIndex,
    )>,

    mut meshes: ResMut<Assets<Mesh>>,

    grid_mat: Res<GridMaterial>,
) {
    let border_rect_mat = &grid_mat.material;

    for layout_event in layout_events.read() {
        if !layout_event.need_respawn {
            continue;
        }

        let Ok((root, layout_handle, mut entity_index)) = layout_roots.get_mut(layout_event.entity)
        else {
            continue;
        };

        let Some(layout) = layouts.get(layout_handle) else {
            continue;
        };

        commands.entity(root).despawn_descendants();
        entity_index.clear();

        let mut count = 0;
        commands.entity(root).with_children(|parent| {
            for (seq_pair, aabb) in layout.aabbs.iter() {
                count += 1;

                let size = aabb.extents();
                let mesh = Rectangle::from_size([size.x as f32, size.y as f32].into());

                let id = parent
                    .spawn((
                        *seq_pair,
                        SpatialBundle::INHERITED_IDENTITY,
                        meshes.add(mesh),
                        border_rect_mat.clone(),
                        Pickable {
                            should_block_lower: false,
                            is_hoverable: true,
                        },
                        // On::<Pointer<Over>>::run(|input: Res<ListenerInput<Pointer<Over>>>| {
                        //     println!("hovering seq pair: {:?}", input.listener());
                        // }),
                    ))
                    .id();

                entity_index.insert(*seq_pair, id);
            }
        });

        println!("spawned {count} tiles");
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
            .enumerate()
            .filter_map(|(local_ix, &data_ix)| {
                Some((local_ix, alignments.alignments.get(data_ix)?))
            });

        let mut count = 0;
        let mut children = Vec::new();
        commands
            .entity(tile_ent)
            .with_children(|parent| {
                for (pair_index, alignment) in tile_als {
                    count += 1;
                    let al_id =
                        parent
                            .spawn((
                                AlignmentIndex {
                                    target: alignment.target_id,
                                    query: alignment.query_id,
                                    pair_index,
                                },
                                Pickable {
                                    should_block_lower: false,
                                    is_hoverable: true,
                                },
                                On::<Pointer<Out>>::send_event::<
                                    super::infobar::InfobarAlignmentEvent,
                                >(),
                                On::<Pointer<Over>>::send_event::<
                                    super::infobar::InfobarAlignmentEvent,
                                >(),
                            ))
                            .id();

                    children.push(al_id);
                }
            })
            .insert(SequencePairAlignmentEntities(children));
        println!("spawned {count} alignment entities");
    }
}

pub(super) fn prepare_alignment_vertices(
    mut commands: Commands,

    cli_args: Res<crate::cli::Cli>,

    alignments: Res<crate::Alignments>,
    alignment_query: Query<
        (Entity, &AlignmentIndex),
        Without<Handle<super::render::gpu_lines::AlignmentVertices>>,
        // (Without<Handle<super::render::AlignmentVertices>>,),
    >,

    mut alignment_vertices_map: ResMut<super::render::gpu_lines::AlignmentVerticesIndex>,
    mut alignment_vertices: ResMut<Assets<super::render::gpu_lines::AlignmentVertices>>,

    // might want to do multiple alignments per task
    mut tasks: Local<HashMap<AlignmentIndex, Task<super::render::gpu_lines::AlignmentVertices>>>,
    // mut tasks: Local<HashMap<AlignmentIndex, Task<Vec<(Vec2, Vec2, crate::CigarOp)>>>>,
    // processing: Local<HashMap<AlignmentIndex, Arc<AtomicBool>>>,
) {
    if cli_args.low_mem {
        return;
    }

    let task_pool = AsyncComputeTaskPool::get();

    for (al_ent, al_ix) in alignment_query.iter() {
        // dbg!();
        if alignment_vertices_map.vertices.contains_key(al_ix) || tasks.contains_key(al_ix) {
            continue;
        }

        // dbg!();
        let Some((location, cigar)) = alignments
            .get(*al_ix)
            .map(|al| (al.location.clone(), al.cigar.clone()))
        else {
            continue;
        };

        let task = task_pool.spawn(async move {
            super::render::gpu_lines::AlignmentVertices::from_location_and_cigar(&location, &cigar)
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
    vertex_index: Res<super::render::gpu_lines::AlignmentVerticesIndex>,
    mut alignment_materials: ResMut<Assets<super::render::gpu_lines::AlignmentPolylineMaterial>>,
    color_schemes: Res<AlignmentColorSchemes>,

    cli_args: Res<crate::cli::Cli>,

    // layout_roots: Query<(Entity, &Handle<SeqPairLayout>, &Children)>,
    // seq_pair_tiles: Query<(&SequencePairTile, &Children)>,
    alignment_query: Query<
        (Entity, &AlignmentIndex),
        (
            Without<Handle<super::render::gpu_lines::AlignmentPolylineMaterial>>,
            Without<Handle<super::render::gpu_lines::AlignmentVertices>>,
        ),
    >,
) {
    if cli_args.low_mem {
        return;
    }

    for (entity, al_ix) in alignment_query.iter() {
        let Some(vertices) = vertex_index.vertices.get(al_ix) else {
            continue;
        };

        // create the polyline material; place at origin since `update_polyline_materials` should run after
        let color_scheme = color_schemes.get(al_ix);
        let material = super::render::gpu_lines::AlignmentPolylineMaterial::from_offset_and_colors(
            [0.0, 0.0],
            color_scheme.clone(),
        );

        commands
            .entity(entity)
            .insert((vertices.clone(), alignment_materials.add(material)));
    }
}

pub(super) fn update_alignment_polyline_materials(
    layouts: Res<Assets<SeqPairLayout>>,
    mut alignment_materials: ResMut<Assets<super::render::gpu_lines::AlignmentPolylineMaterial>>,

    layout_roots: Query<(Entity, &Handle<SeqPairLayout>)>,
    seq_pair_tiles: Query<(&SequencePairTile, &Children)>,
    alignments_query: Query<(
        Entity,
        &Handle<super::render::gpu_lines::AlignmentPolylineMaterial>,
    )>,
) {
    for (root_ent, layout_handle) in layout_roots.iter() {
        let Some(layout) = layouts.get(layout_handle) else {
            continue;
        };

        for (seq_pair, children) in seq_pair_tiles.iter() {
            let Some(aabb) = layout.aabbs.get(seq_pair) else {
                continue;
            };

            let pos = aabb.center() - aabb.half_extents();
            let new_model = Transform::from_xyz(pos.x as f32, pos.y as f32, 0.0).compute_matrix();

            for (_ent, mat_handle) in alignments_query.iter_many(children) {
                if let Some(mat) = alignment_materials.get_mut(mat_handle) {
                    mat.model = new_model;
                }
            }
        }
    }
}

// updates `Transform` component of children of layout roots
pub(super) fn update_layout_tile_positions(
    layouts: Res<Assets<SeqPairLayout>>,

    mut layout_events: EventReader<layout::LayoutChangedEvent>,
    layout_roots: Query<(&Handle<SeqPairLayout>, &Children)>,

    mut seq_pair_tiles: Query<(&SequencePairTile, &mut Transform)>,
) {
    let layout_entities = layout_events.read().map(|ev| ev.entity);
    for (layout_handle, children) in layout_roots.iter_many(layout_entities) {
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

#[derive(Resource)]
struct GridMaterial {
    material: Handle<BorderedRectMaterial>,
}

fn initialize_grid_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<BorderedRectMaterial>>,
) {
    let material = materials.add(crate::app::render::bordered_rect::BorderedRectMaterial {
        fill_color: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
        border_color: LinearRgba::new(0.0, 0.0, 0.0, 1.0),
        border_opacities: 0xFFFFFFFF,
        border_width_px: 2.0,
        alpha_mode: AlphaMode::Blend,
    });

    commands.insert_resource(GridMaterial { material });
}

fn update_grid_material_from_config(
    config: Res<crate::AppConfig>,

    mut materials: ResMut<Assets<BorderedRectMaterial>>,
    grid_mat: Res<GridMaterial>,
) {
    if !config.is_changed() {
        return;
    }

    let Some(mat) = materials.get_mut(&grid_mat.material) else {
        return;
    };

    mat.border_width_px = config.grid_line_width;
}

// resource storing the local (to their parent sequence pair) AABBs of each alignment
//
#[derive(Default, Resource)]
pub struct AlignmentAabbs {
    pub qbvhs: HashMap<SequencePairTile, AabbQbvh<AlignmentIndex>>,
}

#[derive(Event)]
struct AlignmentAabbEvent {
    tile: SequencePairTile,
    qbvh: AabbQbvh<AlignmentIndex>,
}

fn spawn_alignment_aabbs_task(alignments: Res<Alignments>, aabbs: Res<AlignmentAabbs>) {
    // how do i know which sequence pairs to process at this point
    //

    todo!();
}

fn update_alignment_aabbs(
    mut aabbs: ResMut<AlignmentAabbs>,
    mut events: Events<AlignmentAabbEvent>,
) {
    for event in events.drain() {
        aabbs.qbvhs.insert(event.tile, event.qbvh);
    }
}
