use bevy::{
    ecs::system::SystemParam,
    math::U64Vec2,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::*;

use super::render::bordered_rect::BorderedRectMaterial;
use crate::{sequences::SeqId, Alignments};

pub mod layout;

use layout::{AabbQbvh, LayoutEntityIndex, SeqPairLayout};

use avian2d::parry::bounding_volume::Aabb;

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentEntityIndex>()
            .init_resource::<SequencePairEntityIndex>()
            .add_plugins(layout::AlignmentLayoutPlugin)
            .add_plugins(AlignmentAabbPlugin);

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
                    // prepare_alignment_vertices,
                )
                    .chain(),
            );
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

    pub default_layout: Res<'w, layout::DefaultLayout>,
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

    let sum_axis = |lens: &[(SeqId, u64)]| -> u64 { lens.iter().map(|(_, l)| l).sum() };
    let total_target = sum_axis(&targets);
    let total_query = sum_axis(&queries);

    let targets = targets.iter().map(|(i, _)| *i);
    let queries = queries.iter().map(|(i, _)| *i);

    let mut builder = layout::LayoutBuilder::from_axes(&sequences, targets, queries);
    // builder.vertical_limit = Some(total_query as f64);
    // builder.horizontal_limit = Some(total_target as f64);
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

    grid_materials: Res<GridMaterials>,
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

        let max = layout.aabbs.len();
        let mut count = 0;
        commands.entity(root).with_children(|parent| {
            for (seq_pair, aabb) in layout.aabbs.iter() {
                count += 1;

                let is_left = seq_pair.target == layout.target_edges[0];
                let is_right = seq_pair.target == layout.target_edges[1];

                let is_top = seq_pair.query == layout.query_edges[0];
                let is_bottom = seq_pair.query == layout.query_edges[1];

                let (columns, x) = match (is_left, is_right) {
                    (false, false) => (3, 1),
                    (true, false) => (3, 0),
                    (false, true) => (3, 2),
                    (true, true) => (1, 0),
                };

                let (rows, y) = match (is_top, is_bottom) {
                    (false, false) => (3, 1),
                    (true, false) => (3, 0),
                    (false, true) => (3, 2),
                    (true, true) => (1, 0),
                };

                // let columns = match (is_left, is_right) {
                //     (false, false) => 3

                // }

                // let columns = if layout.target_edges[0] == layout.target_edges[1] {
                //     if is
                //     }

                let grid_key = GridMaterialKey {
                    columns,
                    rows,
                    pos: UVec2::new(x, y),
                };

                let Some(material) = grid_materials.materials.get(&grid_key) else {
                    error!("Could not get grid material");
                    continue;
                };

                let size = aabb.extents();
                let mesh = Rectangle::from_size([size.x as f32, size.y as f32].into());

                let z = count as f32 / max as f32;

                let id = parent
                    .spawn((
                        *seq_pair,
                        SpatialBundle {
                            transform: Transform::from_xyz(0.0, 0.0, z),
                            ..SpatialBundle::INHERITED_IDENTITY
                        },
                        meshes.add(mesh),
                        material.clone(),
                        // border_rect_mat.clone(),
                        Pickable {
                            should_block_lower: false,
                            is_hoverable: true,
                        },
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
#[deprecated]
struct GridMaterial {
    material: Handle<BorderedRectMaterial>,
}

#[derive(Resource, Default)]
struct GridMaterials {
    materials: HashMap<GridMaterialKey, Handle<BorderedRectMaterial>>,
}

// the tiles in a grid layout should not all have the same material; while it's a relatively
// minor visual issue, it makes the "inner" tile edges twice the width of the outer edges.
// the material in question supports customizing the width per-side.
//
// to generate all possible (relevant) permutations, we need to consider grids of sizes up to 3
// along each axis. this struct encodes the necessary data, with the `columns` and `rows` fields
// corresponding to the size of these "small" grids, and the `pos` is a tile in the small/material grid
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GridMaterialKey {
    columns: u32,
    rows: u32,
    pos: UVec2,
}

impl GridMaterials {
    // TODO these are probably incorrect; make sure they match the `border_width_modifiers_u` in the shader
    const LEFT_MASK: u32 = 0xFF000000;
    const RIGHT_MASK: u32 = 0x00FF0000;
    const TOP_MASK: u32 = 0x0000FF00;
    const BOTTOM_MASK: u32 = 0x000000FF;

    fn initialize_materials(&mut self, assets: &mut Assets<BorderedRectMaterial>) {
        for columns in 1..=3 {
            for rows in 1..=3 {
                let keys = Self::generate_material_keys_for_grid_size(columns, rows);

                for key in keys {
                    let material = Self::material_for_key(key);
                    self.materials.insert(key, assets.add(material));
                }
            }
        }
    }

    fn material_for_key(key: GridMaterialKey) -> BorderedRectMaterial {
        let mut width_modifiers = 0u32;

        // let half = 0x7F7F7F7F;
        let half = 0x00000000;

        if key.pos.x == 0 {
            width_modifiers |= Self::LEFT_MASK;
        } else {
            width_modifiers |= Self::LEFT_MASK & half;
        }

        if key.pos.x == key.columns - 1 {
            width_modifiers |= Self::RIGHT_MASK;
        } else {
            width_modifiers |= Self::RIGHT_MASK & half;
        }

        if key.pos.y == 0 {
            width_modifiers |= Self::TOP_MASK;
        } else {
            width_modifiers |= Self::TOP_MASK & half;
        }

        if key.pos.y == key.rows - 1 {
            width_modifiers |= Self::BOTTOM_MASK;
        } else {
            width_modifiers |= Self::BOTTOM_MASK & half;
        }

        BorderedRectMaterial {
            border_width_modifiers: width_modifiers,
            ..default()
        }
    }

    fn generate_material_keys_for_grid_size(columns: u32, rows: u32) -> Vec<GridMaterialKey> {
        let mut result = Vec::new();

        for x in 0..columns {
            for y in 0..rows {
                let key = GridMaterialKey {
                    columns,
                    rows,
                    pos: UVec2::new(x, y),
                };
                result.push(key);
            }
        }

        result
    }
}

fn initialize_grid_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<BorderedRectMaterial>>,
) {
    // TODO need several materials depending on tile's relative position in the grid
    // & whether the layout has been "compressed" & tiles are overlapping
    let material = materials.add(crate::app::render::bordered_rect::BorderedRectMaterial {
        fill_color: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
        border_color: LinearRgba::new(0.0, 0.0, 0.0, 1.0),
        border_opacities: 0xFFFFFFFF,
        // border_width_px: 2.0,
        border_width_px: 20.0,
        // border_width_modifiers: 0xFFFFFFFF,
        // border_width_modifiers: 0x336699BB,
        // S/N/E/W??????
        border_width_modifiers: 0xFFAA0000,

        alpha_mode: AlphaMode::Blend,
    });

    commands.insert_resource(GridMaterial { material });

    let mut grid_mats = GridMaterials::default();
    grid_mats.initialize_materials(materials.as_mut());
    commands.insert_resource(grid_mats);
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

struct AlignmentAabbPlugin;
impl Plugin for AlignmentAabbPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentAabbs>()
            .add_systems(Startup, spawn_alignment_aabbs_task)
            .add_systems(PreUpdate, update_alignment_aabbs);
    }
}

// resource storing the local (to their parent sequence pair) AABBs of each alignment
//
#[derive(Default, Resource)]
pub struct AlignmentAabbs {
    pub qbvhs: HashMap<SequencePairTile, AabbQbvh<AlignmentIndex>>,
    task: Option<Task<Vec<AlignmentAabbEvent>>>,
}
struct AlignmentAabbEvent {
    tile: SequencePairTile,
    qbvh: AabbQbvh<AlignmentIndex>,
}

fn spawn_alignment_aabbs_task(alignments: Res<Alignments>, mut aabbs: ResMut<AlignmentAabbs>) {
    let al_ixs = alignments.indices.clone();
    let alignments = alignments.alignments.clone();

    let task_pool = AsyncComputeTaskPool::get();

    let task = task_pool.spawn(async move {
        let mut result = Vec::new();

        for (&(target, query), ixs) in al_ixs.iter() {
            let aligns = ixs.iter().filter_map(|&ix| Some((ix, alignments.get(ix)?)));

            let leaf_data = aligns.map(|(pair_index, alignment)| {
                let loc = &alignment.location;
                let mins: U64Vec2 = [loc.target_range.start, loc.query_range.start].into();
                let maxs: U64Vec2 = [loc.target_range.end, loc.query_range.end].into();

                let p0 = mins.as_dvec2();
                let p1 = maxs.as_dvec2();

                let center = (p0 + p1) * 0.5;
                let half_extents = (p1 - p0) * 0.5;

                let aabb = Aabb::from_half_extents(
                    center.to_array().into(),
                    half_extents.to_array().into(),
                );

                let index = AlignmentIndex {
                    target,
                    query,
                    pair_index,
                };
                (index, aabb)
            });

            let qbvh = AabbQbvh::from_aabbs(leaf_data);
            result.push(AlignmentAabbEvent {
                tile: SequencePairTile { target, query },
                qbvh,
            });
        }

        result
    });

    aabbs.task = Some(task);
}

fn update_alignment_aabbs(
    mut aabbs: ResMut<AlignmentAabbs>,
    // mut events: Events<AlignmentAabbEvent>,
) {
    if let Some(task) = aabbs.task.take() {
        if !task.is_finished() {
            aabbs.task = Some(task);
            return;
        }

        let Some(results) = bevy::tasks::block_on(bevy::tasks::poll_once(task)) else {
            return;
        };

        for event in results {
            aabbs.qbvhs.insert(event.tile, event.qbvh);
        }
    }
}
