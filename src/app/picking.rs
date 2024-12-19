use bevy::prelude::*;
use bevy_mod_picking::prelude::*;
use picking_core::PickSet;

use super::{
    alignments::{
        layout::{LayoutEntityIndex, SeqPairLayout},
        SequencePairAlignmentEntities,
    },
    view::AlignmentViewport,
    SequencePairTile,
};

pub struct PickingPlugin;

impl Plugin for PickingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPickingPlugins).add_systems(
            PreUpdate,
            seq_pair_and_alignment_picking.in_set(PickSet::Backend),
        );
        // .add_systems(PreUpdate, seq_pair_tile_picking.in_set(PickSet::Backend));
        // .add_systems(PreUpdate, alignment_picking_grid.in_set(PickSet::Backend));
        //
    }
}

fn seq_pair_and_alignment_picking(
    pointers: Query<(&PointerId, &PointerLocation)>,
    cameras: Query<(Entity, &Camera, &Projection), With<super::AlignmentCamera>>,
    windows: Query<&Window>,

    alignment_viewport: Res<AlignmentViewport>,
    alignments: Res<crate::Alignments>,

    layouts: Res<Assets<SeqPairLayout>>,
    layout_roots: Query<(&Transform, &Handle<SeqPairLayout>, &LayoutEntityIndex)>,
    seq_pair_tiles: Query<
        (&SequencePairAlignmentEntities, &GlobalTransform),
        With<SequencePairTile>,
    >,

    mut output: EventWriter<backend::PointerHits>,
) {
    let (camera_ent, _camera, proj) = cameras.single();
    let view = alignment_viewport.view;

    let Projection::Orthographic(proj) = proj else {
        panic!("Main camera did not have orthographic projection, this should never happen");
    };

    for (root_transform, layout_handle, entity_index) in layout_roots.iter() {
        let Some(layout) = layouts.get(layout_handle) else {
            continue;
        };

        let root_offset = root_transform.translation.as_dvec3();
        let root_scale = root_transform.scale.as_dvec3();
        // let root_rotation = root_transform.rotation;

        for (ptr_id, ptr_loc) in pointers.iter() {
            let Some(loc) = ptr_loc.location() else {
                continue;
            };

            let window = match loc.target {
                bevy::render::camera::NormalizedRenderTarget::Window(ent) => {
                    windows.get(ent.entity()).unwrap()
                }
                // bevy::render::camera::NormalizedRenderTarget::Image(_) => todo!(),
                // bevy::render::camera::NormalizedRenderTarget::TextureView(_) => todo!(),
                _ => continue,
            };

            let size = window.resolution.size();

            let cursor = loc.position;

            // map pointer screen location to world
            let world_pos = view.map_screen_to_world(size, cursor.to_array());

            // find tile under pointer
            let hit_tiles = layout.layout_qbvh.aabbs_at_point(world_pos);

            for hit_tile in hit_tiles {
                let Some(seq_entity) = entity_index.get(&hit_tile) else {
                    continue;
                };

                let Ok((alignment_entities, tile_transform)) = seq_pair_tiles.get(*seq_entity)
                else {
                    continue;
                };

                let world_z = tile_transform.translation().z;
                let depth = -proj.near - world_z;

                let hit_data = backend::HitData::new(
                    camera_ent,
                    depth,
                    Some(Vec3::new(world_pos.x as f32, world_pos.y as f32, world_z)),
                    None,
                );
                output.send(backend::PointerHits::new(
                    *ptr_id,
                    vec![(*seq_entity, hit_data)],
                    1.0,
                ));

                let Some(tile_alignments) =
                    alignments.pair_alignments((hit_tile.target, hit_tile.query))
                else {
                    continue;
                };

                let tile_offset = layout.aabbs.get(&hit_tile).map(|aabb| {
                    let p = aabb.mins;
                    bevy::math::DVec2::new(p.x, p.y)
                });

                let Some(tile_offset) = tile_offset else {
                    continue;
                };
                let world_pos = bevy::math::DVec2::new(world_pos.x, world_pos.y);

                let pointer_tile = world_pos - tile_offset;
                let p_dvec = pointer_tile;
                let px = p_dvec.x as u64;
                let py = p_dvec.y as u64;

                let mut al_hits = Vec::new();

                for (pair_index, alignment) in tile_alignments.enumerate() {
                    let loc = &alignment.location;

                    if px < loc.target_range.start
                        || px >= loc.target_range.end
                        || py < loc.query_range.start
                        || py >= loc.query_range.end
                    {
                        continue;
                    }

                    // NB: this is kind of hacky, but probably fine for now
                    // (Alignments don't actually have a transform at this point)
                    let alignment_depth = depth - 1.0;
                    let alignment_z = world_z + 1.0;

                    // TODO actually iterate part of the cigar to find the exact
                    // position; as it is this will "hit" the entire alignment AABB
                    let hit_data = backend::HitData::new(
                        camera_ent,
                        alignment_depth,
                        Some(Vec3::new(
                            world_pos.x as f32,
                            world_pos.y as f32,
                            alignment_z,
                        )),
                        None,
                    );
                    let al_entity = alignment_entities[pair_index];
                    al_hits.push((al_entity, hit_data));
                }
                // println!("sending {} alignment pointer hits", al_hits.len());
                output.send(backend::PointerHits::new(*ptr_id, al_hits, 2.0));
            }
        }
    }
}

// fn alignment_in_tile_picking(
//     tile: In()
// )
