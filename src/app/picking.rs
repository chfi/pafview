use bevy::prelude::*;
use bevy_mod_picking::prelude::*;
use picking_core::PickSet;

use super::render::AlignmentDisplayImage;

pub struct PickingPlugin;

impl Plugin for PickingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(DefaultPickingPlugins)
            .add_systems(PreUpdate, alignment_picking_grid.in_set(PickSet::Backend));
        //
    }
}

// this is pretty hacky and makes a number of assumptions;
// it'd be good to have an implementation that works with any number
// of views (e.g. picture-in-picture using multiple display images)
pub fn alignment_picking_grid(
    pointers: Query<(&PointerId, &PointerLocation)>,

    // TODO positions should be taken from the cache associated with the
    // AlignmentDisplayImage, but that's not quite ready yet
    alignments: Res<crate::Alignments>,
    alignment_grid: Res<crate::AlignmentGrid>,

    seq_pair_entity_index: Res<super::alignments::SequencePairEntityIndex>,
    alignment_entity_index: Res<super::alignments::AlignmentEntityIndex>,

    // this should be all `super::render::AlignmentDisplayImage`s that
    // need user interaction w/ alignments... so probably just the
    // main display image, but maybe some type of splitscreen view later
    targets: Query<
        &AlignmentDisplayImage,
        //
        With<super::render::MainAlignmentView>,
    >,

    cameras: Query<(Entity, &Camera), With<super::AlignmentCamera>>,
    windows: Query<&Window>,

    mut output: EventWriter<backend::PointerHits>,
) {
    // let mut seq_pair_hits =
    let (camera_ent, _camera) = cameras.single();

    for display in targets.iter() {
        let Some(view) = display.next_view else {
            continue;
        };

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
            // println!("cursor pos: {cursor:?}");

            // map cursor to world coordinates
            let world_pos = view.map_screen_to_world(size, cursor.to_array());

            // check collision w/ AABB in alignment grid
            // -> get seq pair hit
            let tile = alignment_grid.tile_at_world_point(world_pos);

            let Some(seq_pair @ (tgt_id, qry_id)) = tile else {
                continue;
            };

            let Some(seq_pair_ent) = seq_pair_entity_index.get(&super::SequencePairTile {
                target: tgt_id,
                query: qry_id,
            }) else {
                continue;
            };
            let hit_data = backend::HitData::new(
                camera_ent,
                10.0,
                Some(Vec3::new(world_pos.x as f32, world_pos.y as f32, 10.0)),
                None,
            );

            // output seq pair hit
            let hits = vec![(*seq_pair_ent, hit_data)];
            let ptr_hits = backend::PointerHits::new(*ptr_id, hits, 10.0);
            // println!("sending seq pair pointer hits: {ptr_hits:?}");
            output.send(ptr_hits);

            // if there's a seq pair hit, search alignments associated
            // w/ that seq pair for alignment hits
            let Some(pair_aligns) = alignments.pairs.get(&seq_pair) else {
                continue;
            };

            let seq_x = alignment_grid
                .x_axis
                .global_to_axis_exact(world_pos.x as u64);
            let seq_y = alignment_grid
                .y_axis
                .global_to_axis_exact(world_pos.y as u64);

            let Some(((_tgt_id, seq_x), (_qry_id, seq_y))) = seq_x.zip(seq_y) else {
                continue;
            };

            // output alignment hits
            let hits = pair_aligns
                .iter()
                .enumerate()
                .filter_map(|(ix, al)| {
                    let loc = &al.location;
                    if seq_x >= loc.target_range.start
                        && seq_x < loc.target_range.end
                        && seq_y >= loc.query_range.start
                        && seq_y < loc.query_range.end
                    {
                        let al_ix = super::alignments::Alignment {
                            query: qry_id,
                            target: tgt_id,
                            pair_index: ix,
                        };
                        let al_ent = alignment_entity_index.get(&al_ix)?;

                        let hit_data = backend::HitData::new(
                            camera_ent,
                            5.0,
                            Some(Vec3::new(world_pos.x as f32, world_pos.y as f32, 10.0)),
                            None,
                        );
                        Some((*al_ent, hit_data))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if !hits.is_empty() {
                let ptr_hits = backend::PointerHits::new(*ptr_id, hits, 5.0);
                // println!("sending alignment pointer hits: {ptr_hits:?}");
                output.send(ptr_hits);
            }
        }
    }
}
