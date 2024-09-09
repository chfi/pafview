use bevy::{prelude::*, tasks::AsyncComputeTaskPool, utils::HashMap};

use crate::app::SequencePairTile;

use super::{AlignmentRenderTarget, AlignmentViewer};

pub struct CigarSamplingRenderPlugin;

impl Plugin for CigarSamplingRenderPlugin {
    fn build(&self, app: &mut App) {
        todo!()
    }
}

/*

also using the `AlignmentViewer` component (maybe) as the final image target

but before then, plenty of buffers and scheduling needed...

- a viewer wants a new frame (viewport & canvas size given)
    - has access to position cache
- view is used to compute visible tiles & LOD level
-


probably best to just have one image per tile? maybe? or at least end up that way
- i.e. assign an image to each tile that is being rendered for the given frame


*/

#[derive(Component)]
struct RenderTileTarget {
    //
}

fn setup_render_tile_targets(
    mut commands: Commands,
    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    // color_schemes: Res<AlignmentColorSchemes>,
    cli_args: Res<crate::cli::Cli>,
) {
    for (pair @ &(tgt_id, qry_id), alignments) in alignments.pairs.iter() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        let transform =
            Transform::from_translation(Vec3::new(x_offset as f32, y_offset as f32, 0.0));

        let seq_pair = SequencePairTile {
            target: tgt_id,
            query: qry_id,
        };
        //
        let parent = commands
            .spawn((
                //
                RenderTileTarget {},
                seq_pair,
            ))
            .with_children(|parent| {
                //
            });
    }
}

/*
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileChunk {
    // offset in parent `TileBuffer`'s `data` Vec
    offset: u32,

    // size of chunk in pixels
    dims: UVec2,
}

struct TileBuffer {
    // 8-bit srgba
    data: Vec<[u8; 4]>,

    chunk_map: HashMap<SequencePairTile, TileChunk>,

    // total size of buffer and finalized render image
    canvas_dims: UVec2,
}

impl TileBuffer {
    fn new(size: UVec2) -> Self {
        let len = (size.x * size.y) as usize;
        let data = vec![[0, 0, 0, 0]; len];

        Self {
            data,
            canvas_dims: size,
            chunk_map: HashMap::default(),
        }
    }

    // fn clear(&mut self) {
    //     self.chunk_map.clear();
    // }

    fn allocate_tiles(
        //
        &mut self,

        tiles: impl IntoIterator<Item = (SequencePairTile, UVec2)>,
    ) {

        //
    }
}

fn tiles_for_viewport(
    view: crate::view::View,
    canvas_size: UVec2
) -> Vec<(SequencePairTile, [UVec2; 2])> {

    todo!();
}


#[derive(Component)]
pub struct CigarSamplingViewer;

fn start_render_task(
    //
    mut commands: Commands,

    viewers: Query<&AlignmentViewer, With<CigarSamplingViewer>>,
) {
    let task_pool = AsyncComputeTaskPool::get();

    // for (viewer, target) in viewers.iter() {
    for viewer in viewers.iter() {

        //



    }
}
*/
