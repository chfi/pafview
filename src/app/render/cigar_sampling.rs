use bevy::{
    prelude::*,
    render::view::RenderLayers,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::Pickable;

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

#[derive(Component, Default)]
struct RenderTileTarget {
    last_rendered: Option<RenderParams>,
}

#[derive(Debug, Clone, Reflect)]
struct RenderParams {
    query_seq_bounds: std::ops::Range<u64>,
    target_seq_bounds: std::ops::Range<u64>,

    canvas_size: UVec2,
}

fn setup_render_tile_targets(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,

    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    // color_schemes: Res<AlignmentColorSchemes>,
    cli_args: Res<crate::cli::Cli>,
) {
    for (pair @ &(tgt_id, qry_id), alignments) in alignments.pairs.iter() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        // let transform =
        //     Transform::from_translation(Vec3::new(x_offset as f32, y_offset as f32, 0.0));

        let seq_pair = SequencePairTile {
            target: tgt_id,
            query: qry_id,
        };

        let size = wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        };

        let mut image = Image {
            texture_descriptor: wgpu::TextureDescriptor {
                label: None,
                size,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                mip_level_count: 1,
                sample_count: 1,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
            ..default()
        };

        image.resize(size);

        let img_handle = images.add(image);

        //
        let parent = commands.spawn((
            RenderTileTarget::default(),
            seq_pair,
            img_handle,
            RenderLayers::layer(1),
            SpatialBundle::default(),
            Pickable {
                should_block_lower: false,
                is_hoverable: true,
            },
        ));
        // .with_children(|parent| {
        //     //
        // });
    }
}

fn update_render_tile_transforms(
    grid: Res<crate::AlignmentGrid>,
    viewport: Res<crate::app::view::AlignmentViewport>,
    windows: Query<&Window>,

    mut tiles: Query<(&SequencePairTile, &mut Visibility, &mut Transform), With<RenderTileTarget>>,
) {
    let window = windows.single();

    for (pair, mut visibility, mut transform) in tiles.iter_mut() {
        // get tile position in world
        // set visibility depending on whether tile overlaps view
        // update transforms (offsets)
        todo!();
    }
}

#[derive(Component)]
struct RenderTask {
    task: Task<Vec<u8>>,

    params: RenderParams,
}

fn spawn_render_tasks(
    //
    mut commands: Commands,

    alignments: Res<crate::Alignments>,

    viewport: Res<crate::app::view::AlignmentViewport>,
    windows: Query<&Window>,
    tiles: Query<
        (
            Entity,
            &SequencePairTile,
            &Transform,
            &ViewVisibility,
            &RenderTileTarget,
        ),
        Without<RenderTask>,
    >,
) {
    let window = windows.single();
    let win_size = window.physical_size();
    let size = Vec2::new(win_size.x as f32, win_size.y as f32);

    let task_pool = AsyncComputeTaskPool::get();

    for (tile_ent, pair, transform, vis, render_tile) in tiles.iter() {
        if !vis.get() {
            continue;
        }

        // compute visible area of seq. pair tile in the view
        let query_seq_bounds = todo!();
        let target_seq_bounds = todo!();

        // compute size in pixels of tile on screen
        let canvas_size = todo!();

        let params = RenderParams {
            query_seq_bounds,
            target_seq_bounds,
            canvas_size,
        };

        let alignments = alignments
            .pairs
            .get(&(pair.target, pair.query))
            .unwrap()
            .clone();

        let task = task_pool.spawn(async move {
            let t0 = std::time::Instant::now();

            let len = (canvas_size.x + canvas_size.y) as usize;
            // let mut pixels = vec![[0u8; 4]; len];
            let mut buffer = vec![0u8; len * 4];

            let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut buffer);

            for (ix, alignment) in alignments.iter().enumerate() {
                rasterize_alignment(
                    //
                    pixels,
                    canvas_size,
                    &query_seq_bounds,
                    &target_seq_bounds,
                    alignment,
                );
            }

            let time = t0.elapsed().as_secs_f64();
            println!(
                "rendered tile w/ {} alignments in {time} s",
                alignments.len()
            );

            buffer
        });

        commands
            .entity(tile_ent)
            .insert(RenderTask { task, params });
    }
    //

    //

    //
}

fn finish_render_tasks(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,

    mut tiles: Query<(
        Entity,
        // &SequencePairTile,
        // &Transform,
        // &ViewVisibility,
        &Handle<Image>,
        &mut RenderTileTarget,
        &mut RenderTask,
    )>,
) {
    for (tile, image, mut render, mut task) in tiles.iter_mut() {
        let Some(pixels) = bevy::tasks::block_on(bevy::tasks::poll_once(&mut task.task)) else {
            continue;
        };

        let Some(image) = images.get_mut(image) else {
            panic!("couldn't modify render tile image");
        };

        image.resize(wgpu::Extent3d {
            width: task.params.canvas_size.x,
            height: task.params.canvas_size.y,
            depth_or_array_layers: 1,
        });

        let img_size = task.params.canvas_size;
        image.texture_descriptor.size.width = img_size.x;
        image.texture_descriptor.size.height = img_size.y;
        image.data = pixels;

        commands.entity(tile).remove::<RenderTask>();
    }
}

fn rasterize_alignment(
    pixels: &mut [[u8; 4]],
    px_dims: UVec2,
    query: &std::ops::Range<u64>,
    target: &std::ops::Range<u64>,
    alignment: &crate::Alignment,
) {
    use crate::cigar::IndexedCigar;
    use line_drawing::XiaolinWu;

    for item in alignment.cigar.iter_target_range(target.clone()) {
        // map `item`'s target & query ranges to line endpoints inside `pixels`
    }

    todo!();
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
