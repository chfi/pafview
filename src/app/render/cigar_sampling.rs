use bevy::{
    prelude::*,
    render::view::RenderLayers,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::Pickable;

use crate::app::{ForegroundColor, SequencePairTile};

use super::{AlignmentRenderTarget, AlignmentViewer};

pub struct CigarSamplingRenderPlugin;

impl Plugin for CigarSamplingRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_render_tile_targets);
        app.add_systems(
            Startup,
            setup_tile_debug_time.after(setup_render_tile_targets),
        );

        app.add_systems(Update, update_tile_debug_time);
        /*
            update_render_tile_transforms
            spawn_render_tasks
            finish_render_tasks
        */

        // app.add_systems(Update, update_r)

        app.add_systems(
            //
            PreUpdate,
            finish_render_tasks,
        )
        .add_systems(PostUpdate, update_render_tile_transforms)
        .add_systems(PostUpdate, spawn_render_tasks);
    }
}

#[derive(Default, Component)]
struct TileDebugTimeView {
    last_params: Option<RenderParams>,
}

fn setup_tile_debug_time(
    mut commands: Commands,
    fg_color: Res<ForegroundColor>,
    tiles: Query<Entity, With<RenderTileTarget>>,
) {
    for tile in tiles.iter() {
        commands.entity(tile).with_children(|parent| {
            parent
                .spawn((
                    TileDebugTimeView::default(),
                    RenderLayers::layer(1),
                    Text2dBundle {
                        text: Text::from_section(
                            "",
                            TextStyle {
                                color: fg_color.0,
                                ..default()
                            },
                        ),
                        text_anchor: bevy::sprite::Anchor::BottomLeft,
                        ..default()
                    },
                    // TextBundle::from_section("", TextStyle::default()),
                ))
                .insert(Transform::from_xyz(0., 0., 1.0));
        });
    }
}

fn update_tile_debug_time(
    // mut commands: Commands,
    tiles: Query<(&Children, &RenderTileTarget)>,
    mut dbg_text: Query<(&mut Text, &mut TileDebugTimeView)>,
) {
    for (children, tile) in tiles.iter() {
        for child in children.iter() {
            if let Ok((mut text, mut tile_dbg)) = dbg_text.get_mut(*child) {
                if tile_dbg.last_params.as_ref() == tile.last_rendered.as_ref() {
                    continue;
                }
                let value = if let Some(time) = tile.last_update.as_ref() {
                    format!("{}ms", time.elapsed().as_millis())
                } else {
                    "".to_string()
                };
                text.sections[0].value = value;
                tile_dbg.last_params = tile.last_rendered.clone();
            }
        }
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
    last_update: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Reflect, PartialEq, Eq)]
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
        // let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        // let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

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
        let parent = commands
            .spawn((
                RenderTileTarget::default(),
                seq_pair,
                RenderLayers::layer(1),
                SpatialBundle::default(),
                Pickable {
                    should_block_lower: false,
                    is_hoverable: true,
                },
            ))
            .insert(SpriteBundle {
                sprite: Sprite {
                    anchor: bevy::sprite::Anchor::BottomLeft,
                    ..default()
                },
                ..default()
            })
            .insert(img_handle.clone())
            .id();

        println!(
            "Target {:?}/Query {:?}\t=>{parent:?}",
            seq_pair.target, seq_pair.query
        );
        println!("img handle for tile {parent:?}: {:?}", img_handle);
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
    let win_size = window.size();

    for (pair, mut visibility, mut transform) in tiles.iter_mut() {
        // get tile position in world
        let target_seq_world = grid.x_axis.sequence_axis_range(pair.target);
        let query_seq_world = grid.y_axis.sequence_axis_range(pair.query);

        // set visibility depending on whether tile overlaps view
        // dbg!();
        let Some((target_seq_world, query_seq_world)) = target_seq_world.zip(query_seq_world)
        else {
            continue;
        };

        let t0 = target_seq_world.start.max(viewport.view.x_min as u64);
        let t1 = target_seq_world.end.min(viewport.view.x_max as u64);

        let q0 = query_seq_world.start.max(viewport.view.y_min as u64);
        let q1 = query_seq_world.end.min(viewport.view.y_max as u64);

        if t0 >= t1 || q0 >= q1 {
            // println!("hiding tile\ttgt [{t0}, {t1}]\tqry [{q0}, {q1}]");
            *visibility = Visibility::Hidden;
            continue;
        }
        if *visibility != Visibility::Visible {
            *visibility = Visibility::Visible;
        }

        // update transforms (offsets)
        let tile_pos = viewport
            .view
            .map_world_to_screen(win_size, [t0 as f64, q0 as f64]);
        transform.translation.x = tile_pos.x - win_size.x * 0.5;
        transform.translation.y = win_size.y - tile_pos.y - win_size.y * 0.5;
        // transform.translation.y = tile_pos.y - win_size.y * 0.5;
        // dbg!(tile_pos);
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

    // time: Res<Time>,
    // keys: Res<ButtonInput<KeyCode>>,
    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    viewport: Res<crate::app::view::AlignmentViewport>,

    windows: Query<&Window>,
    tiles: Query<
        (
            Entity,
            &SequencePairTile,
            &Transform,
            &Visibility,
            &RenderTileTarget,
        ),
        Without<RenderTask>,
    >,
) {
    let window = windows.single();
    let win_size = window.physical_size();
    let size = Vec2::new(win_size.x as f32, win_size.y as f32);

    let task_pool = AsyncComputeTaskPool::get();

    for (ix, (tile_ent, pair, transform, vis, render_tile)) in tiles.iter().enumerate() {
        if vis == Visibility::Hidden {
            // println!("not visible");
            continue;
        }

        let update_timer_lapsed = render_tile
            .last_update
            .as_ref()
            .map(|i| i.elapsed().as_millis() > 100)
            .unwrap_or(true);
        if !update_timer_lapsed {
            println!("skipping due to timer");
            continue;
        }

        // dbg!();
        // compute visible area of seq. pair tile in the view
        let target_seq_world = grid.x_axis.sequence_axis_range(pair.target);
        let query_seq_world = grid.y_axis.sequence_axis_range(pair.query);

        let Some((target_seq_world, query_seq_world)) = target_seq_world.zip(query_seq_world)
        else {
            continue;
        };

        // dbg!();
        let t0 = target_seq_world.start.max(viewport.view.x_min as u64);
        let t1 = target_seq_world.end.min(viewport.view.x_max as u64);

        let q0 = query_seq_world.start.max(viewport.view.y_min as u64);
        let q1 = query_seq_world.end.min(viewport.view.y_max as u64);

        if t0 >= t1 || q0 >= q1 {
            continue;
        }
        // dbg!();

        let target_seq_bounds = t0..t1;
        let query_seq_bounds = q0..q1;

        // dbg!((&target_seq_bounds, &query_seq_bounds));

        // compute size in pixels of tile on screen
        let width = win_size.x as f64 * (t1 - t0) as f64 / viewport.view.width();
        let height = win_size.y as f64 * (q1 - q0) as f64 / viewport.view.height();

        let canvas_size = UVec2::new(width as u32, height as u32);
        // println!("canvas_size: {canvas_size:?}");

        if canvas_size.x == 0 || canvas_size.y == 0 {
            continue;
        }
        // dbg!();

        let params = RenderParams {
            query_seq_bounds: q0..q1,
            target_seq_bounds: t0..t1,
            canvas_size,
        };

        if Some(&params) == render_tile.last_rendered.as_ref() {
            continue;
        }

        let alignments = alignments
            .pairs
            .get(&(pair.target, pair.query))
            .unwrap()
            .clone();

        // let ang = (pair.target.0 * pair.query.0) % 50;
        // let ang = (30.0 / ix as f32) * std::f32::consts::PI;
        // let color = Color::hsv(ang as f32 * 400.0, 0.8, 0.8);

        let task = task_pool.spawn(async move {
            // println!("in task for tile {tile_ent:?}; color: {color:?}");
            let t0 = std::time::Instant::now();

            let len = (canvas_size.x * canvas_size.y) as usize;
            // let mut pixels = vec![[0u8; 4]; len];
            // let mut buffer = vec![0u8; len * 4];
            let mut buffer = vec![0u8; len * 4];

            let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut buffer);
            // let rgb = color.to_srgba();
            // pixels.fill([
            //     (rgb.red * 255.0) as u8,
            //     (rgb.green * 255.0) as u8,
            //     (rgb.blue * 255.0) as u8,
            //     255,
            // ]);
            // pixels.fill([255, 0, 0, 255]);
            pixels.fill([0, 0, 0, 0]);

            for (_ix, alignment) in alignments.iter().enumerate() {
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

        // println!("updating image {image:?} for tile {tile:?}");
        let Some(image) = images.get_mut(image) else {
            panic!("couldn't modify render tile image");
        };

        image.resize(wgpu::Extent3d {
            width: task.params.canvas_size.x,
            height: task.params.canvas_size.y,
            depth_or_array_layers: 1,
        });

        // println!("image data size: {}", image.data.len());
        // println!("pixels size: {}", pixels.len());
        let img_size = task.params.canvas_size;
        image.texture_descriptor.size.width = img_size.x;
        image.texture_descriptor.size.height = img_size.y;
        image.data = pixels;

        render.last_rendered = Some(task.params.clone());
        render.last_update = Some(std::time::Instant::now());
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

    // let q0 = query.start as f64;
    // let q1 = query.end as f64;

    // let t0 = target.start as f64;
    // let t1 = target.end as f64;

    // for ((px, py), v) in XiaolinWu::<f64, i32>::new(map_pt(t0, q0), map_pt(t1, q1)) {
    //     if px >= 0 && px < px_dims.x as i32 && py >= 0 && py < px_dims.y as i32 {
    //         let py = px_dims.y as i32 - py - 1;
    //         let ix = (px + py * px_dims.x as i32) as usize;
    //         if ix < pixels.len() {
    //             let alpha = (v * 255.0) as u8;
    //             pixels[ix] = [0, 0, 0, alpha];
    //         }
    //     }
    //     //
    // }

    // let map_pt = |tgt: &std::ops::Range<u64>, qry: &std::ops::Range<u64>, x: f64, y: f64| {
    //     let t0 = tgt.start as f64;
    //     let t1 = tgt.end as f64;
    //     let q0 = qry.start as f64;
    //     let q1 = qry.end as f64;
    //     let xn = (x - t0) / (t1 - t0);
    //     let yn = (y - q0) / (q1 - q0);
    //     (xn * px_dims.x as f64, yn * px_dims.y as f64)
    // };

    // TODO rasterize entire cigar
    for item in alignment.cigar.iter_target_range(target.clone()) {
        // map `item`'s target & query ranges to line endpoints inside `pixels`
        // rather, relative to `pixels` top left corner
        // (or maybe bottom left)

        let t0 = item.target_range.start as f64;
        let t1 = item.target_range.end as f64;
        let q0 = item.query_range.start as f64;
        let q1 = item.query_range.end as f64;

        if (t0 - t1).abs() < 1.0 || (q0 - q1).abs() < 1.0 {
            continue;
        }

        let map_pt = |x: f64, y: f64| {
            let xn = (x - t0) / (t1 - t0);
            let yn = (y - q0) / (q1 - q0);
            (xn * px_dims.x as f64, yn * px_dims.y as f64)
        };

        let start = map_pt(t0, q0);
        let end = map_pt(t1, q1);

        // println!(" >> LINE ENDS:\t{start:?}\t{end:?}");

        for ((px, py), v) in XiaolinWu::<f64, i32>::new(map_pt(t0, q0), map_pt(t1, q1)) {
            if px >= 0 && px < px_dims.x as i32 && py >= 0 && py < px_dims.y as i32 {
                let py = px_dims.y as i32 - py - 1;
                let ix = (px + py * px_dims.x as i32) as usize;
                if ix < pixels.len() {
                    let alpha = (v * 255.0) as u8;
                    pixels[ix] = [0, 0, 0, alpha];
                }
            }
            //
        }
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
