use bevy::{
    math::DVec2,
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
        // app.add_systems(Startup, setup_render_tile_targets);
        // app.add_systems(
        //     Startup,
        //     setup_tile_debug_time.after(setup_render_tile_targets),
        // );

        app.add_systems(Startup, setup_render_tiles_new);
        app.add_systems(Startup, setup_tile_debug_time.after(setup_render_tiles_new));

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

fn line_rasterizer_test(mut commands: Commands, keys: Res<ButtonInput<KeyCode>>) {
    //
}

#[derive(Clone, Copy, PartialEq)]
struct RenderParams {
    view: crate::view::View,
    canvas_size: UVec2,
}

#[derive(Component, Default)]
struct RenderTile {
    tile_grid_pos: UVec2,

    size: UVec2,
    view: Option<crate::view::View>,

    last_rendered: Option<RenderParams>,
    last_update: Option<std::time::Instant>,
}

#[derive(Resource, Clone, Copy)]
struct RenderTileGrid {
    rows: usize,
    columns: usize,
}

fn setup_render_tiles_new(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,

    windows: Query<&Window>,
) {
    let window = windows.single();
    let win_size = window.physical_size();

    let tile_grid = RenderTileGrid {
        rows: 1,
        columns: 1,
    };

    commands.insert_resource(tile_grid);

    let width = win_size.x / tile_grid.rows as u32;
    let height = win_size.y / tile_grid.columns as u32;

    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    for row in 0..tile_grid.rows {
        for column in 0..tile_grid.columns {
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
            let tile = commands
                .spawn((
                    RenderTile {
                        size: win_size,
                        tile_grid_pos: UVec2::new(column as u32, row as u32),
                        ..default()
                    },
                    RenderLayers::layer(1),
                    SpatialBundle::default(),
                    Pickable {
                        should_block_lower: false,
                        is_hoverable: false,
                    },
                ))
                .insert(SpriteBundle {
                    sprite: Sprite { ..default() },
                    ..default()
                })
                .insert(img_handle.clone())
                .id();
        }
    }
}

fn rasterize_world_lines<P: Into<[f64; 2]>>(
    tile_bounds: crate::view::View,
    tile_dims: UVec2,
    tile_data: &mut [[u8; 4]],

    lines: impl IntoIterator<Item = [P; 2]>,
) {
    use line_drawing::XiaolinWu;

    for [p0, p1] in lines {
        let p0 = p0.into();
        let p1 = p1.into();

        let dims = tile_dims.as_vec2();
        let s0 = tile_bounds.map_world_to_screen(dims, p0);
        let s1 = tile_bounds.map_world_to_screen(dims, p1);

        for ((px, py), v) in XiaolinWu::<f32, i32>::new((s0.x, s0.y), (s1.x, s1.y)) {
            if px >= 0 && px < tile_dims.x as i32 && py >= 0 && py < tile_dims.y as i32 {
                let py = tile_dims.y as i32 - py - 1;
                let ix = (px + py * tile_dims.x as i32) as usize;
                if ix < tile_data.len() {
                    let alpha = (v * 255.0) as u8;
                    tile_data[ix] = [0, 0, 0, alpha];
                }
            }
        }
    }
}

fn rasterize_alignments_in_tile<'a, S, I>(
    tile_bounds: crate::view::View,
    tile_pixels: UVec2,
    // tile_mat: ultraviolet::DMat3,
    // grid: &crate::AlignmentGrid,
    seq_pairs: S,
    // alignments: impl IntoIterator<Item = (SequencePairTile, [DVec2; 2], &'a crate::Alignment)>,
) -> Vec<[u8; 4]>
where
    S: IntoIterator<Item = (SequencePairTile, [DVec2; 2], I)>,
    I: IntoIterator<Item = &'a crate::Alignment>,
{
    let px_count = (tile_pixels.x + tile_pixels.y) as usize;
    let mut pixels = vec![[0u8; 4]; px_count];

    let vx_min = tile_bounds.x_min;
    let vx_max = tile_bounds.x_max;

    for (seq_pair, [seq_min, seq_max], alignments) in seq_pairs {
        // seq_min, seq_max encode the position of the seq. pair tile in the world

        for alignment in alignments {
            let al_min = seq_min.x + alignment.location.target_range.start as f64;
            let al_max = al_min + alignment.location.target_range.end as f64;

            let al_min = al_min.clamp(vx_min, vx_max) as u64;
            let al_max = al_max.clamp(vx_min, vx_max) as u64;

            if al_min == al_max {
                println!("no overlap; skipping");
                continue;
            }

            let loc_min = al_min.checked_sub(seq_min.x as u64).unwrap_or_default();
            let loc_max = al_max.checked_sub(seq_min.x as u64).unwrap_or_default();

            if loc_min == loc_max {
                println!("no overlap after offset; skipping");
                continue;
            }
        }
        //
    }

    pixels
}

#[derive(Default, Component)]
struct TileDebugTimeView {
    last_params: Option<RenderParamsOld>,
}

fn setup_tile_debug_time(
    mut commands: Commands,
    fg_color: Res<ForegroundColor>,
    tiles: Query<Entity, With<RenderTile>>,
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
    last_rendered: Option<RenderParamsOld>,
    last_update: Option<std::time::Instant>,
}

#[derive(Debug, Clone, Reflect, PartialEq, Eq)]
struct RenderParamsOld {
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
    mut commands: Commands,

    // time: Res<Time>,
    // keys: Res<ButtonInput<KeyCode>>,
    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    viewport: Res<crate::app::view::AlignmentViewport>,
    render_grid: Res<RenderTileGrid>,

    windows: Query<&Window>,
    tiles: Query<
        (
            Entity,
            // &SequencePairTile,
            // &Transform,
            &RenderTile,
        ),
        Without<RenderTask>,
    >,
) {
    let window = windows.single();
    let win_size = window.physical_size();

    let tile_size = UVec2::new(
        win_size.x / render_grid.columns as u32,
        win_size.y / render_grid.rows as u32,
    );

    let world_tile = DVec2::new(
        win_size.x as f64 / render_grid.columns as f64,
        win_size.y as f64 / render_grid.rows as f64,
    );

    let task_pool = AsyncComputeTaskPool::get();

    for (tile_ent, tile) in tiles.iter() {
        // if vis == Visibility::Hidden {
        //     println!("not visible");
        //     continue;
        // }

        if let Some(last) = tile.last_update.as_ref() {
            let ms = last.elapsed().as_millis();
            if ms < 100 {
                // println!("skipping due to timer");
                continue;
            } else {
                println!("timer lapsed; updating");
            }
        }

        // position of tile on screen
        let tile_pos = tile.tile_grid_pos;
        let s_x0 = (tile_pos.x * tile_size.x) as f64;
        let s_y0 = (tile_pos.y * tile_size.y) as f64;

        let s_x1 = s_x0 + tile_size.x as f64;
        let s_y1 = s_y0 + tile_size.y as f64;

        // compute world viewport corresponding to tile
        let w_x0 = tile_pos.x as f64 * world_tile.x;
        let w_y0 = tile_pos.y as f64 * world_tile.y;

        let w_x1 = w_x0 + world_tile.x;
        let w_y1 = w_y0 + world_tile.y;

        let tile_bounds = crate::view::View {
            x_min: w_x0,
            x_max: w_x1,
            y_min: w_y0,
            y_max: w_y1,
        };

        // find alignments overlapping tile
        // first find overlapping seq. pairs
        let tgts = grid.x_axis.tiles_covered_by_range(w_x0..=w_x1);
        let qrys = grid.y_axis.tiles_covered_by_range(w_y0..=w_y1);
        let Some((tgts, qrys)) = tgts.zip(qrys) else {
            continue;
        };
        let qrys = qrys.collect::<Vec<_>>();

        let tile_pairs = tgts.flat_map(|tgt| qrys.iter().map(move |qry| (tgt, *qry)));

        let params = RenderParams {
            view: tile_bounds,
            canvas_size: tile_size,
        };

        let al_pairs = alignments.pairs.clone();

        // spawn task
        let task = task_pool.spawn(async move {
            // let tiles = tile_pairs.into_iter().filter_map(|pair @ (target, query)| {
            //     // position from alignment grid

            //     // use all alignments & let rasterizer skip out of bounds alignments

            //     todo!();
            // });
            /*
            let tile_alignments = tile_pairs.into_iter().filter_map(|pair @ (target, query)| {
                let als = al_pairs.get(&pair)?;

                let filtered = als.iter().flat_map(|(pair, als)| {
                    als.iter().filter(|&al| {
                        let loc = &al.location;
                        let tgt = &loc.target_range;
                        let qry = &loc.query_range;
                        tgt.start as f64 >= w_x0
                            && tgt.end as f64 <= w_x1
                            && qry.start as f64 >= w_y0
                            && qry.end as f64 <= w_y1
                    })
                });

                Some((
                    SequencePairTile { target, query },
                    // als
                ))
            });
            */
            // .flat_map(|(pair, als)| {
            //     als.iter().filter(|&al| {
            //         let loc = &al.location;
            //         let tgt = &loc.target_range;
            //         let qry = &loc.query_range;
            //         tgt.start as f64 >= w_x0
            //             && tgt.end as f64 <= w_x1
            //             && qry.start as f64 >= w_y0
            //             && qry.end as f64 <= w_y1
            //     })
            // });

            // let buffer = rasterize_alignments_in_tile(tile_bounds, tile_size, tile_alignments);

            todo!();
        });

        commands
            .entity(tile_ent)
            .insert(RenderTask { task, params });
    }
    //
}

/*
fn spawn_render_tasks_old(
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
            println!("not visible");
            continue;
        }

        if let Some(last) = render_tile.last_update.as_ref() {
            let ms = last.elapsed().as_millis();
            if ms < 100 {
                // println!("skipping due to timer");
                continue;
            } else {
                println!("timer lapsed; updating");
            }
        }

        // let update_timer_lapsed = render_tile
        //     .last_update
        //     .as_ref()
        //     .map(|i| i.elapsed().as_millis() > 100)
        //     .unwrap_or(false);
        // if !update_timer_lapsed {
        //     println!("skipping due to timer");
        //     continue;
        // }

        // dbg!();
        // compute visible area of seq. pair tile in the view
        let target_seq_world = grid.x_axis.sequence_axis_range(pair.target);
        let query_seq_world = grid.y_axis.sequence_axis_range(pair.query);

        let Some((target_seq_world, query_seq_world)) = target_seq_world.zip(query_seq_world)
        else {
            continue;
        };

        dbg!();
        let t0 = target_seq_world.start.max(viewport.view.x_min as u64);
        let t1 = target_seq_world.end.min(viewport.view.x_max as u64);

        let q0 = query_seq_world.start.max(viewport.view.y_min as u64);
        let q1 = query_seq_world.end.min(viewport.view.y_max as u64);

        if t0 >= t1 || q0 >= q1 {
            continue;
        }
        dbg!();

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
        dbg!();

        let params = RenderParamsOld {
            query_seq_bounds: q0..q1,
            target_seq_bounds: t0..t1,
            canvas_size,
        };

        // if Some(&params) == render_tile.last_rendered.as_ref() {
        //     continue;
        // }

        dbg!();

        // let alignments = alignments.pairs.clone();
        // let tile_bounds = alignments.pairs.iter().filter_map(|((tgt, qry), als)| {
        //     //
        // });
        let tile_alignments = alignments
            .pairs
            .get(&(pair.target, pair.query))
            .unwrap()
            .clone();

        // let ang = (pair.target.0 * pair.query.0) % 50;
        // let ang = (30.0 / ix as f32) * std::f32::consts::PI;
        // let color = Color::hsv(ang as f32 * 400.0, 0.8, 0.8);

        let task = task_pool.spawn(async move {
            println!("in task for tile {tile_ent:?}");
            // let t0 = std::time::Instant::now();

            let len = (canvas_size.x * canvas_size.y) as usize;
            // let mut pixels = vec![[0u8; 4]; len];
            // let mut buffer = vec![0u8; len * 4];
            let mut buffer = vec![0u8; len * 4];

            let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut buffer);

            let view = crate::view::View {
                x_min: target_seq_bounds.start as f64,
                x_max: target_seq_bounds.end as f64,
                y_min: query_seq_bounds.start as f64,
                y_max: query_seq_bounds.end as f64,
            };
            // let rgb = color.to_srgba();
            // pixels.fill([
            //     (rgb.red * 255.0) as u8,
            //     (rgb.green * 255.0) as u8,
            //     (rgb.blue * 255.0) as u8,
            //     255,
            // ]);
            pixels.fill([255, 0, 0, 255]);
            // pixels.fill([0, 0, 0, 0]);

            println!("rasterizing {} alignments", tile_alignments.len());
            for (_ix, alignment) in tile_alignments.iter().enumerate() {
                println!("{_ix}");

                // rasterize_alignment_into_view_buffer(
                //     pixels,
                //     canvas_size,
                //     view,
                //     alignment,
                //     [target_seq_world.start, query_seq_world.start],
                // );
                rasterize_alignment_old(
                    //
                    pixels,
                    canvas_size,
                    view,
                    [target_seq_world.start, query_seq_world.start],
                    &query_seq_bounds,
                    &target_seq_bounds,
                    alignment,
                );
            }
            dbg!("------rasterization complete-----------------");

            // let time = t0.elapsed().as_secs_f64();

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
*/

fn finish_render_tasks(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,

    mut tiles: Query<(
        Entity,
        // &SequencePairTile,
        // &Transform,
        // &ViewVisibility,
        &Handle<Image>,
        &mut RenderTile,
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

// fn rasterize_alignment_into_view_buffer(
//     pixels: &mut [[u8; 4]],
//     px_dims: UVec2,
//     view: crate::view::View,
//     alignment: &crate::Alignment,
//     al_seq_offsets: [u64; 2],
// ) {
//     use line_drawing::XiaolinWu;

//     let [al_tgt_offset, _al_qry_offset] = al_seq_offsets;

//     // find the intersection of the `alignment`'s target range with
//     // the `view`

//     let al_loc = &alignment.location;
//     let al_world_start = al_tgt_offset + al_loc.target_range.start;
//     let al_world_end = al_tgt_offset + al_loc.target_range.end;

//     println!("{view:?}");

//     let clamped_range = {
//         let start = (view.x_min as u64).clamp(al_world_start, al_world_end);
//         let end = (view.x_max as u64).clamp(al_world_start, al_world_end);

//         if start == end {
//             return;
//         }

//         let x0 = start - al_world_start;
//         let x1 = end - al_world_start;
//         x0..x1
//     };
//     println!("al range: {:?}", clamped_range);

//     //
// }

fn rasterize_alignment_old(
    pixels: &mut [[u8; 4]],
    px_dims: UVec2,
    view: crate::view::View,
    seq_offsets: [u64; 2],
    vis_query: &std::ops::Range<u64>,
    vis_target: &std::ops::Range<u64>,
    alignment: &crate::Alignment,
) {
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
    // let tgt_start = (alignment.location.target_range.start as i64 - seq_target.start).clamp();

    // let tgt_world_start = seq_target.start + alignment.location.target_range.start;
    // let tgt_world_end = tgt_world_start + alignment.location.aligned_target_len();

    // let tgt_start = (seq_target.start + alignment.location.target_range.start) - seq_offsets[0];
    // let tgt_end = tgt_start + (seq_target.end - seq_target.start);
    // dbg!(tgt_start..tgt_end);

    let al_tgt = &alignment.location.target_range;
    let al_tgt = (seq_offsets[0] + al_tgt.start)..(seq_offsets[0] + al_tgt.end);

    let al_start = if vis_target.start <= al_tgt.start {
        0
    } else {
        vis_target.start - al_tgt.start
    };

    let al_end = if vis_target.end >= al_tgt.end {
        al_tgt.end
    } else {
        vis_target.end - al_tgt.start
    };

    let al_range = al_start..al_end;
    /*
    dbg!(&al_tgt);
    let tgt_min = vis_target.start;
    let tgt_max = vis_target.end;

    let al_min = (al_tgt.start).clamp(tgt_min, tgt_max);
    let al_max = (al_tgt.end).clamp(tgt_min, tgt_max);

    dbg!((tgt_min, tgt_max));
    dbg!((al_min, al_max));
    let al_offset = vis_target.start + al_tgt.start;
    dbg!(al_offset);
    // let al_start = al_min - al_offset;
    // let al_end = al_max - al_offset;
    let al_start = al_min.checked_sub(al_offset);
    let al_end = al_max.checked_sub(al_offset);

    println!("{al_min} - {al_offset} = {al_start:?}");
    println!("{al_max} - {al_offset} = {al_end:?}");

    let Some(al_range) = al_start.zip(al_end).map(|(s, e)| s..e) else {
        dbg!((al_start, al_end));
        return;
    };
    */

    dbg!(&al_range);

    /*
    let at0 = al_tgt.start as i64;
    let at1 = al_tgt.end as i64;

    let ct0 = at0.clamp(it0, it1);
    let ct1 = at1.clamp(it0, it1);

    let tgt_start = ct0.checked_sub()

    // if ct0 - ct1 == 0 {
    if tgt_end - tgt_start == 0 {
        return;
    }
    */

    let viewport = {
        let x = (vis_target.start + vis_target.end) as f64 * 0.5;
        let y = (vis_query.start + vis_query.end) as f64 * 0.5;
        let vw = (vis_target.end - vis_target.start) as f64;
        let vh = (vis_query.end - vis_query.start) as f64;

        crate::view::Viewport {
            // view_center: [x, y].into(),
            // view_size: [vw, vh].into(),
            view_center: view.center(),
            view_size: view.size(),
            canvas_offset: [0.0, 0.0].into(),
            canvas_size: [px_dims.x as f32, px_dims.y as f32].into(),
        }
    };

    let mat = viewport.world_screen_dmat3();

    // TODO rasterize entire cigar
    for (ix, item) in alignment
        .cigar
        // .iter_target_range(tgt_start..tgt_end)
        // .iter_target_range(seq_target.clone())
        .iter_target_range(al_range)
        .enumerate()
    {
        // map `item`'s target & query ranges to line endpoints inside `pixels`
        // rather, relative to `pixels` top left corner
        // (or maybe bottom left)

        let t0 = item.target_range.start;
        let t1 = item.target_range.end;
        let q0 = item.query_range.start;
        let q1 = item.query_range.end;

        // let t0 = t0 + alignment.location.target_range.start as f64;
        // let t1 = t1 + alignment.location.target_range.start as f64;
        // let q0 = q0 + alignment.location.query_range.start as f64;
        // let q1 = q1 + alignment.location.query_range.start as f64;

        let t0 = (seq_offsets[0] + alignment.location.map_from_local_target_offset(t0)) as f64;
        let t1 = (seq_offsets[0] + alignment.location.map_from_local_target_offset(t1)) as f64;
        let q0 = (seq_offsets[1] + alignment.location.map_from_local_query_offset(q0)) as f64;
        let q1 = (seq_offsets[1] + alignment.location.map_from_local_query_offset(q1)) as f64;

        // let t0 = t0

        if ix == 0 {
            dbg!();
        }
        if (t0 - t1).abs() < 1.0 || (q0 - q1).abs() < 1.0 {
            continue;
        }
        if ix == 0 {
            dbg!();
        }

        let map_pt = |x: f64, y: f64| {
            let mut s = mat.transform_vec2([x, y].into());
            s.y *= -1.0;
            if ix % 1000 == 0 {
                println!("map_pt({x}, {y}) = {s:?}");
            }
            if s.x >= 0.0 && s.x < px_dims.x as f64 && s.y >= 0.0 && s.y < px_dims.y as f64 {
                Some((s.x, s.y))
            } else {
                None
            }
        };

        // let map_pt = |x: f64, y: f64| {
        //     let xn = (x - t0) / (t1 - t0);
        //     let yn = (y - q0) / (q1 - q0);
        //     (xn * px_dims.x as f64, yn * px_dims.y as f64)
        // };

        let start = map_pt(t0, q0);
        let end = map_pt(t1, q1);

        if let Some((p0, p1)) = start.zip(end) {
            if ix == 0 {
                dbg!();
            }
            // for ((px, py), v) in XiaolinWu::<f64, i32>::new(map_pt(t0, q0), map_pt(t1, q1)) {
            for ((px, py), v) in XiaolinWu::<f64, i32>::new(p0, p1) {
                if px >= 0 && px < px_dims.x as i32 && py >= 0 && py < px_dims.y as i32 {
                    let py = px_dims.y as i32 - py - 1;
                    let ix = (px + py * px_dims.x as i32) as usize;
                    if ix < pixels.len() {
                        let alpha = (v * 255.0) as u8;
                        pixels[ix] = [0, 0, 0, alpha];
                        if ix % 1000 == 0 {
                            println!("setting pixel ({px}, {py})");
                        }
                    }
                }
                //
            }
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
