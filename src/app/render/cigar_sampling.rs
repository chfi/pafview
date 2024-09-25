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

        // app.add_plugins(stress_test::StressTestPlugin);

        app.add_systems(Startup, setup_render_tiles_new);
        app.add_systems(Startup, setup_tile_debug_time.after(setup_render_tiles_new));

        app.add_systems(Update, update_tile_debug_time);
        /*
            update_render_tile_transforms
            spawn_render_tasks
            finish_render_tasks
        */

        // app.add_systems(Update, update_r)

        // app.add_systems(
        //     //
        //     PreUpdate,
        //     finish_render_tasks,
        // )
        // .add_systems(PostUpdate, update_render_tile_transforms)
        app.add_systems(
            PostUpdate,
            (update_render_tile_transforms, spawn_render_tasks).chain(),
        )
        .add_systems(
            PreUpdate,
            (update_image_from_task, cleanup_render_task).chain(),
        );
    }
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
        // rows: 4,
        // columns: 4,
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

    // let mut first = true;

    // for [r,g,b,a] in tile_data.iter_mut() {
    // println!("tile dims: {tile_dims:?}");
    // for val in tile_data.iter_mut() {
    //     *val = [0, 0, 0, 255];
    // }

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
                    // if first {
                    //     first = false;
                    //     println!("plotting to pixel ({px},{py})");
                    // }
                }
            }
        }
    }
}

fn rasterize_alignments_in_tile<'a, S, I>(
    tile_bounds: crate::view::View,
    tile_dims: UVec2,
    // tile_mat: ultraviolet::DMat3,
    // grid: &crate::AlignmentGrid,
    seq_pairs: S,
    // alignments: impl IntoIterator<Item = (SequencePairTile, [DVec2; 2], &'a crate::Alignment)>,
    // ) -> Vec<[u8; 4]>
) -> Vec<u8>
where
    S: IntoIterator<Item = (SequencePairTile, [DVec2; 2], I)>,
    I: IntoIterator<Item = &'a crate::Alignment>,
{
    let px_count = (tile_dims.x * tile_dims.y) as usize;

    let mut buffer = vec![0u8; px_count * 4];
    let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut buffer);

    let vx_min = tile_bounds.x_min;
    let vx_max = tile_bounds.x_max;

    let mut total_time = 0;
    let mut total_lines = 0;

    let mut ix = 0;

    let mut line_buf = Vec::new();

    for (seq_pair, [seq_min, seq_max], alignments) in seq_pairs.into_iter() {
        // seq_min, seq_max encode the position of the seq. pair tile in the world

        for alignment in alignments {
            // dbg!();
            let alignment: &crate::Alignment = alignment;
            let loc = &alignment.location;
            let tgt_len = loc.target_range.end - loc.target_range.start;

            let al_min = seq_min.x + loc.target_range.start as f64;
            let al_max = al_min + tgt_len as f64;

            // println!("{al_min}, {al_max}");
            let cal_min = vx_min.clamp(al_min, al_max) as u64;
            let cal_max = vx_max.clamp(al_min, al_max) as u64;
            // let al_min = al_min.clamp(vx_min, vx_max) as u64;
            // let al_max = al_max.clamp(vx_min, vx_max) as u64;

            // println!("clamped: {al_min}, {al_max}");

            if cal_min == cal_max {
                // println!("no overlap; skipping");
                continue;
            }

            let loc_min = cal_min.checked_sub(seq_min.x as u64).unwrap_or_default();
            let loc_max = cal_max.checked_sub(seq_min.x as u64).unwrap_or_default();

            if loc_min == loc_max {
                // println!("no overlap after offset; skipping");
                continue;
            }

            let start = DVec2::new(loc.target_range.start as f64, loc.query_range.start as f64);
            let end = DVec2::new(loc.target_range.end as f64, loc.query_range.end as f64);
            let w0 = seq_min + start;
            let w1 = seq_min + end;
            rasterize_world_lines(tile_bounds, tile_dims, pixels, [[w0, w1]]);
            println!("rasterizing {w0:?}->{w1:?}");

            line_buf.clear();

            let t0 = std::time::Instant::now();
            let test_iter = alignment.cigar.whole_cigar();
            // println!(
            //     " >> [{ix}] built whole cigar iterator in {}us",
            //     t0.elapsed().as_micros()
            // );

            let t0 = std::time::Instant::now();
            // let lines = alignment
            let iter = alignment.iter_target_range(loc_min..loc_max);
            // println!(
            //     " >> [{ix}] built alignment iterator in {}us\trange {:?}\tloc. tgt {:?}",
            //     t0.elapsed().as_micros(),
            //     loc_min..loc_max,
            //     &loc.target_range,
            // );

            let t0 = std::time::Instant::now();
            let mut count = 0;
            line_buf.extend(iter.filter_map(|item| {
                let emit = count % 1000 == 0;
                count += 1;

                if emit {
                    let tgt = item.target_seq_range();
                    let qry = item.query_seq_range();

                    let w0 = seq_min + DVec2::new(tgt.start as f64, qry.start as f64);
                    let w1 = seq_min + DVec2::new(tgt.end as f64, qry.end as f64);
                    Some([w0, w1])
                } else {
                    None
                }
            }));

            // if ix % 1000 == 0 {
            //     println!("{ix} processed");
            // }
            // .collect::<Vec<_>>();
            // println!(
            //     " >> [{ix}] collected {} lines out of {} ops in {}ms",
            //     line_buf.len(),
            //     count,
            //     t0.elapsed().as_millis()
            // );
            total_lines += line_buf.len();
            total_time += t0.elapsed().as_nanos();
            ix += 1;

            // println!(">> TODO sample with range {loc_min}..{loc_max}");
            // line_buf.clear();

            // let offset = DVec2::new(loc.target_range.start as f64, loc.query_range.start as f64);

            // let range = (loc_min - alignment.location.target_range.start)
            //     ..(loc_max - alignment.location.target_range.start);

            // rasterize_world_lines(tile_bounds, tile_dims, pixels, line_buf.iter().copied());

            /*

            for item in alignment.cigar.iter_target_range(range) {

            }
            */
            // for item in alignment.
        }
        let ms = total_time / 1000;
        println!("collected {total_lines} lines in {ms}ms",);

        //
    }

    buffer
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

/*
fn setup_render_tile_targets(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,

    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    // color_schemes: Res<AlignmentColorSchemes>,
    cli_args: Res<crate::cli::Cli>,
) {
    for (pair @ (tgt_id, qry_id), alignments) in alignments.pairs() {
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
*/

fn update_render_tile_transforms(
    // viewport: Res<crate::app::view::AlignmentViewport>,
    render_grid: Res<RenderTileGrid>,
    windows: Query<&Window>,

    mut tiles: Query<(&mut Transform, &RenderTile)>,
    // mut tiles: Query<(&SequencePairTile, &mut Visibility, &mut Transform), With<RenderTileTarget>>,
) {
    let window = windows.single();
    let win_size = window.size();

    // let tile_dims = UVec2::new(
    //     win_size.x / render_grid.columns as u32,
    //     win_size.y / render_grid.rows as u32,
    // );

    let tile_dims = Vec2::new(
        win_size.x / render_grid.columns as f32,
        win_size.y / render_grid.rows as f32,
    );

    // let top_left = win_size * -0.5;
    let top_left = win_size * 0.0;

    for (mut transform, render_tile) in tiles.iter_mut() {
        let tpos = render_tile.tile_grid_pos;
        // let pos = top_left + tile_dims * tpos.as_vec2() - tile_dims * 0.5;
        let pos = top_left + tile_dims * tpos.as_vec2();

        transform.translation.x = pos.x;
        transform.translation.y = pos.y;

        // println!(
        //     "transform -- T: {:?}\tS: {:?}",
        //     transform.translation, transform.scale,
        // );
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

    keys: Res<ButtonInput<KeyCode>>,
    mut enabled: Local<bool>,
) {
    if keys.just_pressed(KeyCode::Space) {
        *enabled = true;
    }

    if !*enabled {
        return;
    }

    let window = windows.single();
    let win_size = window.physical_size();

    let tile_dims = UVec2::new(
        win_size.x / render_grid.columns as u32,
        win_size.y / render_grid.rows as u32,
    );

    let world_tile = DVec2::new(
        viewport.view.width() / render_grid.columns as f64,
        viewport.view.height() / render_grid.rows as f64,
    );

    let task_pool = AsyncComputeTaskPool::get();

    let vx0 = viewport.view.x_min;
    let vy0 = viewport.view.y_min;

    for (tile_ent, tile) in tiles.iter() {
        if let Some(last) = tile.last_update.as_ref() {
            let ms = last.elapsed().as_millis();
            if ms < 100 {
                // println!("skipping due to timer");
                continue;
            } else {
                // println!("timer lapsed; updating");
            }
        }

        // position of tile on screen
        let tile_pos = tile.tile_grid_pos;
        let s_x0 = (tile_pos.x * tile_dims.x) as f64;
        let s_y0 = (tile_pos.y * tile_dims.y) as f64;

        let s_x1 = s_x0 + tile_dims.x as f64;
        let s_y1 = s_y0 + tile_dims.y as f64;

        // compute world viewport corresponding to tile
        let w_x0 = vx0 + tile_pos.x as f64 * world_tile.x;
        let w_y0 = vy0 + tile_pos.y as f64 * world_tile.y;

        let w_x1 = w_x0 + world_tile.x;
        let w_y1 = w_y0 + world_tile.y;

        let tile_bounds = crate::view::View {
            x_min: w_x0,
            x_max: w_x1,
            y_min: w_y0,
            y_max: w_y1,
        };

        let params = RenderParams {
            view: tile_bounds,
            canvas_size: tile_dims,
        };

        if Some(&params) == tile.last_rendered.as_ref() {
            continue;
        }

        println!("tile bounds: {tile_bounds:?}");

        // find alignments overlapping tile
        // first find overlapping seq. pairs
        let tgts = grid.x_axis.tiles_covered_by_range(w_x0..=w_x1);
        let qrys = grid.y_axis.tiles_covered_by_range(w_y0..=w_y1);
        let Some((tgts, qrys)) = tgts.zip(qrys) else {
            continue;
        };
        let qrys = qrys.collect::<Vec<_>>();

        let tile_pairs = tgts.flat_map(|tgt| qrys.iter().map(move |qry| (tgt, *qry)));

        let tile_positions = tile_pairs
            .filter_map(|(target, query)| {
                let xs = grid.x_axis.sequence_axis_range(target)?;
                let ys = grid.y_axis.sequence_axis_range(query)?;

                let x0 = xs.start as f64;
                let y0 = ys.start as f64;

                let x1 = xs.end as f64;
                let y1 = ys.end as f64;

                Some((
                    SequencePairTile { target, query },
                    [DVec2::new(x0, y0), DVec2::new(x1, y1)],
                ))
            })
            .collect::<Vec<_>>();

        let als = alignments.alignments.clone();
        let al_pairs = alignments.indices.clone();

        // spawn task
        let task = task_pool.spawn(async move {
            let mut count = 0;
            let alignments = tile_positions
                .into_iter()
                .filter_map(|(seq_pair, bounds)| {
                    let key = (seq_pair.target, seq_pair.query);
                    let als = al_pairs
                        .get(&key)?
                        .into_iter()
                        .filter_map(|ix| als.get(*ix))
                        .collect::<Vec<_>>();
                    count += als.len();
                    Some((seq_pair, bounds, als))
                })
                .collect::<Vec<_>>();

            println!(" >> {count} alignments to render");

            rasterize_alignments_in_tile(tile_bounds, tile_dims, alignments)
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

fn update_image_from_task(
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
        // dbg!();
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
    }
}

fn cleanup_render_task(
    mut commands: Commands,
    mut tiles: Query<(Entity, &mut RenderTile, &RenderTask), Changed<RenderTask>>,
) {
    for (ent, mut tile, task) in tiles.iter_mut() {
        if task.task.is_finished() {
            println!("render task finished");
            tile.last_rendered = Some(task.params.clone());
            tile.last_update = Some(std::time::Instant::now());
            commands.entity(ent).remove::<RenderTask>();
        }
    }
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

mod stress_test {
    use super::*;
    use bevy::prelude::*;

    pub(super) struct StressTestPlugin;

    impl Plugin for StressTestPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup_stress_test)
                .add_systems(Update, (spawn_task, finish_task).chain());
        }
    }

    #[derive(Component)]
    struct StressTestTile;

    #[derive(Component)]
    struct StressTestTask {
        task: Task<(Vec<u8>, UVec2)>,
    }

    fn setup_stress_test(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,

        windows: Query<&Window>,
    ) {
        let window = windows.single();
        let win_size = window.physical_size();

        let size = wgpu::Extent3d {
            width: win_size.x,
            height: win_size.y,
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
        let tile = commands
            .spawn((
                StressTestTile,
                RenderLayers::layer(1),
                // SpatialBundle::default(),
                // SpatialBundle {
                //     transform: Transform::from_xyz(0.0, 0.0, 1.0),
                //     ..default()
                // }
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

    fn spawn_task(
        mut commands: Commands,
        keys: Res<ButtonInput<KeyCode>>,

        tiles: Query<Entity, (With<StressTestTile>, Without<StressTestTask>)>,

        windows: Query<&Window>,

        mut lines: Local<Vec<[DVec2; 2]>>,
    ) {
        let window = windows.single();

        let create_lines = lines.is_empty() || keys.just_pressed(KeyCode::Enter);

        let view = crate::view::View {
            x_min: -1000.0,
            y_min: -1000.0,

            x_max: 1_001_000.0,
            y_max: 1_001_000.0,
        };

        if create_lines {
            lines.clear();

            use rand::prelude::*;
            let mut rng = rand::thread_rng();

            for i in 0..1_000 {
                // generate lines
                let x0 = rng.gen_range(0f64..=998_000.0);
                let x1 = rng.gen_range((x0 + 100.0)..=999_000.0);

                let y0 = rng.gen_range(0f64..=998_000.0);
                let y1 = rng.gen_range((y0 + 100.0)..=999_000.0);

                lines.push([DVec2::new(x0, y0), DVec2::new(x1, y1)]);
            }
        }

        let should_render = create_lines || keys.just_pressed(KeyCode::Space);

        if !should_render {
            return;
        }

        let task_pool = AsyncComputeTaskPool::get();
        for tile in tiles.iter() {
            let lines = lines.clone();
            let win_size = window.physical_size();
            let task = task_pool.spawn(async move {
                // let tile_size = UVec2::new(todo!(), todo!());
                let tile_size = win_size;
                let mut pixel_buffer = vec![0u8; (tile_size.x * tile_size.y) as usize * 4];

                let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut pixel_buffer);

                rasterize_world_lines(view, tile_size, pixels, lines);

                (pixel_buffer, tile_size)
            });

            commands.entity(tile).insert(StressTestTask { task });
        }
    }

    fn finish_task(
        mut commands: Commands,
        mut images: ResMut<Assets<Image>>,

        mut tiles: Query<
            (
                Entity,
                // &SequencePairTile,
                // &Transform,
                // &ViewVisibility,
                &Handle<Image>,
                // &mut StressTestTile,
                &mut StressTestTask,
            ),
            With<StressTestTile>,
        >,
    ) {
        for (tile, img_h, mut task) in tiles.iter_mut() {
            //
            let Some((data, img_size)) =
                bevy::tasks::block_on(bevy::tasks::poll_once(&mut task.task))
            else {
                continue;
            };

            let Some(image) = images.get_mut(img_h) else {
                continue;
            };

            image.resize(wgpu::Extent3d {
                width: img_size.x,
                height: img_size.y,
                depth_or_array_layers: 1,
            });

            image.texture_descriptor.size.width = img_size.x;
            image.texture_descriptor.size.height = img_size.y;
            image.data = data;

            commands.entity(tile).remove::<StressTestTask>();
        }
    }
}
