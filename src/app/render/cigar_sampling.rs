use bevy::{
    math::DVec2,
    prelude::*,
    render::view::RenderLayers,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_mod_picking::prelude::Pickable;

use crate::{
    app::{view::AlignmentViewport, ForegroundColor, SequencePairTile},
    math_conv::*,
};

use super::{AlignmentRenderTarget, AlignmentViewer, MainAlignmentView};

pub struct CigarSamplingRenderPlugin;

impl Plugin for CigarSamplingRenderPlugin {
    fn build(&self, app: &mut App) {
        // app.add_systems(Startup, setup_render_tile_targets);
        // app.add_systems(
        //     Startup,
        //     setup_tile_debug_time.after(setup_render_tile_targets),
        // );

        // app.add_plugins(stress_test::StressTestPlugin);

        app.add_systems(Startup, |mut commands: Commands| {
            commands.spawn((
                SpatialBundle::default(),
                RenderTileGrid {
                    rows: 4,
                    columns: 4,
                },
                RenderTileGridCanvasSize {
                    pixels: [800, 600].into(),
                },
                MainAlignmentView,
            ));
        });

        app.add_systems(
            PreUpdate,
            (
                (
                    //
                    spawn_render_grid_children,
                    main_view_render_tile_init,
                )
                    .chain(),
                update_main_viewport_render_grid,
                update_viewport_locked_render_tile_params,
            )
                .chain(),
        );
        // ).add_systems(

        // )

        // app.add_systems(Startup, setup_render_tiles_new);
        // app.add_systems(Startup, setup_tile_debug_time.after(setup_render_tiles_new));

        // app.add_systems(Update, update_tile_debug_time);
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
            (update_image_from_task, cleanup_render_task)
                .chain()
                .before(update_viewport_locked_render_tile_params),
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

#[derive(Resource, Component, Clone, Copy)]
struct RenderTileGridCanvasSize {
    pixels: UVec2,
}

#[derive(Resource, Component, Clone, Copy)]
struct RenderTileGrid {
    // pixel_dims: UVec2,
    rows: usize,
    columns: usize,
}

// creates the children for the `RenderTileGrid`
fn spawn_render_grid_children(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,

    render_tile_grids: Query<
        (Entity, &RenderTileGrid, &RenderTileGridCanvasSize),
        Changed<RenderTileGrid>,
    >,
) {
    for (grid_entity, grid_size, canvas_size) in render_tile_grids.iter() {
        println!("spawning tiles as children on grid");
        let width = canvas_size.pixels.x / grid_size.rows as u32;
        let height = canvas_size.pixels.y / grid_size.columns as u32;

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        commands
            .entity(grid_entity)
            .despawn_descendants()
            .with_children(|parent| {
                //

                for row in 0..grid_size.rows {
                    for column in 0..grid_size.columns {
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
                        let _tile = parent
                            .spawn((
                                RenderTile {
                                    // size: win_size,
                                    tile_grid_pos: UVec2::new(column as u32, row as u32),
                                    ..default()
                                },
                                // RenderLayers::layer(1),
                                SpatialBundle::default(),
                                // Pickable {
                                //     should_block_lower: false,
                                //     is_hoverable: false,
                                // },
                            ))
                            // .insert(SpriteBundle {
                            //     sprite: Sprite {
                            //         // anchor: bevy::sprite::Anchor::TopLeft,
                            //         anchor: bevy::sprite::Anchor::Center,
                            //         ..default()
                            //     },
                            //     ..default()
                            // })
                            .insert(img_handle.clone())
                            .id();
                    }
                }
            });
    }
}

// inserts additional components to the grid marked as the "main view",
// which ensures that the tiles get rendered to the correct camera
// & are updated as the "main viewport" changes
//
// must run after `spawn_render_grid_children`
fn main_view_render_tile_init(
    mut commands: Commands,

    viewport: Res<AlignmentViewport>,

    render_grids: Query<
        // (Entity, &Children, &RenderTileGrid),
        (Entity, &Children),
        (With<super::MainAlignmentView>, Changed<Children>),
        // (With<super::MainAlignmentView>, Added<RenderTileGrid>),
    >,
    // tiles: Query<(&RenderTile, &Handle<Image>)>,
) {
    for (grid_ent, grid_children) in render_grids.iter() {
        commands.entity(grid_ent).insert(viewport.clone());

        for &child in grid_children.iter() {
            commands.entity(child).insert((
                RenderLayers::layer(1),
                Pickable {
                    should_block_lower: false,
                    is_hoverable: false,
                },
                Sprite {
                    anchor: bevy::sprite::Anchor::Center,
                    ..default()
                },
            ));
        }
    }

    //
}

/*
// update `RenderTile`'s `view` and `dims` based on the parent's view
// and size
fn update_render_grid_tile_views(
    render_tile_grids: Query<(&Children, &RenderTileGrid)>,

    mut render_tiles: Query<(
        &mut RenderTile,
        // Option<&crate::app::view::AlignmentViewport>,
    )>,
) {
    for (children, tile_grid) in render_tile_grids.iter() {
        for &tile_entity in children.iter() {
            // let width = win_size.x / tile_grid.rows as u32;
            // let height = win_size.y / tile_grid.columns as u32;

            let Ok(render_tile) = render_tiles.get_mut(tile_entity) else {
                continue;
            };

            //
        }
    }
}
*/

fn update_main_viewport_render_grid(
    viewport: Res<AlignmentViewport>,

    windows: Query<&Window>,

    mut render_tile_grids: Query<
        (&mut AlignmentViewport, &mut RenderTileGridCanvasSize),
        (With<MainAlignmentView>, With<RenderTileGrid>),
    >,
) {
    let window = windows.single();
    let canvas_size = RenderTileGridCanvasSize {
        pixels: window.physical_size(),
    };

    for (mut render_viewport, mut grid_canvas_size) in render_tile_grids.iter_mut() {
        *render_viewport = viewport.clone();
        *grid_canvas_size = canvas_size;
    }
}

// run after systems that update the parent/grid's `AlignmentViewport`
fn update_viewport_locked_render_tile_params(
    mut commands: Commands,
    //
    // render_tile_grids: Query<(&RenderTileGrid, &AlignmentViewport, &Children)>,
    render_grids: Query<(
        Entity,
        &RenderTileGrid,
        &RenderTileGridCanvasSize,
        &AlignmentViewport,
        &Children,
    )>,

    mut render_tiles: Query<
        &mut RenderTile,
        // Option<&crate::app::view::AlignmentViewport>,
    >,
) {
    for (grid_ent, render_grid, grid_canvas_size, render_view, children) in render_grids.iter() {
        let win_size = grid_canvas_size.pixels;
        // let win_size = win_size_px.as_vec2();

        let tile_dims = UVec2::new(
            win_size.x / render_grid.columns as u32,
            win_size.y / render_grid.rows as u32,
        );

        let world_tile_size = DVec2::new(
            render_view.view.width() / render_grid.columns as f64,
            render_view.view.height() / render_grid.rows as f64,
        );
        println!("parent grid {grid_ent:?} - {} children", children.len());

        for &tile_ent in children.iter() {
            dbg!();
            let Ok(mut tile) = render_tiles.get_mut(tile_ent) else {
                continue;
            };
            dbg!();
            //
            // let tile_bounds =
            let tile_pos = tile.tile_grid_pos;
            let s_x0 = (tile_pos.x * tile_dims.x) as f64;
            let s_y0 = (tile_pos.y * tile_dims.y) as f64;

            let s_x1 = s_x0 + tile_dims.x as f64;
            let s_y1 = s_y0 + tile_dims.y as f64;

            // compute world viewport corresponding to tile
            let w_x0 = render_view.view.x_min + tile_pos.x as f64 * world_tile_size.x;
            let w_y0 = render_view.view.y_min + tile_pos.y as f64 * world_tile_size.y;

            let w_x1 = w_x0 + world_tile_size.x;
            let w_y1 = w_y0 + world_tile_size.y;

            let tile_bounds = crate::view::View {
                x_min: w_x0,
                x_max: w_x1,
                y_min: w_y0,
                y_max: w_y1,
            };

            tile.view = Some(tile_bounds);
            tile.size = tile_dims;
        }
    }

    // update the tiles with the "main view" marker entity on their controller/parent entity
    // based on the current viewport & active layout
    //

    // for (children, tile_grid) in render_tile_grids.iter() {
    //     for &tile_entity in children.iter() {
    //         // let width = win_size.x / tile_grid.rows as u32;
    //         // let height = win_size.y / tile_grid.columns as u32;

    //         let Ok(render_tile) = render_tiles.get_mut(tile_entity) else {
    //             continue;
    //         };

    //         //
    //     }
    // }

    //
}

fn spawn_render_tasks(
    mut commands: Commands,

    alignments: Res<crate::Alignments>,
    alignment_grid: Res<crate::AlignmentGrid>,

    render_tile_grids: Query<(&RenderTileGrid, &Children)>,
    // mut tiles: Query<(&mut RenderTile, Has<RenderTask>)>,
    // mut tiles: Query<&mut RenderTile>,
    // mut tiles: Query<(&mut RenderTile, &AlignmentViewport), Without<RenderTask>>,
    tiles: Query<(Entity, &RenderTile), Without<RenderTask>>,
) {
    let task_pool = AsyncComputeTaskPool::get();

    for (render_grid, children) in render_tile_grids.iter() {
        let mut tiles_iter = tiles.iter_many(children);
        // dbg!();

        // while let Some((mut render_tile, is_rendering)) = tiles_iter.fetch_next() {
        // while let Some((mut tile, tile_bounds)) = tiles_iter.fetch_next() {
        while let Some((tile_ent, tile)) = tiles_iter.fetch_next() {
            // dbg!();
            let Some(tile_bounds) = tile.view else {
                continue;
            };

            dbg!();
            if let Some(last) = tile.last_update.as_ref() {
                let ms = last.elapsed().as_millis();
                if ms < 100 {
                    // println!("skipping due to timer");
                    continue;
                } else {
                    // println!("timer lapsed; updating");
                }
            }
            // dbg!();

            let canvas_size = tile.size;

            let params = RenderParams {
                view: tile_bounds,
                canvas_size,
            };

            //

            let w_x0 = params.view.x_min;
            let w_x1 = params.view.x_max;
            let w_y0 = params.view.y_min;
            let w_y1 = params.view.y_max;

            let tgts = alignment_grid.x_axis.tiles_covered_by_range(w_x0..=w_x1);
            let qrys = alignment_grid.y_axis.tiles_covered_by_range(w_y0..=w_y1);
            let Some((tgts, qrys)) = tgts.zip(qrys) else {
                continue;
            };
            // dbg!();
            let qrys = qrys.collect::<Vec<_>>();

            let tile_pairs = tgts.flat_map(|tgt| qrys.iter().map(move |qry| (tgt, *qry)));

            let tile_positions = tile_pairs
                .filter_map(|(target, query)| {
                    let xs = alignment_grid.x_axis.sequence_axis_range(target)?;
                    let ys = alignment_grid.y_axis.sequence_axis_range(query)?;

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

            let dbg_bg_color = {
                let pos = tile.tile_grid_pos;
                let cols = render_grid.columns;

                let i = pos.x + render_grid.columns as u32 * pos.y;
                // let x = (pos.x + render_grid.columns as u32 * pos.y) as f32
                //     / (render_grid.columns * render_grid.rows) as f32;
                // let x = pos.x as f32 / cols as f32;
                // let y = pos.y as f32 / render_grid.rows as f32;

                let hue = i as f32 / (render_grid.columns * render_grid.rows) as f32;
                let lgt = 0.5;
                // let lgt = ((y * std::f32::consts::PI).sin() * 0.25) + 0.5;

                let color = Color::hsl(hue * 360.0, 0.8, lgt).to_srgba();
                let map = |chn: f32| ((chn * 255.0) as u8).min(160);

                [map(color.red), map(color.green), map(color.blue), 160]
            };

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

                // async_io::Timer::after(std::time::Duration::from_millis(1_000)).await;

                rasterize_alignments_in_tile(dbg_bg_color, tile_bounds, canvas_size, alignments)
            });

            println!(
                "spawning task for tile [{:?}][{}, {}]\t view: {:?}",
                tile_ent, tile.tile_grid_pos.x, tile.tile_grid_pos.y, tile.view
            );

            commands
                .entity(tile_ent)
                .insert(RenderTask { task, params });

            //
        }
        // for mut render_tile in tiles.iter_many_mut(children) {
        //     //
        // }
        // for (mut render_tile, is_rendering) in tiles.iter_many_mut(children) {
        //     //
        // }
    }
}

fn rasterize_alignments_in_tile<'a, S, I>(
    dbg_bg_color: [u8; 4],
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
    pixels.fill(dbg_bg_color);

    let vx_min = tile_bounds.x_min;
    let vx_max = tile_bounds.x_max;

    let bp_per_px = tile_bounds.width() / tile_dims.x as f64;

    let mut mask_buf = vec![0u8; pixels.len()];

    let mut path_commands: Vec<zeno::Command> = Vec::new();
    // let mut line_buf = Vec::new();

    let mut embiggened = 0;
    let mut total = 0;

    let mut total_dst = 0f64;

    let mut draw_dot = |mask_buf: &mut [u8], x: f32, y: f32, rad: f32| {
        let radplus = rad + 1.0;
        let x_min = (x - radplus).floor() as usize;
        let x_max = (x + radplus).ceil() as usize;
        let y_min = (y - radplus).floor() as usize;
        let y_max = (y + radplus).ceil() as usize;

        let x0 = x;
        let y0 = y;
        // let x0 = x + 0.5;
        // let y0 = y + 0.5;

        let rad_sq = ((rad * rad) - 0.5).max(0.5);

        for x in x_min..x_max {
            for y in y_min..y_max {
                //
                let i = (x as usize) + (y as usize * tile_dims.x as usize);

                let x1 = x as f32 + 0.5;
                let y1 = y as f32 + 0.5;

                let val = (x1 - x0).powi(2) + (y1 - y0).powi(2);
                // if val < rad_sq && i < mask_buf.len() {
                if i < mask_buf.len() {
                    let px = mask_buf[i];

                    let d = (rad_sq.sqrt() - val.sqrt()).clamp(0.0, 1.0);

                    mask_buf[i] = px.max((d * 255.0) as u8);
                }
                // let dist = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            }
        }
    };

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

            // let t0 = std::time::Instant::now();
            // let mut count = 0;
            // line_buf.clear();
            let iter = alignment.iter_target_range(loc_min..loc_max);
            let mut cmd_iter =
                CigarScreenPathStrokeIter::new(tile_bounds, tile_dims, seq_min, iter);

            path_commands.clear();

            while let Some(cmd) = cmd_iter.emit_next() {
                path_commands.push(cmd);
            }

            /*
            if path_commands.len() == 1 {

            } else if path_commands.len() == 2 {

            } else {

            }
            */

            use zeno::Command as Cmd;

            if let [Cmd::MoveTo(start), Cmd::MoveTo(end) | Cmd::LineTo(end)] =
                path_commands.as_slice()
            {
                let dist = end.distance_to(*start) as f64;
                total_dst += dist;
                if dist < bp_per_px {
                    let d = dist as f32;
                    let x = start.x + 0.5 * (end.x - start.x);
                    let y = start.y + 0.5 * (end.y - start.y);
                    draw_dot(&mut mask_buf, x, y, 1.5);
                    // let i = end.x as usize + end.y as usize * tile_dims.x as usize;
                    // if i < mask_buf.len() {
                    //     mask_buf[i] = 255;
                    // }

                    embiggened += 1;
                    total += 1;

                    continue;
                }
            }

            /*
            if let Some(
                s @ (
                    zeno::Command::MoveTo(start),
                    zeno::Command::LineTo(end) | zeno::Command::MoveTo(end),
                    // zeno::Command::MoveTo(end),
                ),
            ) = path_commands
                .first()
                .cloned()
                .zip(path_commands.last().cloned())
            {
                let dist = end.distance_to(start) as f64;
                total_dst += dist;
                if dist < bp_per_px {
                    /*
                    let angle = (end.y - start.y).atan2(end.x - start.x);

                    path_commands.clear();
                    path_commands.push(s.0);

                    let end_point = zeno::Vector {
                        x: start.x + angle.cos() * 2.0,
                        y: start.y + angle.sin() * 2.0,
                    };

                    path_commands.push(zeno::Command::LineTo(end_point));
                    */

                    let i = end.x as usize + end.y as usize * tile_dims.x as usize;
                    if i < mask_buf.len() {
                        mask_buf[i] = 255;
                    }

                    embiggened += 1;
                    total += 1;

                    continue;
                }
            }
            */

            total += 1;
            zeno::Mask::new(&path_commands)
                .size(tile_dims.x, tile_dims.y)
                .style(zeno::Stroke::new(2.0))
                .origin(zeno::Origin::TopLeft)
                .render_into(&mut mask_buf, None);
            // path_commands.extend();
        }
    }

    for (val, px) in std::iter::zip(mask_buf, pixels) {
        if val > 0 {
            // let val = val.max(32);
            let val = val.max(128);
            // let val = val.max(255);
            // let val = (val as f32 * 10.0).max(255.0) as u8;

            let [rb, gb, bb, ab] = *px;

            let ab = ab as f32 / 255.0;
            let aa = val as f32 / 255.0;
            let ca = 0.0;

            let a = aa + ab * (1.0 - aa);

            let color = |c: u8| {
                let f = ca * aa + (c as f32 / 255.0) * ab;
                (f * 255.0) as u8
            };

            let r = color(rb);
            let g = color(gb);
            let b = color(bb);

            *px = [r, g, b, (a * 255.0) as u8];
            // *px = [0, 0, 0, 64];
        }
    }

    println!(
        "grew {embiggened} out of {total} alignments\taverage length: {}",
        total_dst / embiggened as f64
    );

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

// run after whatever updates the `AlignmentViewport`
fn update_render_tile_transforms(
    // viewport: Res<crate::app::view::AlignmentViewport>,
    // render_grid: Res<RenderTileGrid>,
    // windows: Query<&Window>,
    render_grids: Query<(
        &RenderTileGrid,
        &RenderTileGridCanvasSize,
        &AlignmentViewport,
        &Children,
    )>,

    mut tiles: Query<(&mut Transform, &RenderTile)>,
    // mut tiles: Query<(&SequencePairTile, &mut Visibility, &mut Transform), With<RenderTileTarget>>,
) {
    for (render_grid, grid_canvas_size, render_view, children) in render_grids.iter() {
        let win_size_px = grid_canvas_size.pixels;
        let win_size = win_size_px.as_vec2();

        let tile_dims = Vec2::new(
            win_size.x / render_grid.columns as f32,
            win_size.y / render_grid.rows as f32,
        );

        let top_left = Vec2::new(win_size.x * -0.5, win_size.y * -0.5);

        // let tile_screen_center = |pos: UVec2| {
        let tile_screen_top_left = |pos: UVec2| {
            let offset = pos.as_vec2() * tile_dims;
            top_left + offset
        };

        for &child in children.iter() {
            // }

            // for (mut transform, render_tile) in tiles.iter_mut() {

            let Ok((mut transform, render_tile)) = tiles.get_mut(child) else {
                continue;
            };

            let tpos = render_tile.tile_grid_pos;
            let pos = tile_screen_top_left(tpos) + tile_dims * 0.5;

            let old_view = render_tile.last_rendered.as_ref().map(|params| params.view);
            let new_view = render_tile.view;

            if let Some((old_view, new_view)) = old_view.zip(new_view) {
                let old_mid = old_view.center();
                let new_mid = new_view.center();

                let world_delta = new_mid - old_mid;
                let norm_delta = world_delta / new_view.size();

                let screen_delta = Vec2::new(
                    //
                    norm_delta.x as f32 * tile_dims.x,
                    norm_delta.y as f32 * tile_dims.y,
                );

                let grid_w = win_size_px.x as f64;
                let grid_h = win_size_px.y as f64;

                let old_w_scale = old_view.width() / grid_w;
                let old_h_scale = old_view.height() / grid_h;
                let new_w_scale = new_view.width() / grid_w;
                let new_h_scale = new_view.height() / grid_h;

                let scale = Vec3::new(
                    (old_w_scale / new_w_scale) as f32,
                    (old_h_scale / new_h_scale) as f32,
                    1.0,
                );

                transform.translation.x = pos.x - screen_delta.x;
                transform.translation.y = pos.y - screen_delta.y;
                transform.scale = scale;
            } else {
                transform.translation.x = pos.x;
                transform.translation.y = pos.y;
            }
        }
    }
}

#[derive(Component)]
struct RenderTask {
    task: Task<Vec<u8>>,

    params: RenderParams,
}

fn spawn_render_tasks_old(
    mut commands: Commands,

    // time: Res<Time>,
    // keys: Res<ButtonInput<KeyCode>>,
    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,
    viewport: Res<crate::app::view::AlignmentViewport>,
    render_grid: Res<RenderTileGrid>,

    windows: Query<&Window>,
    mut tiles: Query<
        (
            Entity,
            // &SequencePairTile,
            // &Transform,
            &mut RenderTile,
            Has<RenderTask>,
        ),
        // Without<RenderTask>,
    >,
    // keys: Res<ButtonInput<KeyCode>>,
    // mut enabled: Local<bool>,
    // mut view_timer_state: Local<Option<(Option<std::time::Instant>, crate::view::View)>>,
    // mut spawn_task_timer: Local<Option<std::time::Instant>>,
) {
    // if keys.just_pressed(KeyCode::Space) {
    //     *enabled = true;
    // }
    // if !*enabled {
    //     return;
    // }

    /*
    let should_spawn = spawn_task_timer
        .as_ref()
        .map(|t| t.elapsed().as_millis() > 200)
        .unwrap_or(false);

    if viewport.is_changed() {

    }
    */

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

    for (tile_ent, mut tile, is_rendering) in tiles.iter_mut() {
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

        tile.view = Some(tile_bounds);

        if is_rendering {
            continue;
        }

        if let Some(last) = tile.last_update.as_ref() {
            let ms = last.elapsed().as_millis();
            if ms < 100 {
                // println!("skipping due to timer");
                continue;
            } else {
                // println!("timer lapsed; updating");
            }
        }

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

        let dbg_bg_color = {
            let pos = tile.tile_grid_pos;
            let cols = render_grid.columns;

            let i = pos.x + render_grid.columns as u32 * pos.y;
            // let x = (pos.x + render_grid.columns as u32 * pos.y) as f32
            //     / (render_grid.columns * render_grid.rows) as f32;
            // let x = pos.x as f32 / cols as f32;
            // let y = pos.y as f32 / render_grid.rows as f32;

            let hue = i as f32 / (render_grid.columns * render_grid.rows) as f32;
            let lgt = 0.5;
            // let lgt = ((y * std::f32::consts::PI).sin() * 0.25) + 0.5;

            let color = Color::hsl(hue * 360.0, 0.8, lgt).to_srgba();
            let map = |chn: f32| ((chn * 255.0) as u8).min(160);

            [map(color.red), map(color.green), map(color.blue), 160]
        };

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

            // async_io::Timer::after(std::time::Duration::from_millis(1_000)).await;

            rasterize_alignments_in_tile(dbg_bg_color, tile_bounds, tile_dims, alignments)
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

struct CigarScreenPathStrokeIter<I> {
    iter: I,

    view: crate::view::View,
    dims: UVec2,
    seq_pair_world_offset: DVec2,

    bp_per_px: f64,

    path_open: Option<DVec2>,
    last_end: Option<DVec2>,
}

impl<I: Iterator<Item = crate::paf::AlignmentIterItem>> CigarScreenPathStrokeIter<I> {
    fn new(
        view: crate::view::View,
        dims: UVec2,
        seq_pair_world_offset: impl Into<[f64; 2]>,
        iter: I,
    ) -> Self {
        let bp_per_px = view.width() / dims.x as f64;

        Self {
            iter,

            view,
            dims,
            seq_pair_world_offset: seq_pair_world_offset.into().into(),

            bp_per_px,

            // started: false,
            path_open: None,
            last_end: None,
            // last_world_point: None,
        }
    }

    fn emit_next(&mut self) -> Option<zeno::Command> {
        /*
        the iterator has to know whether the path is open or not; no way around it

        */

        use crate::CigarOp::{Eq, D, I, M, X};
        use zeno::{Command, Vector};

        let origin = self.seq_pair_world_offset;
        let screen_dims = [self.dims.x as f32, self.dims.y as f32];
        let scale_sq = self.bp_per_px.max(1.0).powi(2);

        let mut loop_count = 0;

        loop {
            // let cg_item = self.iter.next()?;
            let Some(cg_item) = self.iter.next() else {
                // emit a final "LineTo" if the inner iterator is done & it's appropriate to do so
                if self.path_open.is_some() {
                    self.path_open = None;
                    if let Some(end) = self.last_end {
                        let ultraviolet::Vec2 { x, y } =
                            self.view.map_world_to_screen(screen_dims, [end.x, end.y]);
                        return Some(Command::LineTo(Vector { x, y }));
                    }
                }
                return None;
            };

            let op = cg_item.op;
            let len = cg_item.op_count;
            // let len = cg_item.op_count as f64;
            let tgt_r = cg_item.target_seq_range();
            let qry_r = cg_item.query_seq_range();

            let dx = op.target_delta(len) as f64;
            let dy = op.query_delta(len) as f64;

            let w_start = {
                let x = (tgt_r.start as f64) + origin.x;
                let y = (qry_r.start as f64) + origin.y;
                DVec2::new(x, y)
            };
            let w_end = {
                let x = (tgt_r.end as f64) + origin.x;
                let y = (qry_r.end as f64) + origin.y;
                DVec2::new(x, y)
            };
            self.last_end = Some(w_end);

            if let Some(path_0) = self.path_open {
                let diff = path_0 - w_start;
                let dist_sq = diff.length_squared();

                if dist_sq > scale_sq {
                    if matches!(op, M | Eq | X) {
                        self.path_open = Some(w_start);
                        let s_start = self
                            .view
                            .map_world_to_screen(screen_dims, [w_start.x, w_start.y]);
                        // println!("emitting LineTo({}, {})", s_start.x, s_start.y);
                        // println!("looped {loop_count} before emitting");
                        return Some(Command::LineTo(Vector {
                            x: s_start.x,
                            y: s_start.y,
                        }));
                    } else {
                        // emit MoveTo(w_end)... maybe?
                        // or actually just keep going, i guess?
                    }
                }
            } else {
                if matches!(op, M | Eq | X) {
                    self.path_open = Some(w_start);
                    let s_start = self
                        .view
                        .map_world_to_screen(screen_dims, [w_start.x, w_start.y]);
                    // println!("emitting MoveTo({}, {})", s_start.x, s_start.y);
                    // println!("looped {loop_count} before emitting");
                    return Some(Command::MoveTo(Vector {
                        x: s_start.x,
                        y: s_start.y,
                    }));
                } else {
                    // there's no open path, and we're not going to emit a stroke,
                    // so... don't do anything
                }
            }

            loop_count += 1;
        }
    }
}

/*
rather than working directly with a world view and exact tile dimensions,
and outputting tile-space (pixel) zeno path commands, as the
`CigarScreenPathStrokeIter` above,

this iterator only takes the view scale and alignment-local target range
into account, outputting world-unit (basepair) path commands with a minimum
size based on the `bp_per_px` scale. these positions must then be offset
by the alignment location and seq. pair offset before rasterization
*/
struct SimpleCigarPathStrokeIter<I> {
    iter: I,
    bp_per_px: f64,

    target_range: std::ops::Range<u64>,

    current_item: Option<crate::CigarIterItem>,
    last_emitted: Option<StrokeCmd>,
    cursor: DVec2,
    // cursor: CigarPathCursor,
    // state: Option<CigarPathState>,
    // current_op: Option<(crate::CigarOp, u32)>,

    // segment_start: Option<[f64; 2]>,
}

struct CigarPathCursor {
    offset: DVec2,
    stroke_start: Option<DVec2>,
    // drawing: bool,
}

impl<I: Iterator<Item = crate::CigarIterItem>> SimpleCigarPathStrokeIter<I> {
    // target_range must match `iter`'s target range
    fn new(bp_per_px: f64, target_range: std::ops::Range<u64>, iter: I) -> Self {
        Self {
            iter,
            bp_per_px,

            target_range,

            current_item: None,
            last_emitted: None,
            cursor: DVec2::ZERO,
            /*
            cursor: CigarPathCursor {
                offset: DVec2::ZERO,
                stroke_start: None,
                // drawing: false,
            }, // state: None,
            */
        }
    }

    //

    fn process_item(&mut self, item: crate::CigarIterItem) -> Option<StrokeCmd> {
        todo!();
    }

    fn emit(&mut self) -> Option<StrokeCmd> {
        //
        use crate::CigarOp::{Eq, D, I, M, X};
        if self.last_emitted.is_none() {
            // NB: this will get moved by the seq. pair & alignment offset
            // later... mostly here to help me think
            let cmd = StrokeCmd::MoveTo(DVec2::ZERO);
            self.last_emitted = Some(cmd);
            return Some(cmd);
        }

        let scale_sq = self.bp_per_px * self.bp_per_px;

        loop {
            if let Some(item) = self.current_item.take() {
                if matches!(item.op, M | Eq | X) {
                    let cmd = StrokeCmd::MoveTo(DVec2::ZERO);
                    self.last_emitted = Some(cmd);
                    // return Some()
                }
            }

            let Some(item) = self.iter.next() else {
                todo!();
                // output final LineTo command if needed
            };

            let op = item.op;
            let len = item.op_count;
            let dx = op.target_delta(len) as f64;
            let dy = op.query_delta(len) as f64;

            let len_sq = dx * dx + dy * dy;

            if matches!(op, M | Eq | X) {
                if len_sq >= scale_sq {
                    // this one's enough to emit by itself
                }
            }

            //
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum StrokeCmd {
    MoveTo(DVec2),
    LineTo(DVec2),
}

impl<I: Iterator<Item = crate::CigarIterItem>> Iterator for SimpleCigarPathStrokeIter<I> {
    type Item = StrokeCmd;

    fn next(&mut self) -> Option<Self::Item> {
        todo!();
        /*
        match self.state.take() {
            Some(CigarPathState::Done) => None,
            None => {
                let Some(item) = self.iter.next() else {
                    self.state = Some(CigarPathState::Done);
                    return None;
                };

                // update state and recurse; won't even happen more than once
                self.state = Some(CigarPathState::StartOf(item));
                return self.next();
            }
            Some(CigarPathState::StartOf(item)) => {
                //

                self.state = Some(CigarPathState::EndOf(item));
                todo!();
            }
            Some(CigarPathState::EndOf(item)) => {
                //

                self.state = Some(CigarPathState::EndOf(item));
                todo!();
            }
        }

        // if self.current_op.is_none() {
        //     self.current_op = self.iter.next();
        // }
        */
    }
}

fn setup_render_tiles_new(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,

    windows: Query<&Window>,
) {
    let window = windows.single();
    let win_size = window.physical_size();

    let tile_grid = RenderTileGrid {
        // rows: 1,
        // columns: 1,
        // rows: 2,
        // columns: 2,
        // pixel_dims: win_size,
        rows: 4,
        columns: 4,
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
            let _tile = commands
                .spawn((
                    RenderTile {
                        // size: win_size,
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
                    sprite: Sprite {
                        // anchor: bevy::sprite::Anchor::TopLeft,
                        anchor: bevy::sprite::Anchor::Center,
                        ..default()
                    },
                    ..default()
                })
                .insert(img_handle.clone())
                .id();
        }
    }
}
