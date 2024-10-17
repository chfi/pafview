use avian2d::parry::bounding_volume::Aabb;
use bevy::{
    math::DVec2,
    prelude::*,
    render::view::RenderLayers,
    tasks::{AsyncComputeTaskPool, Task},
    utils::HashMap,
};
use bevy_inspector_egui::{
    inspector_options::ReflectInspectorOptions, quick::ResourceInspectorPlugin, InspectorOptions,
};
use bevy_mod_picking::prelude::Pickable;

use crate::{
    app::{
        alignments::layout::SeqPairLayout, view::AlignmentViewport, ForegroundColor,
        SequencePairTile,
    },
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
                set_render_tile_visibility,
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

        for &tile_ent in children.iter() {
            let Ok(mut tile) = render_tiles.get_mut(tile_ent) else {
                continue;
            };

            let tile_pos = tile.tile_grid_pos;
            let s_x0 = (tile_pos.x * tile_dims.x) as f64;
            let s_y0 = (tile_pos.y * tile_dims.y) as f64;

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
}

fn spawn_render_tasks(
    mut commands: Commands,

    alignments: Res<crate::Alignments>,
    // alignment_grid: Res<crate::AlignmentGrid>,
    default_layout: Res<crate::app::alignments::layout::DefaultLayout>,

    layouts: Res<Assets<SeqPairLayout>>,

    render_tile_grids: Query<(&RenderTileGrid, &Children, Option<&Handle<SeqPairLayout>>)>,

    tiles: Query<(Entity, &RenderTile), Without<RenderTask>>,
) {
    let task_pool = AsyncComputeTaskPool::get();

    for (render_grid, children, layout) in render_tile_grids.iter() {
        let layout = layout
            .or(Some(&default_layout.layout))
            .and_then(|h| layouts.get(h));

        let Some(layout) = layout else {
            continue;
        };

        for (tile_ent, tile) in tiles.iter_many(children) {
            let Some(tile_bounds) = tile.view else {
                continue;
            };

            if let Some(last) = tile.last_update.as_ref() {
                let ms = last.elapsed().as_millis();
                if ms < 100 {
                    // println!("skipping due to timer");
                    continue;
                } else {
                    // println!("timer lapsed; updating");
                }
            }

            let canvas_size = tile.size;

            let params = RenderParams {
                view: tile_bounds,
                canvas_size,
            };

            if Some(&params) == tile.last_rendered.as_ref() {
                continue;
            }

            let w_x0 = params.view.x_min;
            let w_x1 = params.view.x_max;
            let w_y0 = params.view.y_min;
            let w_y1 = params.view.y_max;

            let tiles = layout
                .layout_qbvh
                .tiles_in_rect(params.view.center(), params.view.size() * 0.5);

            let tile_aabbs = tiles
                .into_iter()
                .filter_map(|seq_pair| {
                    let aabb = layout.aabbs.get(&seq_pair)?;
                    Some((seq_pair, *aabb))
                })
                .collect::<Vec<_>>();

            let als = alignments.alignments.clone();
            let al_pairs = alignments.indices.clone();

            let dbg_bg_color = {
                let pos = tile.tile_grid_pos;
                let i = pos.x + render_grid.columns as u32 * pos.y;

                let hue = i as f32 / (render_grid.columns * render_grid.rows) as f32;
                let lgt = 0.5;

                let color = Color::hsl(hue * 360.0, 0.8, lgt).to_srgba();
                let map = |chn: f32| ((chn * 255.0) as u8).min(160);

                [map(color.red), map(color.green), map(color.blue), 160]
            };

            // spawn task
            let task = task_pool.spawn(async move {
                let mut count = 0;
                let alignments = tile_aabbs
                    .into_iter()
                    .filter_map(|(seq_pair, bounds)| {
                        let key = (seq_pair.target, seq_pair.query);
                        let als = al_pairs
                            .get(&key)?
                            .into_iter()
                            .filter_map(|ix| als.get(*ix))
                            .collect::<Vec<_>>();
                        count += als.len();

                        // let p0 = bounds.mins;
                        // let offset = DVec2::new(p0.x, p0.y);
                        Some((seq_pair, bounds, als))
                    })
                    .collect::<Vec<_>>();

                // println!(" >> {count} alignments to render");

                rasterize_alignments_in_tile(dbg_bg_color, tile_bounds, canvas_size, alignments)
            });

            // println!(
            //     "spawning task for tile [{:?}][{}, {}]\t view: {:?}",
            //     tile_ent, tile.tile_grid_pos.x, tile.tile_grid_pos.y, tile.view
            // );

            commands
                .entity(tile_ent)
                .insert(RenderTask { task, params });

            //
        }
    }
}

fn set_render_tile_visibility(mut tiles: Query<(&RenderTile, &mut Visibility)>) {
    for (tile, mut visibility) in tiles.iter_mut() {
        if tile.last_rendered.is_none() || tile.view.is_none() {
            *visibility = Visibility::Hidden;
        } else {
            *visibility = Visibility::Inherited;
        }
    }
}

fn rasterize_alignments_in_tile<'a, S, I>(
    dbg_bg_color: [u8; 4],
    tile_bounds: crate::view::View,
    tile_dims: UVec2,
    seq_pairs: S,
) -> Vec<u8>
where
    S: IntoIterator<Item = (SequencePairTile, Aabb, I)>,
    I: IntoIterator<Item = &'a crate::Alignment>,
{
    let px_count = (tile_dims.x * tile_dims.y) as usize;

    let mut buffer = vec![0u8; px_count * 4];
    let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut buffer);
    // pixels.fill(dbg_bg_color);

    let vx_min = tile_bounds.x_min;
    let vx_max = tile_bounds.x_max;

    let bp_per_px = tile_bounds.width() / tile_dims.x as f64;

    let mut mask_buf = vec![0u8; pixels.len()];

    let mut path_commands: Vec<zeno::Command> = Vec::new();
    // let mut line_buf = Vec::new();

    let mut embiggened = 0;
    let mut total = 0;

    let mut total_dst = 0f64;

    let draw_dot = |mask_buf: &mut [u8], x: f32, y: f32, rad: f32| {
        let radplus = rad + 1.0;
        let x_min = (x - radplus).floor() as usize;
        let x_max = (x + radplus).ceil() as usize;
        let y_min = (y - radplus).floor() as usize;
        let y_max = (y + radplus).ceil() as usize;

        let x0 = x;
        let y0 = y;

        let rad_sq = ((rad * rad) - 0.5).max(0.5);

        for x in x_min..x_max {
            for y in y_min..y_max {
                //
                let i = (x as usize) + (y as usize * tile_dims.x as usize);

                let x1 = x as f32 + 0.5;
                let y1 = y as f32 + 0.5;

                let val = (x1 - x0).powi(2) + (y1 - y0).powi(2);
                if i < mask_buf.len() {
                    let px = mask_buf[i];
                    let d = (rad_sq.sqrt() - val.sqrt()).clamp(0.0, 1.0);

                    // NB: explicitly using Ord::max to quiet rust-analyzer
                    // TODO: make an issue
                    let val = Ord::max(px, (d * 255.0) as u8);
                    mask_buf[i] = val;
                }
            }
        }
    };

    // for (seq_pair, [seq_min, seq_max], alignments) in seq_pairs.into_iter() {
    for (seq_pair, seq_aabb, alignments) in seq_pairs.into_iter() {
        // seq_min, seq_max encode the position of the seq. pair tile in the world

        for alignment in alignments {
            let alignment: &crate::Alignment = alignment;
            let loc = &alignment.location;
            let tgt_len = loc.target_range.end - loc.target_range.start;

            let al_min = seq_aabb.mins.x + loc.target_range.start as f64;
            let al_max = al_min + tgt_len as f64;

            let cal_min = vx_min.clamp(al_min, al_max) as u64;
            let cal_max = vx_max.clamp(al_min, al_max) as u64;
            if cal_min == cal_max {
                // println!("no overlap; skipping");
                continue;
            }

            let loc_min = cal_min
                .checked_sub(seq_aabb.mins.x as u64)
                .unwrap_or_default();
            let loc_max = cal_max
                .checked_sub(seq_aabb.mins.x as u64)
                .unwrap_or_default();

            if loc_min == loc_max {
                // println!("no overlap after offset; skipping");
                continue;
            }

            // let t0 = std::time::Instant::now();
            // let mut count = 0;
            // line_buf.clear();
            let iter = alignment.iter_target_range(loc_min..loc_max);
            let mut cmd_iter = CigarScreenPathStrokeIter::new(
                tile_bounds,
                tile_dims,
                [seq_aabb.mins.x, seq_aabb.mins.y],
                iter,
            );

            path_commands.clear();

            while let Some(cmd) = cmd_iter.emit_next() {
                path_commands.push(cmd);
            }

            let first = path_commands.first().clone();
            let last = path_commands.last().clone();

            use zeno::Command as Cmd;
            if let Some((Cmd::MoveTo(start) | Cmd::LineTo(start), Cmd::LineTo(end))) =
                first.zip(last)
            {
                let dist = end.distance_to(*start) as f64;
                total_dst += dist;
                if dist < 1.0 {
                    let x = start.x + 0.5 * (end.x - start.x);
                    let y = start.y + 0.5 * (end.y - start.y);
                    draw_dot(&mut mask_buf, x, y, 1.5);

                    embiggened += 1;
                    total += 1;

                    continue;
                }
                //
            }

            total += 1;
            zeno::Mask::new(&path_commands)
                .size(tile_dims.x, tile_dims.y)
                .style(zeno::Stroke::new(2.0))
                .origin(zeno::Origin::TopLeft)
                .render_into(&mut mask_buf, None);
            // path_commands.extend();
        }
    }

    for (i, (val, px)) in std::iter::zip(mask_buf, pixels).enumerate() {
        let x = i % tile_dims.x as usize;
        // NB: quick hack to avoid vertical line artifacts
        if x == 0 {
            continue;
        }
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

    // println!(
    //     "grew {embiggened} out of {total} alignments\taverage length: {}",
    //     total_dst / embiggened as f64
    // );

    buffer
}

#[derive(Debug, Clone, Reflect, PartialEq, Eq)]
struct RenderParamsOld {
    query_seq_bounds: std::ops::Range<u64>,
    target_seq_bounds: std::ops::Range<u64>,

    canvas_size: UVec2,
}

// run after whatever updates the `AlignmentViewport`
fn update_render_tile_transforms(
    render_grids: Query<(
        &RenderTileGrid,
        &RenderTileGridCanvasSize,
        &AlignmentViewport,
        &Children,
    )>,

    mut tiles: Query<(&mut Transform, &RenderTile)>,
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

            let tgt_r = cg_item.target_seq_range();
            let qry_r = cg_item.query_seq_range();

            let mut w_start = {
                let x = (tgt_r.start as f64) + origin.x;
                let y = (qry_r.start as f64) + origin.y;
                DVec2::new(x, y)
            };
            let mut w_end = {
                let x = (tgt_r.end as f64) + origin.x;
                let y = (qry_r.end as f64) + origin.y;
                DVec2::new(x, y)
            };

            if cg_item.strand() == crate::Strand::Reverse {
                std::mem::swap(&mut w_start.y, &mut w_end.y);
            }

            let s_start = self
                .view
                .map_world_to_screen(screen_dims, [w_start.x, w_start.y]);

            self.last_end = Some(w_end);
            if let Some(_path_0) = self.path_open {
                if matches!(op, I | D) {
                    if len as f64 > self.bp_per_px {
                        self.path_open = None;
                        return Some(Command::LineTo(Vector {
                            x: s_start.x,
                            y: s_start.y,
                        }));
                    }
                }
            } else {
                if matches!(op, M | Eq | X) {
                    self.path_open = Some(w_start);
                    // println!("emitting MoveTo({}, {})", s_start.x, s_start.y);
                    // println!("looped {loop_count} before emitting");
                    return Some(Command::MoveTo(Vector {
                        x: s_start.x,
                        y: s_start.y,
                    }));
                }
            }

            loop_count += 1;
        }
    }
}
