use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::camera::{CameraProjection, ScalingMode},
    utils::tracing::Instrument,
};

use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
    PolylinePlugin,
};
use clap::Parser;
use rustc_hash::FxHashMap;
use wgpu::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

use crate::{
    paf::AlignmentIndex,
    render::{color::PafColorSchemes, exact::CpuViewRasterizerEgui},
    sequences::SeqId,
    PafViewerApp,
};

pub struct PafViewerPlugin;

impl Plugin for PafViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ViewEvent>()
            .add_systems(Startup, setup_base_level_display_image)
            .add_systems(Startup, (setup, prepare_alignments).chain())
            .add_systems(
                Update,
                update_camera.before(bevy::render::camera::camera_system::<OrthographicProjection>),
            )
            .add_systems(
                Update,
                (
                    resize_base_level_image_handle,
                    run_base_level_cpu_rasterizer,
                    update_base_level_image,
                )
                    .chain(),
            );
    }
}

#[derive(Resource)]
pub struct PafViewer {
    pub app: PafViewerApp,
}

#[derive(Resource)]
pub struct AlignmentRasterizer {
    rasterizer: CpuViewRasterizerEgui,
}

#[derive(Resource)]
pub struct AlignmentColorSchemes {
    colors: PafColorSchemes,
}

#[derive(Component, Debug, Clone, Copy)]
struct SequencePairTile {
    target: SeqId,
    query: SeqId,
}

fn update_camera(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut mouse_motion: EventReader<MouseMotion>,

    mut camera_query: Query<(&mut Transform, &mut Projection), With<Camera>>,
    window: Query<&Window>,
) {
    let window = window.single();

    let win_size = Vec2::new(window.resolution.width(), window.resolution.height());

    let scroll_delta = mouse_wheel
        .read()
        .map(|ev| {
            // TODO scale based on ev.unit
            ev.y
        })
        .sum::<f32>();

    let mut mouse_delta = mouse_motion.read().map(|ev| ev.delta).sum::<Vec2>();
    mouse_delta.y *= -1.0;

    for (mut transform, mut proj) in camera_query.iter_mut() {
        let Projection::Orthographic(proj) = proj.as_mut() else {
            continue;
        };

        let xv = proj.area.width() * 0.01;
        let yv = proj.area.height() * 0.01;

        let mut dv = Vec2::ZERO;

        if keyboard.pressed(KeyCode::ArrowLeft) {
            dv.x -= xv;
        }
        if keyboard.pressed(KeyCode::ArrowRight) {
            dv.x += xv;
        }

        if keyboard.pressed(KeyCode::ArrowUp) {
            dv.y += yv;
        }
        if keyboard.pressed(KeyCode::ArrowDown) {
            dv.y -= yv;
        }

        if mouse_button.pressed(MouseButton::Left) {
            dv -= (mouse_delta / win_size) * proj.area.size()
        }

        transform.translation.x += dv.x;
        transform.translation.y += dv.y;

        if scroll_delta.abs() > 0.0 {
            let zoom = if scroll_delta > 0.0 {
                1.0 + scroll_delta * 0.05
            } else {
                1.0 - scroll_delta.abs() * 0.05
            };

            proj.scale *= zoom;
        }
    }
}

fn setup(mut commands: Commands, viewer: Res<PafViewer>) {
    let grid = &viewer.app.alignment_grid;

    let x_len = grid.x_axis.total_len as f32;
    let y_len = grid.y_axis.total_len as f32;

    // prepare camera etc.

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        projection: OrthographicProjection {
            // near: todo!(),
            // far: todo!(),
            // viewport_origin: todo!(),
            // scaling_mode: ScalingMode::AutoMin { min_width: (), min_height: () }
            scale: 100_000.0,
            // area: todo!(),
            ..default()
        }
        .into(),
        // camera: Camera {
        //     // hdr: true,
        //     ..default()
        // },
        ..default()
    });

    // commands.spawn(Camera2dBundle
}

pub fn run(app: PafViewerApp) -> anyhow::Result<()> {
    // let args = crate::cli::Cli::parse();

    let args = crate::cli::Cli::parse();

    let paf_color_schemes = if let Some(path) = args.color_schemes {
        PafColorSchemes::from_paf_like(&app.sequences, &app.alignments, path).unwrap_or_default()
    } else {
        PafColorSchemes::default()
    };

    let mut rasterizer = CpuViewRasterizerEgui::initialize();
    {
        rasterizer
            .tile_cache
            .cache_tile_buffers_for(&paf_color_schemes.default);
        paf_color_schemes.overrides.values().for_each(|colors| {
            rasterizer.tile_cache.cache_tile_buffers_for(colors);
        });
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PolylinePlugin)
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(PafViewer { app })
        .insert_resource(AlignmentRasterizer { rasterizer })
        .insert_resource(AlignmentColorSchemes {
            colors: paf_color_schemes,
        })
        .add_plugins(PafViewerPlugin)
        .run();
    //

    Ok(())
}

fn prepare_alignments(
    mut commands: Commands,
    viewer: Res<PafViewer>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    // create entities for the grid (tiles) & polylines for alignments

    // TODO pull in per-alignment color schemes
    // let op_colors = {
    //     use crate::cigar::CigarOp as Cg;
    //     [(Cg::M, Color::BLACK),
    //      (Cg::Eq, Color::BLACK),
    //      (Cg::X, Color::RED),
    //     ].into_iter().collect::<FxHashMap<_,_>>()
    //     };

    let mut op_mats = FxHashMap::default();
    use crate::cigar::CigarOp as Cg;
    for (op, color) in [
        (Cg::M, Color::BLACK),
        (Cg::Eq, Color::BLACK),
        (Cg::X, Color::RED),
    ] {
        let mat = polyline_materials.add(PolylineMaterial {
            width: 4.0,
            color,
            depth_bias: 0.0,
            perspective: true,
        });
        op_mats.insert(op, mat);
    }

    let grid = &viewer.app.alignment_grid;

    for (pair_id @ &(tgt_id, qry_id), alignments) in viewer.app.alignments.pairs.iter() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        let transform =
            Transform::from_translation(Vec3::new(x_offset as f32, y_offset as f32, 0.0));

        let parent = commands
            .spawn((
                SequencePairTile {
                    target: tgt_id,
                    query: qry_id,
                },
                SpatialBundle {
                    transform,
                    ..default()
                },
            ))
            .with_children(|parent| {
                for (_ix, alignment) in alignments.iter().enumerate() {
                    // bevy_polyline only supports single color polylines,
                    // so to draw cigar ops with different colors, use one
                    // polyline per op

                    let location = &alignment.location;

                    // let align_ix = AlignmentIndex {
                    //     pair: *pair_id,
                    //     index: ix,
                    // };

                    let mut m_verts = Vec::new();
                    let mut eq_verts = Vec::new();
                    let mut x_verts = Vec::new();

                    let mut tgt_cg = 0;
                    let mut qry_cg = 0;

                    for (op, count) in alignment.cigar.whole_cigar() {
                        let tgt_start = tgt_cg;
                        let qry_start = qry_cg;

                        let (tgt_end, qry_end) = match op {
                            Cg::Eq | Cg::X | Cg::M => {
                                tgt_cg += count as u64;
                                qry_cg += count as u64;
                                //
                                (tgt_start + count as u64, qry_start + count as u64)
                            }
                            Cg::I => {
                                qry_cg += count as u64;
                                //
                                (tgt_start, qry_start + count as u64)
                            }
                            Cg::D => {
                                tgt_cg += count as u64;
                                //
                                (tgt_start + count as u64, qry_start)
                            }
                        };

                        let tgt_range = location.map_from_aligned_target_range(tgt_start..tgt_end);
                        let qry_range = location.map_from_aligned_query_range(qry_start..qry_end);

                        let mut from =
                            ultraviolet::DVec2::new(tgt_range.start as f64, qry_range.start as f64);
                        let mut to =
                            ultraviolet::DVec2::new(tgt_range.end as f64, qry_range.end as f64);

                        if location.query_strand.is_rev() {
                            std::mem::swap(&mut from.y, &mut to.y);
                        }

                        let p0 = Vec3::new(from.x as f32, from.y as f32, 0.0);
                        let p1 = Vec3::new(to.x as f32, to.y as f32, 0.0);

                        let buf = match op {
                            Cg::M => &mut m_verts,
                            Cg::Eq => &mut eq_verts,
                            Cg::X => &mut x_verts,
                            _ => continue,
                        };

                        buf.push(p0);
                        buf.push(p1);
                    }

                    for (op, vertices) in [(Cg::M, m_verts), (Cg::Eq, eq_verts), (Cg::X, x_verts)] {
                        let material = op_mats.get(&op).unwrap().clone();
                        let polyline = polylines.add(Polyline { vertices });

                        parent.spawn(PolylineBundle {
                            polyline,
                            material,
                            // transform: transform.clone(),
                            ..default()
                        });
                    }
                }
            });
    }
}

// NB: will probably/maybe replace these events with a component
// and observers or something
#[derive(Debug, Clone, Event)]
struct ViewEvent {
    view: crate::view::View,
}

#[derive(Resource)]
struct LastBaseLevelBuffer {
    view: crate::view::View,
    pixel_buffer: crate::PixelBuffer,
}

#[derive(Resource)]
struct LastBaseLevelHandle {
    // source: Option<LastBaseLevelBuffer>,
    last_view: Option<crate::view::View>,
    last_size: UVec2,
    handle: Handle<Image>,
    // handle: Option<Handle<Image>>,
}

#[derive(Component)]
struct BaseLevelViewUiRoot;

fn setup_base_level_display_image(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window>,
) {
    let window = windows.single();

    let size = Extent3d {
        width: window.physical_width(),
        height: window.physical_height(),
        depth_or_array_layers: 1,
    };

    // let img_handle = Image::new_fill(Extent3d {
    //     width: size[0],
    //     height: size[1],
    //     depth_or_array_layers: 1,
    // }, wgpu::TextureDimension::D2, &[0], wgpu::TextureFormat::, asset_usage)
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            // | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    // fill image.data with zeroes
    image.resize(size);

    let last_size = UVec2::new(window.physical_width(), window.physical_height());

    let img_handle = images.add(image);

    commands.insert_resource(LastBaseLevelHandle {
        last_view: None,
        handle: img_handle.clone(),
        last_size,
    });

    commands.spawn((
        BaseLevelViewUiRoot,
        NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                ..default()
            },
            background_color: Color::WHITE.into(),
            ..default()
        },
        UiImage::new(img_handle),
    ));
}

fn resize_base_level_image_handle(
    // mut commands: Commands,
    mut images: ResMut<Assets<Image>>,

    mut image_handle: ResMut<LastBaseLevelHandle>,

    cameras: Query<&Camera>,
    // window: Query<&Window>,
) {
    // let window = window.single();

    let new_size = cameras.single().physical_target_size().unwrap();

    let need_resize = new_size != image_handle.last_size;

    if need_resize {
        let image = images.get_mut(&image_handle.handle).unwrap();

        let new_extent = Extent3d {
            width: new_size.x,
            height: new_size.y,
            ..default()
        };

        image.resize(new_extent);
        image_handle.last_size = new_size;
    }
}

//
fn run_base_level_cpu_rasterizer(
    mut commands: Commands,

    window: Query<&Window>,
    viewer: Res<PafViewer>,
    color_schemes: Res<AlignmentColorSchemes>,
    mut rasterizer: ResMut<AlignmentRasterizer>,

    mut view_events: EventReader<ViewEvent>,
) {
    let window = window.single();
    // let width = window.resolution.width();
    // let height = window.resolution.height();

    let canvas_size = [window.physical_width(), window.physical_height()];

    let Some(view) = view_events.read().last() else {
        log::info!("no view events for CPU rasterizer");
        return;
    };

    let rasterizer = &mut rasterizer.rasterizer;

    let pixel_buffer = crate::render::exact::draw_alignments_with_color_schemes(
        &rasterizer.tile_cache,
        &color_schemes.colors,
        &viewer.app.sequences,
        &viewer.app.alignment_grid,
        &viewer.app.alignments,
        &view.view,
        canvas_size,
    );

    let Some(pixel_buffer) = pixel_buffer else {
        log::info!("no output for CPU rasterizer");
        return;
    };

    commands.insert_resource(LastBaseLevelBuffer {
        view: view.view,
        pixel_buffer,
    });

    //
}

fn update_base_level_image(
    // mut commands: Commands,

    //
    mut images: ResMut<Assets<Image>>,
    last_buffer: Option<Res<LastBaseLevelBuffer>>,
    image_handle: Res<LastBaseLevelHandle>,
    // ui_root: Query<Entity, With<BaseLevelViewUiRoot>>,
) {
    let Some(last_buffer) = last_buffer else {
        log::info!("no base level image buffer available");
        return;
    };

    let image = images.get_mut(&image_handle.handle).unwrap();

    let pixels: &[u8] = bytemuck::cast_slice(&last_buffer.pixel_buffer.pixels);
    image.data = pixels.to_vec();
    // image.data = bytemuck::

    // let ui_root = ui_root.single();

    // let mut ui_cmds = commands.entity(ui_root);

    /*
    let texture = if let Some(handle) = image_handle.as_ref()

    // ui_cmds.insert(UiImage::new(texture)
    */

    //
}
