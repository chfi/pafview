pub mod alignments;
pub mod annotations;
pub mod gui;
pub mod render;
pub mod rulers;
pub mod selection;
pub mod view;

use bevy::prelude::*;

use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
    PolylinePlugin,
};
use clap::Parser;
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
        app.add_plugins(bevy_egui::EguiPlugin)
            .add_event::<BaseLevelViewEvent>()
            .init_resource::<AlignmentRenderConfig>()
            .add_plugins(gui::MenubarPlugin)
            .add_plugins(view::AlignmentViewPlugin)
            .add_plugins(annotations::AnnotationsPlugin)
            .add_plugins(rulers::ViewerRulersPlugin)
            .add_plugins(selection::RegionSelectionPlugin)
            .add_plugins(render::AlignmentRendererPlugin)
            .add_systems(Startup, setup_base_level_display_image)
            .add_systems(
                Startup,
                (setup, setup_screenspace_camera, prepare_alignments).chain(),
            )
            .add_systems(PreUpdate, config_update_grid_material)
            .add_systems(
                Update,
                (
                    send_base_level_view_events,
                    update_base_level_display_visibility,
                    update_alignment_line_segment_visibility,
                )
                    .after(view::update_camera_from_viewport),
            )
            // .add_systems(PreUpdate, resize_screenspace_camera_target)
            .add_systems(
                Update,
                (
                    resize_base_level_image_handle,
                    run_base_level_cpu_rasterizer,
                    update_base_level_image,
                )
                    .chain()
                    .after(send_base_level_view_events),
            )
            .add_systems(PostUpdate, save_app_config);

        #[cfg(debug_assertions)]
        app.add_plugins(bevy::dev_tools::fps_overlay::FpsOverlayPlugin {
            config: bevy::dev_tools::fps_overlay::FpsOverlayConfig {
                text_config: TextStyle {
                    font_size: 40.0,
                    color: Color::srgb(0.0, 0.0, 0.0),
                    font: default(),
                },
            },
        });
    }
}

// #[derive(Resource)]
// pub struct PafViewer {
//     pub app: PafViewerApp,
// }

#[derive(Resource)]
pub struct AlignmentRasterizer {
    rasterizer: CpuViewRasterizerEgui,
}

#[derive(Resource)]
pub struct AlignmentColorSchemes {
    colors: PafColorSchemes,
}

#[derive(Resource)]
pub struct AlignmentRenderConfig {
    base_level_render_min_bp_per_px: f32,
}

impl std::default::Default for AlignmentRenderConfig {
    fn default() -> Self {
        Self {
            base_level_render_min_bp_per_px: 1.0,
        }
    }
}

#[derive(Component, Debug, Clone, Copy)]
struct SequencePairTile {
    target: SeqId,
    query: SeqId,
}

#[derive(Component, Debug)]
struct Alignment;

#[derive(Component, Debug)]
pub struct AlignmentCamera;

#[derive(Component, Debug)]
pub struct ScreenspaceCamera;

fn setup_screenspace_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    //
    windows: Query<&Window>,
) {
    let window = windows.single();
    let win_size = window.resolution.physical_size();

    let size = Extent3d {
        width: win_size.x,
        height: win_size.y,
        depth_or_array_layers: 1,
    };

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    // fill image.data with zeroes
    image.resize(size);

    let image_handle = images.add(image);

    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                order: 1,
                // target: bevy::render::camera::RenderTarget::Image(image_handle.clone()),
                ..default()
            },
            ..default()
        },
        bevy::render::view::RenderLayers::layer(1),
        ScreenspaceCamera,
    ));
}

#[derive(Resource)]
struct GridMaterial {
    handle: Handle<PolylineMaterial>,
}

fn config_update_grid_material(
    mut materials: ResMut<Assets<PolylineMaterial>>,
    app_config: Res<crate::AppConfig>,
    // viewer: Res<PafViewer>,
    grid_mat: Option<Res<GridMaterial>>,
) {
    let Some(handle) = grid_mat.as_ref().map(|m| &m.handle) else {
        return;
    };

    let Some(mat) = materials.get_mut(handle) else {
        return;
    };

    mat.width = app_config.grid_line_width;
}

fn setup(
    mut commands: Commands,

    alignment_grid: Res<crate::AlignmentGrid>,

    // viewer: Res<PafViewer>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    // NB: initial values don't matter here as the camera will be updated
    // from the AlignmentViewport resource
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
            projection: OrthographicProjection {
                // scale: 1.0,
                scale: 100_000.0,
                // scaling_mode: W
                ..default()
            }
            .into(),
            ..default()
        },
        AlignmentCamera,
    ));

    // create polylines for grid
    let grid = &alignment_grid;

    let mut vertices: Vec<Vec3> = Vec::new();

    for x in grid.x_axis.offsets() {
        let x = x as f32;
        let y0 = 0f32;
        let y1 = grid.y_axis.total_len as f32;

        vertices.push(Vec3::new(x, y0, 0.0));
        vertices.push(Vec3::new(x, y1, 0.0));

        // https://github.com/ForesightMiningSoftwareCorporation/bevy_polyline/issues/20#issuecomment-1035624250
        vertices.push(Vec3::splat(std::f32::INFINITY));
    }

    for y in grid.y_axis.offsets() {
        let y = y as f32;
        let x0 = 0f32;
        let x1 = grid.x_axis.total_len as f32;

        vertices.push(Vec3::new(x0, y, 0.0));
        vertices.push(Vec3::new(x1, y, 0.0));
        vertices.push(Vec3::splat(std::f32::INFINITY));
    }

    let grid_mat = polyline_materials.add(PolylineMaterial {
        width: 1.0,
        color: Color::BLACK.into(),
        depth_bias: 0.0,
        perspective: false,
    });

    commands.insert_resource(GridMaterial {
        handle: grid_mat.clone(),
    });

    let grid = commands.spawn(PolylineBundle {
        polyline: polylines.add(Polyline { vertices }),
        material: grid_mat,
        ..default()
    });
}

pub fn run(app: PafViewerApp) -> anyhow::Result<()> {
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
        .insert_resource(app.app_config)
        .insert_resource(app.sequences)
        .insert_resource(app.alignments)
        .insert_resource(app.alignment_grid)
        // .insert_resource(PafViewer { app })
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
    alignments: Res<crate::Alignments>,
    alignment_grid: Res<crate::AlignmentGrid>,
    color_schemes: Res<AlignmentColorSchemes>,

    mut alignment_materials: ResMut<Assets<render::AlignmentPolylineMaterial>>,
    mut alignment_vertices: ResMut<Assets<render::AlignmentVertices>>,
) {
    let grid = &alignment_grid;

    for (pair_id @ &(tgt_id, qry_id), alignments) in alignments.pairs.iter() {
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
                for (ix, alignment) in alignments.iter().enumerate() {
                    let color_scheme = color_schemes.colors.get(AlignmentIndex {
                        pair: (alignment.target_id, alignment.query_id),
                        index: ix,
                    });

                    let material = render::AlignmentPolylineMaterial::from_alignment(
                        grid,
                        alignment,
                        color_scheme.clone(),
                    );

                    let vertices = render::AlignmentVertices::from_alignment(alignment);

                    let location = &alignment.location;

                    parent.spawn((
                        alignment_materials.add(material),
                        alignment_vertices.add(vertices),
                    ));
                }
            });
    }
}

// NB: will probably/maybe replace these events with a component
// and observers or something
#[derive(Debug, Clone, Event)]
struct BaseLevelViewEvent {
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

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            // format: TextureFormat::Bgra8UnormSrgb,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        ..default()
    };

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
    mut images: ResMut<Assets<Image>>,
    mut image_handle: ResMut<LastBaseLevelHandle>,

    cameras: Query<&Camera, With<AlignmentCamera>>,
) {
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

fn send_base_level_view_events(
    cameras: Query<&Camera, With<AlignmentCamera>>,
    viewport: Res<view::AlignmentViewport>,
    render_config: Res<AlignmentRenderConfig>,
    mut view_events: EventWriter<BaseLevelViewEvent>,
) {
    let camera = cameras.single();
    let size = camera.physical_target_size().unwrap();

    let bp_per_px = {
        let pixels = size.x as f32;
        let bp = viewport.view.width() as f32;
        bp / pixels
    };

    if bp_per_px > render_config.base_level_render_min_bp_per_px {
        return;
    }

    view_events.send(BaseLevelViewEvent {
        view: viewport.view,
    });
}

fn run_base_level_cpu_rasterizer(
    mut commands: Commands,

    cameras: Query<&Camera, With<AlignmentCamera>>,

    sequences: Res<crate::Sequences>,
    alignment_grid: Res<crate::AlignmentGrid>,
    alignments: Res<crate::Alignments>,

    // viewer: Res<PafViewer>,
    color_schemes: Res<AlignmentColorSchemes>,
    mut rasterizer: ResMut<AlignmentRasterizer>,

    mut view_events: EventReader<BaseLevelViewEvent>,
) {
    let camera = cameras.single();

    let size = camera.physical_target_size().unwrap();
    let canvas_size = [size.x, size.y];

    let Some(view) = view_events.read().last() else {
        log::trace!("no view events for CPU rasterizer");
        return;
    };

    let rasterizer = &mut rasterizer.rasterizer;

    let pixel_buffer = crate::render::exact::draw_alignments_with_color_schemes(
        &rasterizer.tile_cache,
        &color_schemes.colors,
        &sequences,
        &alignment_grid,
        &alignments,
        &view.view,
        canvas_size,
    );

    let Some(pixel_buffer) = pixel_buffer else {
        log::trace!("no output for CPU rasterizer");
        return;
    };

    commands.insert_resource(LastBaseLevelBuffer {
        view: view.view,
        pixel_buffer,
    });
}

fn update_base_level_image(
    mut images: ResMut<Assets<Image>>,
    last_buffer: Option<Res<LastBaseLevelBuffer>>,
    image_handle: Res<LastBaseLevelHandle>,
) {
    let Some(last_buffer) = last_buffer else {
        log::trace!("no base level image buffer available");
        return;
    };

    let image = images.get_mut(&image_handle.handle).unwrap();

    let pixels: &[u8] = bytemuck::cast_slice(&last_buffer.pixel_buffer.pixels);
    image.data = pixels.to_vec();
}

fn update_alignment_line_segment_visibility(
    cameras: Query<(&Camera, &Projection), With<AlignmentCamera>>,
    render_config: Res<AlignmentRenderConfig>,
    mut visibility: Query<&mut Visibility, With<Alignment>>,
) {
    let (camera, camera_proj) = cameras.single();

    let size = camera.physical_target_size().unwrap();

    let Projection::Orthographic(proj) = camera_proj else {
        unreachable!();
    };

    let bp_per_px = {
        let pixels = size.x as f32;
        let bp = proj.area.width();
        bp / pixels
    };

    for mut visibility in visibility.iter_mut() {
        if bp_per_px > render_config.base_level_render_min_bp_per_px {
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

fn update_base_level_display_visibility(
    cameras: Query<(&Camera, &Projection), With<AlignmentCamera>>,
    render_config: Res<AlignmentRenderConfig>,
    mut visibility: Query<&mut Visibility, With<BaseLevelViewUiRoot>>,
) {
    let (camera, camera_proj) = cameras.single();

    let size = camera.physical_target_size().unwrap();

    let Projection::Orthographic(proj) = camera_proj else {
        unreachable!();
    };

    let bp_per_px = {
        let pixels = size.x as f32;
        let bp = proj.area.width();
        bp / pixels
    };

    let mut visibility = visibility.single_mut();

    if bp_per_px > render_config.base_level_render_min_bp_per_px {
        *visibility = Visibility::Hidden;
    } else {
        *visibility = Visibility::Visible;
    }
}

fn save_app_config(
    mut exit_events: EventReader<bevy::app::AppExit>,
    app_config: Res<crate::AppConfig>,
) {
    if let Some(_exit) = exit_events.read().last() {
        if let Err(e) = crate::config::save_app_config(&app_config) {
            log::error!("Error saving application settings file: {e:?}");
        }
    }
}
