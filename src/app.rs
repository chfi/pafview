pub mod alignments;
pub mod annotations;
pub mod figure_export;
pub mod gui;
pub mod infobar;
pub mod picking;
pub mod render;
pub mod rulers;
pub mod selection;
pub mod view;

pub use alignments::{AlignmentIndex, SequencePairTile};

use bevy::prelude::*;

use bevy_polyline::{material::PolylineMaterial, PolylinePlugin};
use clap::Parser;
use wgpu::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages};

use crate::{
    render::{color::PafColorSchemes, exact::CpuViewRasterizerEgui},
    PafViewerApp,
};

pub struct PafViewerPlugin;

impl Plugin for PafViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_egui::EguiPlugin)
            // .add_plugins(bevy_inspector_egui::quick::WorldInspectorPlugin::default())
            .init_resource::<AlignmentRenderConfig>()
            .add_plugins(alignments::AlignmentsPlugin)
            .add_plugins(gui::MenubarPlugin)
            .add_plugins(infobar::InfobarPlugin)
            .add_plugins(view::AlignmentViewPlugin)
            .add_plugins(annotations::AnnotationsPlugin)
            .add_plugins(rulers::ViewerRulersPlugin)
            .add_plugins(selection::RegionSelectionPlugin)
            .add_plugins(picking::PickingPlugin)
            // .add_plugins(figure_export::FigureExportPlugin)
            .add_plugins(render::bordered_rect::BorderedRectRenderPlugin)
            .add_systems(Startup, (setup, setup_screenspace_camera).chain())
            .add_systems(Last, save_app_config);

        // TODO: create a plugin that combines & manages all the render plugins

        app.add_plugins(render::gpu_lines::AlignmentRendererPlugin)
            .add_plugins(render::base_level::BaselevelCigarRenderPlugin);

        // NB: these should all be replaced or are otherwise vestigial
        // app.add_systems(PreUpdate, config_update_grid_material)
        //     .add_systems(Startup, setup_base_level_display_image)
        //     .add_systems(
        //         Update,
        //         (
        //             send_base_level_view_events,
        //             update_base_level_display_visibility,
        //         )
        //             .after(view::update_camera_from_viewport),
        //     )
        // .add_systems(PreUpdate, resize_screenspace_camera_target)
        // .add_systems(
        //     Update,
        //     (
        //         resize_base_level_image_handle,
        //         run_base_level_cpu_rasterizer,
        //         update_base_level_image,
        //     )
        //         .chain()
        //         .after(send_base_level_view_events),
        // );

        let args = crate::cli::Cli::parse();

        if args.low_mem {
            app.add_plugins(render::cigar_sampling::CigarSamplingRenderPlugin);
        }

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
pub struct AlignmentColorSchemes {
    pub colors: PafColorSchemes,
}

impl AlignmentColorSchemes {
    pub(crate) fn get(
        &self,
        alignment: &alignments::AlignmentIndex,
    ) -> &crate::render::color::AlignmentColorScheme {
        self.colors.get(alignment)
    }
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

fn setup(mut commands: Commands) {
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
}

#[derive(Resource)]
pub struct ForegroundColor(pub Color);

pub fn run(app: PafViewerApp) -> anyhow::Result<()> {
    let args = crate::cli::Cli::parse();

    let mut paf_color_schemes = if args.dark_mode {
        PafColorSchemes::dark_mode()
    } else {
        PafColorSchemes::default()
    };

    if let Some(path) = args.color_schemes.as_ref() {
        paf_color_schemes
            .fill_from_paf_like(&app.sequences, &app.alignments, path)
            .unwrap_or_default()
    }

    let mut rasterizer = CpuViewRasterizerEgui::initialize();
    {
        rasterizer
            .tile_cache
            .cache_tile_buffers_for(&paf_color_schemes.default);
        paf_color_schemes.overrides.values().for_each(|colors| {
            rasterizer.tile_cache.cache_tile_buffers_for(colors);
        });
    }

    let paf_opt_fields = crate::paf::PafMetadata::from_paf(&app.sequences, &args.paf);

    let mut viewer_app = App::new();

    let paf_file_name = args
        .paf
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    let window_title = format!("pafview - {paf_file_name}");

    let (foreground_color, clear_color) = if args.dark_mode {
        (ForegroundColor(Color::WHITE), ClearColor(Color::BLACK))
    } else {
        (ForegroundColor(Color::BLACK), ClearColor(Color::WHITE))
    };

    viewer_app
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: window_title,
                ..default()
            }),
            exit_condition: bevy::window::ExitCondition::OnAllClosed,
            ..default()
        }))
        .add_plugins(PolylinePlugin)
        .insert_resource(args)
        .insert_resource(clear_color)
        .insert_resource(foreground_color)
        .insert_resource(app.app_config)
        .insert_resource(app.sequences)
        .insert_resource(app.alignments)
        .insert_resource(app.alignment_grid)
        .insert_resource(AlignmentColorSchemes {
            colors: paf_color_schemes,
        })
        .add_plugins(PafViewerPlugin);

    if let Ok(opt_fields) = paf_opt_fields {
        viewer_app.insert_resource(opt_fields);
    }

    viewer_app.run();

    Ok(())
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
