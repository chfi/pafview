pub mod alignments;
pub mod annotations;
pub mod assets;
// pub mod figure_export;
pub mod gui;
pub mod infobar;
pub mod input;
pub mod picking;
pub mod render;
pub mod rulers;
// pub mod selection;
pub mod paf_window;
pub mod svg_export;
pub mod view;

pub use alignments::{AlignmentIndex, SequencePairTile};

use bevy::{prelude::*, render::view::RenderLayers};

use bevy_polyline::PolylinePlugin;
use clap::Parser;

use crate::{
    render::{color::PafColorSchemes, exact::CpuViewRasterizerEgui},
    PafViewerApp,
};

pub struct PafViewerPlugin;

impl Plugin for PafViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_egui::EguiPlugin)
            // .add_plugins(bevy_inspector_egui::quick::WorldInspectorPlugin::default())
            .add_plugins(avian2d::PhysicsPlugins::default().with_length_unit(100.0))
            .add_plugins(avian2d::debug_render::PhysicsDebugPlugin::default())
            .insert_gizmo_config(
                avian2d::prelude::PhysicsGizmos {
                    aabb_color: Some(Color::linear_rgb(1.0, 0.1, 0.3)),
                    collider_color: Some(Color::linear_rgb(1.0, 0.1, 0.3)),
                    ..default()
                },
                GizmoConfig {
                    render_layers: RenderLayers::layer(1),
                    ..default()
                },
            )
            .insert_resource(avian2d::prelude::Gravity(bevy::math::DVec2::ZERO))
            .add_plugins(assets::ViewerAssetsPlugin)
            .add_plugins(input::InputPlugin)
            .add_plugins(alignments::AlignmentsPlugin)
            .add_plugins(gui::MenubarPlugin)
            .add_plugins(infobar::InfobarPlugin)
            .add_plugins(view::AlignmentViewPlugin)
            .add_plugins(annotations::AnnotationsPlugin)
            .add_plugins(rulers::ViewerRulersPlugin)
            // .add_plugins(selection::RegionSelectionPlugin)
            .add_plugins(picking::PickingPlugin)
            // .add_plugins(figure_export::FigureExportPlugin)
            .add_plugins(render::bordered_rect::BorderedRectRenderPlugin)
            .add_plugins(paf_window::PafListWindowPlugin)
            .add_systems(Startup, setup_cameras)
            .add_systems(PreUpdate, update_screenspace_camera)
            .add_systems(Last, save_app_config);

        app.add_plugins(svg_export::SvgExportPlugin)
            .add_plugins(crate::toast::ToastMessagePlugin);

        app.add_plugins(render::sampled_lines::SampledAlignmentRendererPlugin)
            .add_plugins(render::base_level::BaselevelCigarRenderPlugin);

        // #[cfg(feature = "tracy")]
        // {
        //     app.sub_app_mut(bevy::render::RenderApp).add_systems(
        //         bevy::render::Render,
        //         (|| {
        //             tracing_tracy::client::frame_mark();
        //         })
        //         .after(bevy::render::renderer::render_system),
        //     );
        // }
        #[cfg(feature = "renderdoc")]
        {
            use renderdoc::{RenderDoc as RenderDocApi, V141};
            #[derive(Resource, Deref)]
            struct RenderDoc(RenderDocApi<V141>);

            let rd = RenderDoc(RenderDocApi::new().unwrap());

            app.sub_app_mut(bevy::render::RenderApp)
                .insert_resource(rd)
                .add_systems(
                    bevy::render::Render,
                    (|rd: Res<RenderDoc>,
                      mut capturing: Local<Bool>,
                      keys: Res<ButtonInput<KeyCode>>| {
                        if keys.just_pressed(KeyCode::F12) {
                            *capturing = !*capturing;
                            if *capturing {
                                println!("renderdoc capture enabled");
                            } else {
                                println!("renderdoc capture disabled");
                            }
                        }

                        if *capturing {
                            rd.trigger_capture();
                        }
                    })
                    .after(bevy::render::renderer::render_system),
                )
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

#[derive(Component, Debug)]
pub struct AlignmentCamera;

#[derive(Component, Debug)]
pub struct ScreenspaceCamera;

fn setup_cameras(mut commands: Commands) {
    // NB: initial values don't matter here as the camera will be updated
    // from the AlignmentViewport resource
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
            projection: OrthographicProjection {
                scale: 100_000.0,
                near: -1000.0,
                far: 1000.0,
                ..default()
            }
            .into(),
            ..default()
        },
        AlignmentCamera,
    ));

    commands.spawn((
        Camera2dBundle {
            camera: Camera {
                order: 1,
                ..default()
            },
            ..default()
        },
        bevy::render::view::RenderLayers::layer(1),
        ScreenspaceCamera,
        IsDefaultUiCamera,
    ));
}

fn update_screenspace_camera(
    mut camera: Query<(&mut Transform, &Camera), With<ScreenspaceCamera>>,
) {
    for (mut transform, camera) in camera.iter_mut() {
        if let Some(size) = camera.logical_target_size() {
            transform.translation.x = size.x * 0.5;
            transform.translation.y = size.y * 0.5;
        }
    }
}

#[derive(Resource)]
pub struct AppForegroundColor(pub Color);
#[derive(Resource)]
pub struct AppBackgroundColor(pub Color);

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

    let (fg_color, bg_color) = if args.dark_mode {
        (Color::WHITE, Color::BLACK)
    } else {
        (Color::BLACK, Color::WHITE)
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
        .insert_resource(AppForegroundColor(fg_color))
        .insert_resource(AppBackgroundColor(bg_color))
        .insert_resource(ClearColor(bg_color))
        .insert_resource(app.app_config)
        .insert_resource(app.sequences)
        .insert_resource(app.alignments)
        .insert_resource(paf_color_schemes)
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
