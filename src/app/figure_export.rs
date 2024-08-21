use std::sync::Arc;

use bevy::{prelude::*, utils::HashMap};
use bevy_egui::{EguiContexts, EguiUserTextures};

use crate::paf::AlignmentIndex;

use super::{
    render::{AlignmentDisplayBackImage, AlignmentRenderTarget, AlignmentViewer},
    view::AlignmentViewport,
};

pub struct FigureExportPlugin;

impl Plugin for FigureExportPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FigureExportWindowOpen>()
            .add_systems(Startup, setup_figure_export_window)
            .add_systems(
                PreUpdate,
                (update_egui_textures, show_figure_export_window)
                    .chain()
                    .after(bevy_egui::EguiSet::BeginFrame),
            )
            .add_systems(
                PreUpdate,
                swap_egui_textures.before(super::render::swap_rendered_alignment_viewer_images),
            );

        // app.add_systems(
        //     Update,
        //     |query: Query<
        //         (Entity, &AlignmentRenderTarget),
        //         (With<FigureExportImage>, Added<AlignmentRenderTarget>),
        //     >| {
        //         for (ent, tgt) in query.iter() {
        //             println!("render target added for {ent:?}:\t{tgt:?}");
        //         }
        //     },
        // );
    }
}

#[derive(Debug, Default, Resource, Reflect)]
pub struct FigureExportWindowOpen {
    pub is_open: bool,
}

// initializes the figure export window object and relevant assets (image for rendering etc.)
fn setup_figure_export_window(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = wgpu::Extent3d {
        width: 512,
        height: 512,
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

    let back_image = image.clone();

    let img_handle = images.add(image);
    let back_img_handle = images.add(back_image);

    let display_img = commands
        .spawn((
            FigureExportImage,
            AlignmentViewer::default().with_bg_color(Color::WHITE),
            img_handle,
            AlignmentDisplayBackImage {
                image: back_img_handle,
            },
        ))
        .id();

    // println!("figure export display sprite: {display_img:?}");

    commands.insert_resource(FigureExportWindow {
        display_img,
        egui_textures: None,
    });
}

#[derive(Component)]
struct FigureExportImage;

#[derive(Resource)]
struct FigureExportWindow {
    display_img: Entity,
    egui_textures: Option<[egui::TextureId; 2]>,
    // front_egui_img: Option<egui::TextureId>,
    // back_egui_img: Option<egui::TextureId>,
}

fn swap_egui_textures(
    mut fig_export: ResMut<FigureExportWindow>,
    // mut egui_textures: ResMut<EguiUserTextures>,
    sprites: Query<(
        // Entity,
        &Handle<Image>,
        &AlignmentDisplayBackImage,
        &AlignmentRenderTarget,
    )>,
) {
    let Ok((front, back, tgt)) = sprites.get(fig_export.display_img) else {
        return;
    };

    if tgt.is_ready.load() {
        if let Some(txs) = fig_export.egui_textures.as_mut() {
            txs.swap(0, 1);
        }
    }
}

fn update_egui_textures(
    mut egui_textures: ResMut<EguiUserTextures>,

    mut fig_export: ResMut<FigureExportWindow>,
    img_query: Query<(&AlignmentViewer, &Handle<Image>, &AlignmentDisplayBackImage)>,
) {
    // the egui_textures field should be set to None when the images change (e.g. due to resize)
    // (and/or maybe this should be redesigned)
    if fig_export.egui_textures.is_some() {
        return;
    }

    let Ok((_disp_image, image, back_img)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let front = egui_textures.add_image(image.clone_weak());
    let back = egui_textures.add_image(back_img.image.clone_weak());

    fig_export.egui_textures = Some([front, back]);
}

fn show_figure_export_window(
    mut commands: Commands,
    mut contexts: EguiContexts,

    mut window_open: ResMut<FigureExportWindowOpen>,
    // mut is_rendering: Local<bool>,
    alignment_viewport: Res<AlignmentViewport>,

    img_query: Query<(Entity, &AlignmentViewer, &AlignmentDisplayBackImage)>,
    mut fig_export: ResMut<FigureExportWindow>,
) {
    let Ok((_disp_ent, _disp_image, back_image)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let image = &back_image.image;

    // let is_rendering = rendering_query.get(disp_ent).is_ok();
    let is_rendering = false;

    let ctx = contexts.ctx_mut();

    egui::Window::new("Figure Exporter")
        .default_width(600.0)
        .resizable(true)
        .open(&mut window_open.is_open)
        .show(&ctx, |ui| {
            //

            if let Some([front, _]) = fig_export.egui_textures.as_ref() {
                ui.image((*front, egui::vec2(500.0, 500.0)));
            }

            let mut button = |text: &str| ui.add_enabled(!is_rendering, egui::Button::new(text));

            if button("render current view").clicked() {
                commands
                    .entity(fig_export.display_img)
                    .insert(AlignmentRenderTarget {
                        alignment_view: alignment_viewport.view.clone(),
                        canvas_size: [500, 500].into(),
                        image: image.clone(),
                        is_ready: Arc::new(false.into()),
                    });
            }

            if button("render initial view").clicked() {
                let view = &alignment_viewport.initial_view;

                let new_view = alignment_viewport
                    .view
                    .fit_ranges_in_view_f64(Some(view.x_range()), Some(view.y_range()));

                commands
                    .entity(fig_export.display_img)
                    .insert(AlignmentRenderTarget {
                        alignment_view: new_view,
                        canvas_size: [500, 500].into(),
                        image: image.clone(),
                        is_ready: Arc::new(false.into()),
                    });
            }
        });
}
