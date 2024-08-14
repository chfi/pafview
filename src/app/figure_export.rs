use std::sync::Arc;

use bevy::{prelude::*, utils::HashMap};
use bevy_egui::{EguiContexts, EguiUserTextures};

use crate::paf::AlignmentIndex;

use super::{
    render::{AlignmentDisplayBackImage, AlignmentDisplayImage, AlignmentRenderTarget},
    view::AlignmentViewport,
};

pub struct FigureExportPlugin;

impl Plugin for FigureExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_figure_export_window)
            .add_systems(
                PreUpdate,
                (update_egui_textures, show_figure_export_window)
                    .chain()
                    .after(bevy_egui::EguiSet::BeginFrame),
            );
        //
    }
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
            AlignmentDisplayImage::default(),
            img_handle,
            AlignmentDisplayBackImage {
                image: back_img_handle,
            },
        ))
        .id();

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

// fn swap_egui_textures(

//     mut fig_export: ResMut<FigureExportWindow>,
//     img_query: Query<(
//         &AlignmentDisplayImage,
//         &Handle<Image>,
//         &AlignmentDisplayBackImage,
//     ), (With<FigureExportImage>, )>,
// ) {

// }

fn update_egui_textures(
    mut egui_textures: ResMut<EguiUserTextures>,

    mut fig_export: ResMut<FigureExportWindow>,
    img_query: Query<(
        &AlignmentDisplayImage,
        &Handle<Image>,
        &AlignmentDisplayBackImage,
    )>,
) {
    // the egui_textures field should be set to None when the images change (e.g. due to resize)
    // (and/or maybe this should be redesigned)
    if fig_export.egui_textures.is_some() {
        return;
    }

    let Ok((disp_image, image, back_img)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let front = egui_textures.add_image(image.clone_weak());
    let back = egui_textures.add_image(back_img.image.clone_weak());

    fig_export.egui_textures = Some([front, back]);
}

fn show_figure_export_window(
    mut commands: Commands,
    mut contexts: EguiContexts,

    alignment_viewport: Res<AlignmentViewport>,

    img_query: Query<(&AlignmentDisplayImage, &Handle<Image>)>,
    mut fig_export: ResMut<FigureExportWindow>,
) {
    let Ok((disp_image, image)) = img_query.get(fig_export.display_img) else {
        return;
    };

    let ctx = contexts.ctx_mut();

    egui::Window::new("Figure Exporter")
        .default_width(600.0)
        .resizable(true)
        .show(&ctx, |ui| {
            //

            if let Some([front, _]) = fig_export.egui_textures.as_ref() {
                ui.image((*front, egui::vec2(500.0, 500.0)));
            }

            if ui.button("render current view").clicked() {
                commands
                    .entity(fig_export.display_img)
                    .insert(AlignmentRenderTarget {
                        alignment_view: alignment_viewport.view.clone(),
                        canvas_size: [500, 500].into(),
                        image: image.clone(),
                        is_ready: Arc::new(false.into()),
                    });
            }
        });
}
