use bevy::prelude::*;

use crate::{
    app::{alignments::layout::SeqPairLayout, view::AlignmentViewport},
    render::exact::CpuViewRasterizerEgui,
};

use super::RenderParams;

pub struct BaselevelCigarRenderPlugin;

impl Plugin for BaselevelCigarRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, initialize_rasterizer)
            .add_systems(Startup, setup_base_level_viewer)
            .add_systems(
                PreUpdate,
                (
                    update_base_level_viewer_view,
                    render_base_level_views,
                    set_base_level_viewer_visibility,
                )
                    .chain(),
            );
    }
}

pub const BASE_LEVEL_VIEWER_MIN_BP_PER_PX: f64 = 1.0;

#[derive(Default, Component)]
struct BaselevelViewer {
    view: Option<crate::view::View>,
    last_rendered: Option<RenderParams>,
    // last_update: Option<std::time::Instant>,
}

#[derive(Resource, Deref, DerefMut)]
struct AlignmentRasterizer(CpuViewRasterizerEgui);

fn initialize_rasterizer(
    mut commands: Commands,
    paf_color_schemes: Res<crate::app::AlignmentColorSchemes>,
) {
    let rasterizer = {
        let mut rasterizer = CpuViewRasterizerEgui::initialize();
        rasterizer
            .tile_cache
            .cache_tile_buffers_for(&paf_color_schemes.colors.default);
        paf_color_schemes
            .colors
            .overrides
            .values()
            .for_each(|colors| {
                rasterizer.tile_cache.cache_tile_buffers_for(colors);
            });
        rasterizer
    };

    commands.insert_resource(AlignmentRasterizer(rasterizer));
}

fn setup_base_level_viewer(
    //
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
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

    commands
        .spawn((
            BaselevelViewer::default(),
            SpatialBundle::default(),
            SpriteBundle::default(),
        ))
        .insert(images.add(image));
}

/*
fn resize_base_level_viewer_image(//
    viewers: Query<&Handle<Image>
) {
    //
}
*/

fn update_base_level_viewer_view(
    viewport: Res<AlignmentViewport>,
    mut viewers: Query<&mut BaselevelViewer>,
    // windows: Query<&Window>,
) {
    // TODO maybe take viewer transform into account; doesn't matter for now
    // let Ok(window) = windows.get_single() else { return; };

    // let canvas_size = window.physical_size();
    let view = viewport.view;

    for mut viewer in viewers.iter_mut() {
        viewer.view = Some(view);
    }
}

// hide when zoomed out
fn set_base_level_viewer_visibility(
    //
    mut viewers: Query<(&mut Visibility, &BaselevelViewer)>,
) {
    for (mut visibility, viewer) in viewers.iter_mut() {
        if let Some(params) = viewer.last_rendered {
            let bp_per_px = params.view.width() / params.canvas_size.x as f64;

            if bp_per_px > BASE_LEVEL_VIEWER_MIN_BP_PER_PX {
                *visibility = Visibility::Hidden;
            } else {
                *visibility = Visibility::Visible;
            }
        }
    }
}

fn render_base_level_views(
    //
    rasterizer: Res<AlignmentRasterizer>,
    color_schemes: Res<crate::app::AlignmentColorSchemes>,
    sequences: Res<crate::Sequences>,
    // alignment_grid: Res<crate::AlignmentGrid>,
    alignments: Res<crate::Alignments>,

    mut images: ResMut<Assets<Image>>,

    layouts: Res<Assets<SeqPairLayout>>,
    layout_roots: Query<&Handle<SeqPairLayout>>,

    viewers: Query<(&BaselevelViewer, &Handle<Image>)>,
    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    let canvas_size = window.physical_size();

    for (viewer, viewer_image) in viewers.iter() {
        if let Some(view) = viewer.view {
            let bp_per_px = view.width() / canvas_size.x as f64;

            if bp_per_px > BASE_LEVEL_VIEWER_MIN_BP_PER_PX {
                continue;
            }

            // let layouts_iter = layout_roots
            let layouts_iter = layout_roots.iter().filter_map(|handle| layouts.get(handle));

            let pixel_buffer = crate::render::exact::draw_seq_pair_layouts_with_color_schemes(
                &rasterizer.0.tile_cache,
                &color_schemes.colors,
                &sequences,
                &alignments,
                &view,
                [canvas_size.x, canvas_size.y],
                layouts_iter,
            );

            if let Some(image) = images.get_mut(viewer_image) {
                let pixels: &[u8] = bytemuck::cast_slice(&pixel_buffer.pixels);
                image.data = pixels.to_vec();
            }
        }
    }
}
