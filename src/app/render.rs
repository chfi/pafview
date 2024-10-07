use std::sync::Arc;

use bevy::{
    ecs::system::{lifetimeless::SRes, EntityCommands},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            CachedRenderPipelineId, PipelineCache, RenderPipelineDescriptor, ShaderType,
            UniformBuffer,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
        view::RenderLayers,
        Extract, Render, RenderApp, RenderSet,
    },
    utils::HashMap,
};
use wgpu::{util::BufferInitDescriptor, ColorWrites, ShaderStages, VertexStepMode};

use crate::{
    math_conv::{ConvertFloat32, ConvertVec2},
    render::color::AlignmentColorScheme,
    CigarOp,
};

use super::view::AlignmentViewport;

mod base_level;
pub(crate) mod cigar_sampling;
pub mod layout;

/*

rendering is done by creating a sprite with the
`AlignmentDisplayImage` component, which then can be given the
`AlignmentRenderTarget` component, with a given alignment view, to
trigger a render (the GPU line renderer or CPU base-level rasterizer
will be used depending on the view scale)

the sprite can also be given a map of alignment position overrides;
if present, only the alignments with overrides will be rendered to
the texture used by the sprite

the plugin setup creates an alignment display sprite that is rendered
to the screenspace camera (`RenderLayer` 1) and updated based on the
`AlignmentViewport` resource



*/

pub struct AlignmentRendererPlugin;

impl Plugin for AlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<AlignmentVertices>()
            .add_plugins(RenderAssetPlugin::<GpuAlignmentVertices>::default())
            .add_plugins(ExtractComponentPlugin::<Handle<AlignmentVertices>>::default())
            .init_asset::<AlignmentPolylineMaterial>()
            .add_plugins(RenderAssetPlugin::<GpuAlignmentPolylineMaterial>::default())
            .add_plugins(ExtractComponentPlugin::<Handle<AlignmentPolylineMaterial>>::default())
            .init_asset::<AlignmentLayoutMaterials>()
            .add_plugins(RenderAssetPlugin::<AlignmentLayoutMaterials>::default())
            .add_plugins(ExtractComponentPlugin::<Handle<AlignmentLayoutMaterials>>::default())
            .add_plugins(ExtractComponentPlugin::<LineOnlyAlignment>::default())
            .insert_resource(AlignmentShaderConfig {
                line_width: 8.0,
                brightness: 1.0,
            })
            .init_resource::<AlignmentVerticesIndex>()
            .add_plugins(ExtractComponentPlugin::<AlignmentViewer>::default())
            .add_plugins(ExtractComponentPlugin::<AlignmentViewerImages>::default())
            .add_plugins(ExtractComponentPlugin::<AlignmentRenderTarget>::default())
            .add_plugins(ExtractResourcePlugin::<AlignmentShaderConfig>::default())
            .add_systems(
                Startup,
                prepare_alignment_grid_layout_materials.after(super::prepare_alignments),
            )
            .add_systems(
                PreUpdate,
                (
                    update_alignment_shader_config,
                    swap_rendered_alignment_viewer_images,
                    (
                        resize_alignment_viewer_back_image,
                        update_swapped_viewer_sprite,
                        update_main_alignment_viewer_sprite_transform, // MainAlignmentView only
                    )
                        .after(swap_rendered_alignment_viewer_images),
                ),
            )
            .add_systems(
                Startup,
                setup_main_alignment_viewer.after(prepare_alignment_grid_layout_materials),
            )
            .add_systems(
                PreUpdate,
                (
                    update_main_alignment_image_view,
                    set_main_viewer_image_size.before(resize_alignment_viewer_back_image),
                ),
            )
            .add_systems(
                Update,
                trigger_alignment_viewer_line_render
                    .after(super::view::update_camera_from_viewport),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<AlignmentPolylinePipeline>()
            .add_systems(
                Render,
                (init_display_image_vertex_bind_group, update_vertex_uniforms)
                    .chain()
                    .in_set(RenderSet::Prepare),
            )
            // .add_systems(ExtractSchedule, extract_alignment_position_overrides)
            .add_systems(ExtractSchedule, extract_alignment_viewer_children)
            .add_systems(Render, queue_draw_alignment_lines.in_set(RenderSet::Render))
            // .add_systems(
            //     Render,
            //     (queue_alignment_draw, queue_alignment_draw_overrides).in_set(RenderSet::Render),
            // )
            .add_systems(Render, cleanup_finished_renders.in_set(RenderSet::Cleanup));
    }
}

#[derive(Debug, Component, ExtractComponent, Clone)]
pub struct AlignmentViewer {
    pub next_view: Option<crate::view::View>,
    rendered_view: Option<crate::view::View>,
    last_render_time: Option<std::time::Instant>,

    pub image_size: UVec2,
    pub background_color: Color,
}

impl AlignmentViewer {
    pub fn clear_rendered(&mut self) {
        self.rendered_view = None;
        self.last_render_time = None;
    }
}

#[derive(Component)]
struct ExtractedViewerChildren(Vec<Entity>);

fn extract_alignment_viewer_children(
    mut commands: Commands,
    viewers: Extract<Query<(Entity, &Children), With<AlignmentViewer>>>,
    children: Extract<Query<&Handle<AlignmentLayoutMaterials>>>,
) {
    for (viewer_entity, viewer_children) in viewers.iter() {
        let child_handles: Vec<Entity> = viewer_children
            .iter()
            .filter_map(|&child_entity| children.get(child_entity).ok().map(|_| child_entity))
            .collect();

        if !child_handles.is_empty() {
            commands
                .insert_or_spawn_batch([(viewer_entity, ExtractedViewerChildren(child_handles))]);
        }
    }
}

// NB: for now the shader config is shared by all alignment display images
fn update_alignment_shader_config(
    app_config: Res<crate::config::AppConfig>,
    mut shader_config: ResMut<AlignmentShaderConfig>,
) {
    let cfg = &app_config;

    if cfg.alignment_line_width != shader_config.line_width {
        shader_config.line_width = cfg.alignment_line_width;
    }
}

// marker for alignment images that are linked to the main viewport
#[derive(Debug, Component)]
pub struct MainAlignmentView;

#[derive(Resource)]
pub struct AlignmentGridLayoutMaterials {
    pub line_only: Handle<AlignmentLayoutMaterials>,
    pub with_base_level: Handle<AlignmentLayoutMaterials>,
}

pub(crate) fn prepare_alignment_grid_layout_materials(
    mut commands: Commands,
    mut alignment_mats: ResMut<Assets<AlignmentPolylineMaterial>>,
    mut layout_mats: ResMut<Assets<AlignmentLayoutMaterials>>,

    alignments: Res<crate::Alignments>,
    grid: Res<crate::AlignmentGrid>,

    color_schemes: Res<super::AlignmentColorSchemes>,
    vertex_index: Res<AlignmentVerticesIndex>,
) {
    let mut line_only_pos = Vec::new();
    let mut with_base_level_pos = Vec::new();

    for ((target_id, query_id), aligns) in alignments.pairs() {
        let x_range = grid.x_axis.sequence_axis_range(target_id);
        let y_range = grid.y_axis.sequence_axis_range(query_id);

        let Some((x_range, y_range)) = x_range.zip(y_range) else {
            continue;
        };

        let x0 = x_range.start as f64;
        let y0 = y_range.start as f64;
        let pos = Vec2::new(x0 as f32, y0 as f32);

        for (ix, alignment) in aligns.enumerate() {
            let index = super::alignments::AlignmentIndex {
                target: target_id,
                query: query_id,
                pair_index: ix,
            };

            if alignment.cigar.is_empty() {
                line_only_pos.push((index, pos));
            } else {
                with_base_level_pos.push((index, pos));
            }
        }
    }

    let line_only = layout_mats.add(AlignmentLayoutMaterials::from_positions_iter(
        &mut alignment_mats,
        &vertex_index,
        &color_schemes,
        line_only_pos,
    ));

    let with_base_level = layout_mats.add(AlignmentLayoutMaterials::from_positions_iter(
        &mut alignment_mats,
        &vertex_index,
        &color_schemes,
        with_base_level_pos,
    ));

    commands.insert_resource(AlignmentGridLayoutMaterials {
        line_only,
        with_base_level,
    });
}

// spawns an entity with everything set up to render an alignment view
// to an image, using the "global" alignment grid layout.
//
pub(crate) fn spawn_alignment_viewer_grid_layout<'a>(
    commands: &'a mut Commands,
    images: &mut Assets<Image>,
    grid_layout: &AlignmentGridLayoutMaterials,
) -> EntityCommands<'a> {
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
    let front_image = image;
    let back_image = front_image.clone();

    let front = images.add(front_image);
    let back = images.add(back_image);

    let mut depth_buffer = Image {
        texture_descriptor: wgpu::TextureDescriptor {
            label: None,
            size,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth16Unorm,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };

    depth_buffer.resize(size);
    let front_depth = depth_buffer;
    let back_depth = front_depth.clone();

    let front_depth = images.add(front_depth);
    let back_depth = images.add(back_depth);

    let mut display_sprite = commands.spawn((
        AlignmentViewer::default(),
        AlignmentViewerImages {
            front,
            back,

            front_depth,
            back_depth,
        },
    ));

    display_sprite.with_children(|parent| {
        parent.spawn(grid_layout.with_base_level.clone());
        parent.spawn((grid_layout.line_only.clone(), LineOnlyAlignment));
    });
    //
    display_sprite
}

pub(crate) fn swap_rendered_alignment_viewer_images(
    mut commands: Commands,

    mut viewers: Query<(
        Entity,
        &AlignmentRenderTarget,
        &mut AlignmentViewer,
        &mut AlignmentViewerImages,
    )>,
) {
    for (entity, render_target, mut viewer, mut images) in viewers.iter_mut() {
        if !render_target.is_ready.load() {
            continue;
        }

        viewer.rendered_view = Some(render_target.alignment_view);
        viewer.last_render_time = Some(std::time::Instant::now());

        let images = images.as_mut();
        let front = &mut images.front;
        let back = &mut images.back;
        std::mem::swap(front, back);

        commands.entity(entity).remove::<AlignmentRenderTarget>();
    }
}

fn update_swapped_viewer_sprite(
    mut sprites: Query<
        (&mut Handle<Image>, &AlignmentViewerImages),
        Changed<AlignmentViewerImages>,
    >,
) {
    for (mut img, images) in sprites.iter_mut() {
        *img = images.front.clone();
    }
}

fn setup_main_alignment_viewer(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,

    grid_layout: Res<AlignmentGridLayoutMaterials>,
) {
    use bevy_mod_picking::prelude::*;

    let mut viewer = spawn_alignment_viewer_grid_layout(&mut commands, &mut images, &grid_layout);

    viewer.insert((
        MainAlignmentView,
        Pickable {
            should_block_lower: false,
            is_hoverable: false,
        },
        RenderLayers::layer(1),
        SpriteBundle {
            sprite: Sprite {
                color: Color::WHITE,
                ..default()
            },
            // texture: img_handle.clone(),
            transform: Transform::IDENTITY,
            ..default()
        },
    ));
}

fn update_main_alignment_image_view(
    alignment_viewport: Res<AlignmentViewport>,
    mut sprites: Query<&mut AlignmentViewer, With<MainAlignmentView>>,
) {
    for mut display_img in sprites.iter_mut() {
        display_img.next_view = Some(alignment_viewport.view);
    }
}

fn update_main_alignment_viewer_sprite_transform(
    images: Res<Assets<Image>>,
    windows: Query<&Window>,
    mut display_sprites: Query<
        (
            &mut Transform,
            &mut Sprite,
            &Handle<Image>,
            &AlignmentViewer,
        ),
        With<MainAlignmentView>,
    >,
) {
    let window = windows.single();
    let win_size = window.resolution.size();
    let dpi_scale = window.resolution.scale_factor();

    for (mut transform, mut sprite, img_handle, display_img) in display_sprites.iter_mut() {
        let Some((next_view, last_view)) = display_img.next_view.zip(display_img.rendered_view)
        else {
            continue;
        };

        // resize sprite to take display scale into account
        if let Some(img) = images.get(img_handle) {
            sprite.custom_size = Some(img.size_f32() / dpi_scale);
        }

        let old_mid = last_view.center();
        if last_view == next_view {
            *transform = Transform::IDENTITY;
        } else {
            let new_mid = next_view.center();

            let world_delta = new_mid - old_mid;
            let norm_delta = world_delta / next_view.size();

            let w_rat = last_view.width() / next_view.width();
            let h_rat = last_view.height() / next_view.height();

            let screen_delta = norm_delta.to_f32() * [win_size.x, win_size.y].as_uv();

            *transform =
                Transform::from_translation(Vec3::new(-screen_delta.x, -screen_delta.y, 0.0))
                    .with_scale(Vec3::new(w_rat as f32, h_rat as f32, 1.0));
        }
    }
}

#[derive(Debug, Component)]
pub struct AlignmentColor {
    pub color_scheme: AlignmentColorScheme,
}

impl Default for AlignmentViewer {
    fn default() -> Self {
        Self {
            next_view: None,
            rendered_view: None,
            last_render_time: None,
            background_color: Color::NONE,
            image_size: [512, 512].into(),
        }
    }
}

impl AlignmentViewer {
    pub fn with_bg_color(&self, color: Color) -> Self {
        Self {
            next_view: self.next_view,
            background_color: color,
            ..default()
        }
    }
}

#[derive(Component, ExtractComponent, Clone)]
pub struct AlignmentViewerImages {
    pub front: Handle<Image>,
    pub back: Handle<Image>,

    pub front_depth: Handle<Image>,
    pub back_depth: Handle<Image>,
}

#[derive(Debug, Clone, Component, ExtractComponent)]
pub struct AlignmentRenderTarget {
    pub alignment_view: crate::view::View,
    pub canvas_size: UVec2,

    pub is_ready: Arc<crossbeam::atomic::AtomicCell<bool>>,
}

#[derive(Component)]
pub(super) struct Rendering;

#[derive(Component)]
struct VertexBindGroup {
    group_0: BindGroup,

    proj_buffer: UniformBuffer<Mat4>,
    config_buffer: UniformBuffer<GpuAlignmentRenderConfig>,
}

fn init_display_image_vertex_bind_group(
    mut commands: Commands,

    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline: Res<AlignmentPolylinePipeline>,

    sprites: Query<Entity, Added<AlignmentViewer>>,
) {
    for entity in sprites.iter() {
        let mut proj_buffer: UniformBuffer<_> = Mat4::IDENTITY.into();
        let mut config_buffer: UniformBuffer<_> = GpuAlignmentRenderConfig {
            line_width: 8.0,
            brightness: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
        .into();

        proj_buffer.write_buffer(&render_device, &render_queue);
        config_buffer.write_buffer(&render_device, &render_queue);

        let group_0 = render_device.create_bind_group(
            Some("AlignmentRenderConfig"),
            &pipeline.proj_config_layout,
            &BindGroupEntries::sequential((
                proj_buffer.binding().unwrap(),
                config_buffer.binding().unwrap(),
            )),
        );

        commands.entity(entity).insert(VertexBindGroup {
            group_0,
            proj_buffer,
            config_buffer,
        });
    }
}

fn update_vertex_uniforms(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,

    shader_config: Res<AlignmentShaderConfig>,

    mut targets: Query<(&AlignmentRenderTarget, &mut VertexBindGroup), Without<Rendering>>,
) {
    for (target, mut uniforms) in targets.iter_mut() {
        let proj_uv = target.alignment_view.to_mat4();
        let proj = Mat4::from_cols_array(proj_uv.as_array());
        uniforms.proj_buffer.set(proj);
        uniforms
            .proj_buffer
            .write_buffer(&render_device, &render_queue);

        // if config.is_changed() {
        let mut cfg = uniforms.config_buffer.get().clone();
        cfg.line_width = shader_config.line_width / target.canvas_size.x as f32;
        cfg.brightness = shader_config.brightness;
        uniforms.config_buffer.set(cfg);
        uniforms
            .config_buffer
            .write_buffer(&render_device, &render_queue);
        // }
    }
}

fn resize_alignment_viewer_back_image(
    mut images: ResMut<Assets<Image>>,

    viewers: Query<(&AlignmentViewer, &AlignmentViewerImages), Changed<AlignmentViewer>>,
) {
    for (viewer, viewer_imgs) in viewers.iter() {
        for img in [&viewer_imgs.back, &viewer_imgs.back_depth] {
            if let Some(img) = images.get_mut(img) {
                let set_size = viewer.image_size;
                if img.size() != set_size {
                    img.resize(wgpu::Extent3d {
                        width: set_size.x,
                        height: set_size.y,
                        depth_or_array_layers: 1,
                    });
                }
            }
        }
    }
}

fn set_main_viewer_image_size(
    windows: Query<&Window>,
    mut viewers: Query<&mut AlignmentViewer, With<MainAlignmentView>>,
) {
    let window = windows.single();

    for mut viewer in viewers.iter_mut() {
        viewer.image_size = window.resolution.physical_size();
    }
}

fn trigger_alignment_viewer_line_render(
    mut commands: Commands,

    frame_count: Res<bevy::core::FrameCount>,
    shader_config: Res<AlignmentShaderConfig>,

    viewers: Query<
        (Entity, &AlignmentViewer, &AlignmentViewerImages),
        Without<AlignmentRenderTarget>,
    >,
) {
    if frame_count.0 < 3 {
        return;
    }

    for (viewer_ent, viewer, viewer_imgs) in viewers.iter() {
        // this could be handled with a timer...
        // & the delay shouldn't be built-in to every viewer (though it shouldn't matter)
        if let Some(ms_since_last_render) = viewer.last_render_time.map(|t| t.elapsed().as_millis())
        {
            if ms_since_last_render < 10 {
                return;
            }
        }

        if let Some(next_view) = viewer.next_view {
            let changed_view = viewer.rendered_view != Some(next_view);
            if changed_view || shader_config.is_changed() {
                commands.entity(viewer_ent).insert(AlignmentRenderTarget {
                    alignment_view: next_view,
                    canvas_size: viewer.image_size,
                    is_ready: Arc::new(false.into()),
                });
            }
        }
    }
}

#[derive(Clone, Resource)]
pub struct AlignmentPolylinePipeline {
    pub proj_config_layout: BindGroupLayout,
    pub color_scheme_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,

    pub pipeline: CachedRenderPipelineId,

    pub shader: Handle<Shader>,
}

impl FromWorld for AlignmentPolylinePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        use bevy::render::render_resource::{self, binding_types};

        let proj_config_layout = render_device.create_bind_group_layout(
            "AlignmentRenderConfig",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (
                    binding_types::uniform_buffer::<Mat4>(false),
                    binding_types::uniform_buffer::<GpuAlignmentRenderConfig>(false),
                ),
            ),
        );

        let color_scheme_layout = render_device.create_bind_group_layout(
            "AlignmentColorScheme",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (binding_types::uniform_buffer::<GpuAlignmentColorScheme>(
                    false,
                ),),
            ),
        );

        let model_layout = render_device.create_bind_group_layout(
            "AlignmentColorScheme",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (binding_types::uniform_buffer::<Mat4>(false),),
            ),
        );

        let shader = Shader::from_wgsl(
            include_str!("../../assets/shaders/lines_color_scheme.wgsl"),
            "internal/shaders/lines_color_scheme.wgsl",
        );
        let shader = world.resource::<AssetServer>().add(shader);
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("Alignment Render Pipeline".into()),
            layout: vec![
                proj_config_layout.clone(),
                color_scheme_layout.clone(),
                model_layout.clone(),
            ],
            push_constant_ranges: vec![],
            vertex: render_resource::VertexState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: "vs_main".into(),
                buffers: vec![render_resource::VertexBufferLayout {
                    array_stride: 5 * std::mem::size_of::<u32>() as u64,
                    step_mode: VertexStepMode::Instance,
                    attributes: vec![
                        render_resource::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        render_resource::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        render_resource::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            fragment: Some(render_resource::FragmentState {
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: "fs_main".into(),
                targets: vec![Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: render_resource::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth16Unorm,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..default()
            },
        });

        Self {
            proj_config_layout,
            color_scheme_layout,
            model_layout,
            pipeline,
            shader,
        }
    }
}

struct AlignmentPolylineBindGroups {
    // group_0: BindGroup,
    group_1: BindGroup,
    group_2: BindGroup,
}

#[derive(Component)]
#[allow(dead_code)]
struct GpuAlignmentPolylineMaterial {
    color_scheme_buffer: UniformBuffer<GpuAlignmentColorScheme>,
    model_buffer: UniformBuffer<Mat4>,

    bind_groups: AlignmentPolylineBindGroups,
}

impl RenderAsset for GpuAlignmentPolylineMaterial {
    type SourceAsset = AlignmentPolylineMaterial;

    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SRes<AlignmentPolylinePipeline>,
    );

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        param: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self, bevy::render::render_asset::PrepareAssetError<Self::SourceAsset>> {
        // TODO: need to extract color scheme somewhere
        let color_scheme = GpuAlignmentColorScheme::from(source_asset.color_scheme);
        let mut color_scheme_buffer: UniformBuffer<_> = color_scheme.into();
        let mut model_buffer: UniformBuffer<_> = source_asset.model.into();

        color_scheme_buffer.write_buffer(&param.0, &param.1);
        model_buffer.write_buffer(&param.0, &param.1);

        let group_1 = param.0.create_bind_group(
            None,
            &param.2.color_scheme_layout,
            &BindGroupEntries::sequential((color_scheme_buffer.binding().unwrap(),)),
        );

        let group_2 = param.0.create_bind_group(
            None,
            &param.2.model_layout,
            &BindGroupEntries::sequential((model_buffer.binding().unwrap(),)),
        );

        let bind_groups = AlignmentPolylineBindGroups { group_1, group_2 };

        let asset = Self {
            color_scheme_buffer,
            model_buffer: source_asset.model.into(),
            bind_groups,
        };

        Ok(asset)
    }
}

#[derive(Resource, ExtractResource, Clone, Copy, PartialEq, Debug)]
pub struct AlignmentShaderConfig {
    pub line_width: f32,
    pub brightness: f32,
}

#[derive(Asset, Debug, PartialEq, Clone, TypePath)]
pub struct AlignmentPolylineMaterial {
    color_scheme: AlignmentColorScheme,
    model: Mat4,
}

impl AlignmentPolylineMaterial {
    pub fn from_alignment(
        grid: &crate::AlignmentGrid,
        alignment: &crate::Alignment,
        color_scheme: AlignmentColorScheme,
    ) -> Self {
        let x_range = grid
            .x_axis
            .sequence_axis_range(alignment.target_id)
            .unwrap();
        let y_range = grid.y_axis.sequence_axis_range(alignment.query_id).unwrap();

        let x = x_range.start as f64;
        let y = y_range.start as f64;

        let model = Transform::from_xyz(x as f32, y as f32, 0.0).compute_matrix();

        Self {
            model,
            color_scheme,
        }
    }
}

#[derive(ShaderType, Component, Clone)]
struct GpuAlignmentRenderConfig {
    line_width: f32,
    brightness: f32,
    _pad0: f32,
    _pad1: f32,
}

#[derive(ShaderType, Component, Clone, Default)]
struct GpuAlignmentColorScheme {
    m_bg: Vec4,
    eq_bg: Vec4,
    x_bg: Vec4,
    i_bg: Vec4,
    d_bg: Vec4,
}

impl From<crate::render::color::AlignmentColorScheme> for GpuAlignmentColorScheme {
    fn from(value: crate::render::color::AlignmentColorScheme) -> Self {
        let map_color = |c: egui::Color32| {
            Vec4::new(
                c.r() as f32 / u8::MAX as f32,
                c.g() as f32 / u8::MAX as f32,
                c.b() as f32 / u8::MAX as f32,
                c.a() as f32 / u8::MAX as f32,
            )
        };

        GpuAlignmentColorScheme {
            m_bg: map_color(value.m_bg),
            eq_bg: map_color(value.eq_bg),
            x_bg: map_color(value.x_bg),
            i_bg: map_color(value.i_bg),
            d_bg: map_color(value.d_bg),
        }
    }
}

#[derive(Debug, Default, Asset, Clone, TypePath)]
pub struct AlignmentVertices {
    data: Vec<(Vec2, Vec2, CigarOp)>,
}

impl AlignmentVertices {
    pub fn from_alignment_ignore_cigar(alignment: &crate::Alignment) -> Self {
        let location = &alignment.location;

        let tgt_range = &location.target_range;
        let qry_range = &location.query_range;

        let mut from = Vec2::new(tgt_range.start as f32, qry_range.start as f32);
        let mut to = Vec2::new(tgt_range.end as f32, qry_range.end as f32);

        if location.query_strand.is_rev() {
            std::mem::swap(&mut from.y, &mut to.y);
        }

        let vertices = vec![(from, to, CigarOp::M)];

        return Self { data: vertices };
    }

    pub fn from_alignment(alignment: &crate::Alignment) -> Self {
        if alignment.cigar.is_empty() {
            return Self::from_alignment_ignore_cigar(alignment);
        }

        use crate::cigar::CigarOp;

        let location = &alignment.location;

        let mut vertices = Vec::new();
        let mut tgt_cg = 0;
        let mut qry_cg = 0;

        for (op, count) in alignment.cigar.whole_cigar() {
            // tgt_cg and qry_cg are offsets from the start of the cigar
            let tgt_start = tgt_cg;
            let qry_start = qry_cg;

            let (tgt_end, qry_end) = match op {
                CigarOp::Eq | CigarOp::X | CigarOp::M => {
                    tgt_cg += count as u64;
                    qry_cg += count as u64;
                    //
                    (tgt_start + count as u64, qry_start + count as u64)
                }
                CigarOp::I => {
                    qry_cg += count as u64;
                    //
                    (tgt_start, qry_start + count as u64)
                }
                CigarOp::D => {
                    tgt_cg += count as u64;
                    //
                    (tgt_start + count as u64, qry_start)
                }
            };

            let tgt_range = location.map_from_aligned_target_range(tgt_start..tgt_end);
            let qry_range = location.map_from_aligned_query_range(qry_start..qry_end);

            let mut from = Vec2::new(tgt_range.start as f32, qry_range.start as f32);
            let mut to = Vec2::new(tgt_range.end as f32, qry_range.end as f32);

            if location.query_strand.is_rev() {
                std::mem::swap(&mut from.y, &mut to.y);
            }

            vertices.push((from, to, op));
        }

        Self { data: vertices }
    }
}

struct GpuAlignmentVertices {
    vertex_buffer: Buffer,
    segment_count: u32,
}

impl RenderAsset for GpuAlignmentVertices {
    type SourceAsset = AlignmentVertices;

    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        vertices: Self::SourceAsset,
        render_device: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let segment_count = vertices.data.len() as u32;
        let data = vertices
            .data
            .into_iter()
            .flat_map(|(from, to, op)| {
                let mut out = [0u8; 4 * 5];

                let op_ix = match op {
                    CigarOp::M => 0u32,
                    CigarOp::Eq => 1,
                    CigarOp::X => 2,
                    CigarOp::I => 3,
                    CigarOp::D => 4,
                };

                out[0..16].clone_from_slice(bytemuck::cast_slice(&[from, to]));
                out[16..20].clone_from_slice(bytemuck::cast_slice(&[op_ix]));

                out
            })
            .collect::<Vec<_>>();

        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            usage: wgpu::BufferUsages::VERTEX,
            label: Some("Alignment Vertex Buffer"),
            contents: &data,
        });

        Ok(GpuAlignmentVertices {
            vertex_buffer,
            segment_count,
        })
    }
}

fn cleanup_finished_renders(
    mut commands: Commands,
    targets: Query<(Entity, &AlignmentRenderTarget), With<Rendering>>,
) {
    for (entity, tgt) in targets.iter() {
        if tgt.is_ready.load() {
            commands.entity(entity).remove::<AlignmentRenderTarget>();
        }
    }
}

#[derive(Debug, Clone, Copy, Component, ExtractComponent)]
pub struct LineOnlyAlignment;

#[derive(Resource, Default)]
pub struct AlignmentVerticesIndex {
    pub vertices: HashMap<super::alignments::AlignmentIndex, Handle<AlignmentVertices>>,
}

#[derive(Debug, Asset, Clone, Reflect)]
pub struct AlignmentLayoutMaterials {
    materials: HashMap<super::alignments::AlignmentIndex, Handle<AlignmentPolylineMaterial>>,
    vertices: HashMap<super::alignments::AlignmentIndex, Handle<AlignmentVertices>>,
}

impl RenderAsset for AlignmentLayoutMaterials {
    type SourceAsset = Self;

    type Param = ();

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        _param: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        Ok(source_asset)
    }
}

impl AlignmentLayoutMaterials {
    pub fn from_positions_iter(
        material_store: &mut Assets<AlignmentPolylineMaterial>,
        vertex_index: &AlignmentVerticesIndex,
        color_schemes: &super::AlignmentColorSchemes,
        positions: impl IntoIterator<Item = (super::alignments::AlignmentIndex, Vec2)>,
    ) -> Self {
        let mut materials = HashMap::default();
        let mut vertices = HashMap::default();

        for (alignment, pos) in positions {
            let Some(verts) = vertex_index.vertices.get(&alignment) else {
                continue;
            };

            let model = Transform::from_xyz(pos.x, pos.y, 0.0).compute_matrix();
            let color_scheme = color_schemes.get(&alignment).clone();

            let mat = AlignmentPolylineMaterial {
                color_scheme,
                model,
            };

            materials.insert(alignment, material_store.add(mat));
            vertices.insert(alignment, verts.clone());
        }

        Self {
            materials,
            vertices,
        }
    }
}

fn queue_draw_alignment_lines(
    mut commands: Commands,

    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<AlignmentPolylinePipeline>,

    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_vertices: Res<RenderAssets<GpuAlignmentVertices>>,
    gpu_materials: Res<RenderAssets<GpuAlignmentPolylineMaterial>>,

    layout_mats: Res<RenderAssets<AlignmentLayoutMaterials>>,

    // children: Query<&Children>,
    targets: Query<
        (
            Entity,
            &AlignmentViewer,
            &AlignmentViewerImages,
            &AlignmentRenderTarget,
            &VertexBindGroup,
            &ExtractedViewerChildren,
        ),
        Without<Rendering>,
    >,

    alignments: Query<(&Handle<AlignmentLayoutMaterials>, Has<LineOnlyAlignment>)>,
) {
    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
        return;
    };

    for (tgt_entity, viewer, viewer_imgs, render_target, vx_bind_group, children) in targets.iter()
    {
        //
        let Some(view) = viewer.next_view else {
            continue;
        };

        let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AlignmentRenderer".into()),
        });

        let tgt_handle = &viewer_imgs.back;
        let tgt_img = gpu_images.get(tgt_handle).unwrap();

        let depth_handle = &viewer_imgs.back_depth;
        let depth_img = gpu_images.get(depth_handle).unwrap();

        // let bp_per_px = (view.width() / tgt_img.size.x as f64) as f32;
        let px_per_bp = tgt_img.size.x as f64 / view.width();

        let bg = viewer.background_color.to_srgba();
        let clear_color = wgpu::Color {
            r: bg.red as f64,
            g: bg.green as f64,
            b: bg.blue as f64,
            a: bg.alpha as f64,
        };

        {
            let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("AlignmentRendererPass".into()),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &tgt_img.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_img.texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &vx_bind_group.group_0, &[]);

            for &child in children.0.iter() {
                let Ok((mats, line_only)) = alignments.get(child) else {
                    dbg!();
                    continue;
                };

                // TODO `super::AlignmentRenderConfig` has the `base_level...min_bp_per_px`
                // value, but it's not extracted into the render world & should be reorganized
                // so i'll just hardcode it for now (might need to take DPI scaling into account)
                if px_per_bp > 1.0 && !line_only {
                    continue;
                }

                let Some(mats) = layout_mats.get(mats) else {
                    continue;
                };

                for (alignment, mat) in mats.materials.iter() {
                    let Some(verts) = mats.vertices.get(alignment) else {
                        continue;
                    };

                    let gpu_mat = gpu_materials.get(mat);
                    let gpu_verts = gpu_vertices.get(verts);

                    let Some((mat, verts)) = gpu_mat.zip(gpu_verts) else {
                        continue;
                    };

                    pass.set_bind_group(1, &mat.bind_groups.group_1, &[]);
                    pass.set_bind_group(2, &mat.bind_groups.group_2, &[]);
                    pass.set_vertex_buffer(0, wgpu::Buffer::slice(&verts.vertex_buffer, ..));

                    pass.draw(0..6, 0..verts.segment_count);
                }
            }
        }

        render_queue.0.submit([cmds.finish()]);

        let is_ready = render_target.is_ready.clone();
        render_queue.0.on_submitted_work_done(move || {
            is_ready.store(true);
        });
        commands.entity(tgt_entity).insert(Rendering);
    }
}
