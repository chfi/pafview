use std::sync::Arc;

use bevy::{
    ecs::system::lifetimeless::SRes,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{
            PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssetUsages, RenderAssets,
        },
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
    paf::AlignmentIndex,
    render::color::AlignmentColorScheme,
    CigarOp,
};

use super::view::AlignmentViewport;

mod base_level;

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
            .insert_resource(AlignmentShaderConfig {
                line_width: 8.0,
                brightness: 1.0,
            })
            .init_resource::<AlignmentVerticesIndex>()
            .add_plugins(ExtractComponentPlugin::<AlignmentDisplayImage>::default())
            .add_plugins(ExtractComponentPlugin::<AlignmentRenderTarget>::default())
            .add_plugins(ExtractResourcePlugin::<AlignmentShaderConfig>::default())
            .add_systems(Startup, setup_main_alignment_display_image)
            .add_systems(PreUpdate, create_override_material_assets)
            .add_systems(
                PreUpdate,
                (
                    update_alignment_display_target,
                    update_alignment_shader_config,
                ),
            )
            .add_systems(PreUpdate, update_main_alignment_image_view)
            .add_systems(
                Update,
                (
                    update_main_alignment_image_view,
                    trigger_render,
                    update_alignment_display_transform,
                )
                    .chain()
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
            .add_systems(ExtractSchedule, extract_alignment_position_overrides)
            .add_systems(
                Render,
                (queue_alignment_draw, queue_alignment_draw_overrides).in_set(RenderSet::Render),
            )
            .add_systems(Render, cleanup_finished_renders.in_set(RenderSet::Cleanup));
    }
}

#[derive(Resource, Default)]
pub struct AlignmentVerticesIndex {
    pub vertices: HashMap<AlignmentIndex, Handle<AlignmentVertices>>,
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

fn setup_main_alignment_display_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    //

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

    let display_sprite = commands.spawn((
        MainAlignmentView,
        AlignmentDisplayImage::default(),
        AlignmentDisplayBackImage {
            image: back_img_handle.clone(),
        },
        RenderLayers::layer(1),
        SpriteBundle {
            sprite: Sprite {
                color: Color::WHITE,
                ..default()
            },
            texture: img_handle.clone(),
            transform: Transform::IDENTITY,
            ..default()
        },
    ));
}

fn update_main_alignment_image_view(
    alignment_viewport: Res<AlignmentViewport>,
    mut sprites: Query<&mut AlignmentDisplayImage, With<MainAlignmentView>>,
) {
    for mut display_img in sprites.iter_mut() {
        display_img.next_view = Some(alignment_viewport.view);
    }
}

// checks all alignment display sprites that have an `AlignmentRenderTarget`
// signaling a render in progress; if the render is complete, the sprite texture
// is flipped to the new render and the `AlignmentRenderTarget` component removed
pub(super) fn update_alignment_display_target(
    mut commands: Commands,

    mut sprites: Query<(
        Entity,
        &mut Handle<Image>,
        &AlignmentRenderTarget,
        &mut AlignmentDisplayImage,
        &mut AlignmentDisplayBackImage,
    )>,
) {
    for (entity, mut old_sprite_img, render_target, mut display_img, mut back_image) in
        sprites.iter_mut()
    {
        if !render_target.is_ready.load() {
            return;
        }

        display_img.rendered_view = Some(render_target.alignment_view);
        display_img.last_render_time = Some(std::time::Instant::now());

        // println!("swapping front/back for {entity:?}");
        std::mem::swap(old_sprite_img.as_mut(), &mut back_image.image);

        commands.entity(entity).remove::<AlignmentRenderTarget>();
    }
}

fn update_alignment_display_transform(
    images: Res<Assets<Image>>,
    // alignment_viewport: Res<AlignmentViewport>,
    windows: Query<&Window>,
    mut display_sprites: Query<
        (
            &mut Transform,
            &mut Sprite,
            &Handle<Image>,
            &AlignmentDisplayImage,
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
                    .with_scale(Vec3::new(w_rat as f32, h_rat as f32, 0.0));
        }
    }
}

#[derive(Debug, Component)]
pub struct AlignmentColor {
    pub color_scheme: AlignmentColorScheme,
}

#[derive(Debug, Component, ExtractComponent, Clone, Default)]
pub struct AlignmentDisplayImage {
    pub next_view: Option<crate::view::View>,
    rendered_view: Option<crate::view::View>,
    last_render_time: Option<std::time::Instant>,
}

#[derive(Component)]
pub struct AlignmentDisplayBackImage {
    pub image: Handle<Image>,
}

#[derive(Debug, Clone, Component, ExtractComponent)]
pub struct AlignmentRenderTarget {
    pub alignment_view: crate::view::View,
    pub canvas_size: UVec2,

    pub image: Handle<Image>,
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

    sprites: Query<Entity, Added<AlignmentDisplayImage>>,
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

fn trigger_render(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,
    frame_count: Res<bevy::core::FrameCount>,
    // alignment_viewport: Res<AlignmentViewport>,
    shader_config: Res<AlignmentShaderConfig>,

    windows: Query<&Window>,
    display_sprites: Query<
        (Entity, &AlignmentDisplayImage, &AlignmentDisplayBackImage),
        (With<MainAlignmentView>, Without<AlignmentRenderTarget>),
    >,
) {
    if frame_count.0 < 3 {
        return;
    }

    for (sprite_ent, display_img, back_image) in display_sprites.iter() {
        if let Some(ms_since_last_render) = display_img
            .last_render_time
            .map(|t| t.elapsed().as_millis())
        {
            if ms_since_last_render < 10 {
                return;
            }
        }

        let win_size = windows.single().resolution.physical_size();

        // resize back image
        let resized = {
            let tgt_img = images.get_mut(&back_image.image).unwrap();
            let size = tgt_img.size();

            if size.x != win_size.x || size.y != win_size.y {
                tgt_img.resize(wgpu::Extent3d {
                    width: win_size.x,
                    height: win_size.y,
                    depth_or_array_layers: 1,
                });
                true
            } else {
                false
            }
        };

        if let Some(next_view) = display_img.next_view {
            let changed_view = display_img.rendered_view != Some(next_view);
            if changed_view || resized || shader_config.is_changed() {
                // println!(
                //     "triggering render for {sprite_ent:?}\t\
                //     view changed: {changed_view}, resized: {resized}"
                // );
                commands.entity(sprite_ent).insert(AlignmentRenderTarget {
                    alignment_view: next_view,
                    canvas_size: win_size,
                    image: back_image.image.clone(),
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
            depth_stencil: None,
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

        // let x = (x_range.start as f64) + 0.5 * (x_range.end - x_range.start) as f64;
        // let y = (y_range.start as f64) + 0.5 * (y_range.end - y_range.start) as f64;
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
    // data: Vec<u8>,
}

impl AlignmentVertices {
    pub fn from_alignment(alignment: &crate::Alignment) -> Self {
        use crate::cigar::CigarOp;
        // use ultraviolet::DVec2;

        let mut vertices = Vec::new();

        let mut tgt_cg = 0;
        let mut qry_cg = 0;

        let location = &alignment.location;

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

fn queue_alignment_draw(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<AlignmentPolylinePipeline>,

    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_alignments: Res<RenderAssets<GpuAlignmentVertices>>,
    gpu_materials: Res<RenderAssets<GpuAlignmentPolylineMaterial>>,

    targets: Query<
        (Entity, &AlignmentRenderTarget, &VertexBindGroup),
        (
            Without<Rendering>,
            Without<ExtractedAlignmentMaterialOverrides>,
        ),
    >,
    alignments: Query<(
        &Handle<AlignmentVertices>,
        &Handle<AlignmentPolylineMaterial>,
    )>,
) {
    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
        return;
    };

    for (tgt_entity, tgt, proj_config) in targets.iter() {
        let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AlignmentRenderer".into()),
        });

        let tgt_handle = &tgt.image;

        let tgt_img = gpu_images.get(tgt_handle).unwrap();

        let attch = wgpu::RenderPassColorAttachment {
            view: &tgt_img.texture_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                // load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        };

        {
            let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("AlignmentRendererPass".into()),
                color_attachments: &[Some(attch)],
                ..default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &proj_config.group_0, &[]);

            for (vertices, material) in alignments.iter() {
                let Some(vertices) = gpu_alignments.get(vertices) else {
                    println!("vertices missing!");
                    continue;
                };
                let Some(material) = gpu_materials.get(material) else {
                    println!("material missing!");
                    continue;
                };

                pass.set_bind_group(1, &material.bind_groups.group_1, &[]);
                pass.set_bind_group(2, &material.bind_groups.group_2, &[]);
                pass.set_vertex_buffer(0, wgpu::Buffer::slice(&vertices.vertex_buffer, ..));

                pass.draw(0..6, 0..vertices.segment_count);
            }
        }

        // this will be one submit per render, which is fine for
        // as long as there will only be one (or a few) per frame,
        // but later it may be worth combining into one
        render_queue.0.submit([cmds.finish()]);

        let is_ready = tgt.is_ready.clone();
        render_queue.0.on_submitted_work_done(move || {
            is_ready.store(true);
        });
        commands.entity(tgt_entity).insert(Rendering);
    }
}

fn cleanup_finished_renders(
    mut commands: Commands,
    targets: Query<(Entity, &AlignmentRenderTarget), With<Rendering>>,
) {
    for (entity, tgt) in targets.iter() {
        // print!("{entity:?} rendering: ");
        if tgt.is_ready.load() {
            // println!("complete");
            // println!("finished render for entity {entity:?}");
            commands.entity(entity).remove::<AlignmentRenderTarget>();
        } else {
            // println!("pending");
        }
    }
}

#[derive(Debug, Component, Clone)]
pub struct AlignmentPositionOverrides {
    positions: HashMap<AlignmentIndex, Vec2>,
    materials: HashMap<AlignmentIndex, Handle<AlignmentPolylineMaterial>>,
    vertices: HashMap<AlignmentIndex, Handle<AlignmentVertices>>,
}

fn create_override_material_assets(
    color_schemes: Res<super::AlignmentColorSchemes>,
    vertex_index: Res<AlignmentVerticesIndex>,

    mut materials: ResMut<Assets<AlignmentPolylineMaterial>>,

    mut overrides: Query<&mut AlignmentPositionOverrides, Changed<AlignmentPositionOverrides>>,
) {
    for mut overrides in overrides.iter_mut() {
        let mut override_mats = Vec::with_capacity(overrides.positions.len());

        for (&ix, &pos) in overrides.positions.iter() {
            let model = Transform::from_xyz(pos.x, pos.y, 0.0).compute_matrix();
            let color_scheme = color_schemes.colors.get(ix).clone();

            let mat = AlignmentPolylineMaterial {
                color_scheme,
                model,
            };

            override_mats.push((ix, materials.add(mat)));
        }

        for (ix, mat_handle) in override_mats {
            let vert_handle = vertex_index.vertices.get(&ix).unwrap();
            overrides.vertices.insert(ix, vert_handle.clone());
            overrides.materials.insert(ix, mat_handle);
        }
    }
}

#[derive(Debug, Component)]
struct ExtractedAlignmentMaterialOverrides {
    materials: HashMap<AlignmentIndex, Handle<AlignmentPolylineMaterial>>,
    vertices: HashMap<AlignmentIndex, Handle<AlignmentVertices>>,
}

fn extract_alignment_position_overrides(
    mut commands: Commands,
    query: Extract<
        Query<(Entity, &AlignmentPositionOverrides), Changed<AlignmentPositionOverrides>>,
    >,
) {
    let mut values = Vec::new();

    for (entity, overrides) in query.iter() {
        values.push((
            entity,
            ExtractedAlignmentMaterialOverrides {
                materials: overrides.materials.clone(),
                vertices: overrides.vertices.clone(),
            },
        ));
    }

    commands.insert_or_spawn_batch(values);
}

fn queue_alignment_draw_overrides(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<AlignmentPolylinePipeline>,

    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_alignments: Res<RenderAssets<GpuAlignmentVertices>>,
    gpu_materials: Res<RenderAssets<GpuAlignmentPolylineMaterial>>,

    targets: Query<
        (
            Entity,
            &AlignmentRenderTarget,
            &VertexBindGroup,
            &ExtractedAlignmentMaterialOverrides,
        ),
        Without<Rendering>,
    >,
    // alignments: Query<(
    //     &Handle<AlignmentVertices>,
    //     &Handle<AlignmentPolylineMaterial>,
    // )>,
) {
    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
        return;
    };

    for (tgt_entity, tgt, proj_config, overrides) in targets.iter() {
        let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AlignmentRenderer".into()),
        });

        let tgt_handle = &tgt.image;

        let tgt_img = gpu_images.get(tgt_handle).unwrap();

        let attch = wgpu::RenderPassColorAttachment {
            view: &tgt_img.texture_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                // load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        };

        {
            let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("AlignmentRendererPass".into()),
                color_attachments: &[Some(attch)],
                ..default()
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &proj_config.group_0, &[]);

            for (ix, vertices) in overrides.vertices.iter() {
                let Some(vertices) = gpu_alignments.get(vertices) else {
                    println!("vertices missing!");
                    continue;
                };
                let material = overrides.materials.get(ix).unwrap();
                let Some(material) = gpu_materials.get(material) else {
                    println!("material missing!");
                    continue;
                };

                pass.set_bind_group(1, &material.bind_groups.group_1, &[]);
                pass.set_bind_group(2, &material.bind_groups.group_2, &[]);
                pass.set_vertex_buffer(0, wgpu::Buffer::slice(&vertices.vertex_buffer, ..));

                pass.draw(0..6, 0..vertices.segment_count);
            }
        }

        // this will be one submit per render, which is fine for
        // as long as there will only be one (or a few) per frame,
        // but later it may be worth combining into one
        render_queue.0.submit([cmds.finish()]);

        let is_ready = tgt.is_ready.clone();
        render_queue.0.on_submitted_work_done(move || {
            is_ready.store(true);
        });
        commands.entity(tgt_entity).insert(Rendering);
    }
}
