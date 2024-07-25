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
        Render, RenderApp, RenderSet,
    },
};
use wgpu::{util::BufferInitDescriptor, ColorWrites, ShaderStages, VertexStepMode};

use crate::{render::color::AlignmentColorScheme, CigarOp};

use super::view::AlignmentViewport;

/*



*/

pub struct AlignmentRendererPlugin;

impl Plugin for AlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<AlignmentVertices>()
            .init_asset::<AlignmentPolylineMaterial>()
            .add_plugins(RenderAssetPlugin::<GpuAlignmentVertices>::default())
            .add_plugins(ExtractComponentPlugin::<AlignmentRenderTarget>::default())
            .add_plugins(ExtractComponentPlugin::<Handle<AlignmentVertices>>::default())
            .add_plugins(ExtractResourcePlugin::<AlignmentShaderConfig>::default())
            .add_systems(Startup, setup_alignment_display_image)
            .add_systems(PreUpdate, update_alignment_display_target)
            .add_systems(
                Update,
                trigger_render.run_if(|view: Res<AlignmentViewport>| view.is_changed()),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<AlignmentPolylinePipeline>()
            .add_systems(Startup, setup_projection_vertex_config_bind_group)
            /*
            .add_systems(
                Render,
                (create_alignment_uniforms, update_alignment_uniforms)
                    .chain()
                    .in_set(RenderSet::PrepareResources),
            )
            .add_systems(
                Render,
                create_alignment_bind_groups.in_set(RenderSet::PrepareBindGroups),
            )
            */
            .add_systems(
                Render,
                update_projection_config_uniforms.in_set(RenderSet::PrepareResources),
            )
            .add_systems(Render, queue_alignment_draw.in_set(RenderSet::Render))
            .add_systems(Render, cleanup_finished_renders.in_set(RenderSet::Cleanup));
    }
}

fn setup_alignment_display_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
        AlignmentDisplayImage,
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

    commands.insert_resource(AlignmentBackImage {
        image: back_img_handle,
    })
}

fn update_alignment_display_target(
    mut commands: Commands,

    mut back_image: ResMut<AlignmentBackImage>,

    mut sprites: Query<
        (Entity, &mut Handle<Image>, &AlignmentRenderTarget),
        With<AlignmentDisplayImage>,
    >,
) {
    let Ok((entity, mut old_sprite_img, render_target)) = sprites.get_single_mut() else {
        return;
    };

    if !render_target.is_ready.load() {
        return;
    }

    std::mem::swap(old_sprite_img.as_mut(), &mut back_image.image);

    commands.entity(entity).remove::<AlignmentRenderTarget>();
}

#[derive(Debug, Component)]
pub struct AlignmentColor {
    pub color_scheme: AlignmentColorScheme,
}

#[derive(Debug, Component)]
pub struct AlignmentDisplayImage;

#[derive(Resource)]
struct AlignmentBackImage {
    image: Handle<Image>,
}

#[derive(Clone, Component, ExtractComponent)]
pub struct AlignmentRenderTarget {
    pub alignment_view: crate::view::View,
    pub canvas_size: UVec2,

    image: Handle<Image>,
    is_ready: Arc<crossbeam::atomic::AtomicCell<bool>>,
}

#[derive(Component)]
struct Rendering;

#[derive(Resource)]
struct AlignmentProjVertexBindGroup {
    group_0: BindGroup,

    proj_buffer: UniformBuffer<Mat4>,
    config_buffer: UniformBuffer<GpuAlignmentRenderConfig>,
}

fn setup_projection_vertex_config_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<AlignmentPolylinePipeline>,
) {
    let proj_buffer: UniformBuffer<_> = Mat4::IDENTITY.into();
    let config_buffer: UniformBuffer<_> = GpuAlignmentRenderConfig {
        line_width: 4.0,
        brightness: 1.0,
        _pad0: 0.0,
        _pad1: 0.0,
    }
    .into();

    let group_0 = render_device.create_bind_group(
        Some("AlignmentRenderConfig"),
        &pipeline.proj_config_layout,
        &BindGroupEntries::sequential((
            proj_buffer.binding().unwrap(),
            config_buffer.binding().unwrap(),
        )),
    );

    commands.insert_resource(AlignmentProjVertexBindGroup {
        group_0,
        proj_buffer,
        config_buffer,
    });
}

fn update_projection_config_uniforms(
    mut uniforms: ResMut<AlignmentProjVertexBindGroup>,
    config: Res<AlignmentShaderConfig>,

    targets: Query<&AlignmentRenderTarget, Without<Rendering>>,
) {
    let Ok(target) = targets.get_single() else {
        return;
    };

    let proj_uv = target.alignment_view.to_mat4();
    let proj = Mat4::from_cols_array(proj_uv.as_array());
    uniforms.proj_buffer.set(proj);

    if config.is_changed() {
        let mut cfg = uniforms.config_buffer.get().clone();
        cfg.line_width = config.line_width;
        cfg.brightness = config.brightness;
        uniforms.config_buffer.set(cfg);
    }
}

fn trigger_render(
    mut commands: Commands,

    mut images: ResMut<Assets<Image>>,
    back_image: Res<AlignmentBackImage>,

    alignment_viewport: Res<AlignmentViewport>,

    // active_renders: Query<&AlignmentRenderTarget>,
    windows: Query<&Window>,
    display_sprites: Query<Entity, (With<AlignmentDisplayImage>, Without<AlignmentRenderTarget>)>,
) {
    println!("in trigger_render()");
    let Ok(sprite_ent) = display_sprites.get_single() else {
        return;
    };

    println!("got sprite: {sprite_ent:?}");

    let win_size = windows.single().resolution.physical_size();

    // resize back image
    {
        let tgt_img = images.get_mut(&back_image.image).unwrap();
        let size = tgt_img.size();

        if size.x != win_size.x || size.y != win_size.y {
            tgt_img.resize(wgpu::Extent3d {
                width: win_size.x,
                height: win_size.y,
                depth_or_array_layers: 1,
            });
        }
    }

    println!("inserting AlignmentRenderTarget component");
    // insert AlignmentRenderTarget component on sprite
    commands.entity(sprite_ent).insert(AlignmentRenderTarget {
        alignment_view: alignment_viewport.view,
        canvas_size: win_size,
        image: back_image.image.clone(),
        is_ready: Arc::new(false.into()),
    });
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

        let shader = world.load_asset::<Shader>("shaders/lines_color_scheme.wgsl");
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
        let color_scheme = GpuAlignmentColorScheme {
            m_bg: Vec4::new(0.0, 0.0, 0.0, 1.0),
            eq_bg: Vec4::new(0.0, 0.0, 0.0, 1.0),
            x_bg: Vec4::new(1.0, 0.0, 0.0, 1.0),
            i_bg: Vec4::new(1.0, 1.0, 1.0, 1.0),
            d_bg: Vec4::new(1.0, 1.0, 1.0, 1.0),
        };
        let color_scheme_buffer: UniformBuffer<_> = color_scheme.into();
        let model_buffer: UniformBuffer<_> = source_asset.model.into();

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

// #[derive(ShaderType, Component, Clone)]
#[derive(Asset, Debug, PartialEq, Clone, TypePath)]
pub struct AlignmentPolylineMaterial {
    // projection: Mat4,
    // config: AlignmentRenderConfig,
    color_scheme: AlignmentColorScheme,
    model: Mat4,
    // projection: UniformBuffer<Mat4>,
    // config: UniformBuffer<GpuAlignmentConfig>,
}

impl AlignmentPolylineMaterial {
    pub fn from_alignment(alignment: &crate::Alignment) -> Self {
        // derive model from location
        // pass color scheme as argument
        // config... shouldn't even be here
        // alignment.location
        //
        todo!();
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

#[derive(Debug, Default, Asset, Clone, TypePath)]
pub struct AlignmentVertices {
    data: Vec<(Vec2, Vec2, CigarOp)>,
    // data: Vec<u8>,
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
    // mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<AlignmentPolylinePipeline>,

    proj_config: Res<AlignmentProjVertexBindGroup>,

    gpu_images: Res<RenderAssets<GpuImage>>,
    gpu_alignments: Res<RenderAssets<GpuAlignmentVertices>>,

    // draw_params: Option<Res<AlignmentRenderParams>>,
    targets: Query<(Entity, &AlignmentRenderTarget), Without<Rendering>>,
    // target: Option<Res<AlignmentRenderTarget>>,
    alignments: Query<(&Handle<AlignmentVertices>, &GpuAlignmentPolylineMaterial)>,
) {
    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
        return;
    };

    for (_tgt_entity, tgt) in targets.iter() {
        let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("AlignmentRenderer".into()),
        });

        let tgt_handle = &tgt.image;

        let tgt_img = gpu_images.get(tgt_handle).unwrap();

        let attch = wgpu::RenderPassColorAttachment {
            view: &tgt_img.texture_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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
                pass.set_bind_group(1, &material.bind_groups.group_1, &[]);
                pass.set_bind_group(2, &material.bind_groups.group_2, &[]);

                let vertices = gpu_alignments.get(vertices).unwrap();
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
    }

    //
}

fn cleanup_finished_renders(
    mut commands: Commands,
    targets: Query<(Entity, &AlignmentRenderTarget), With<Rendering>>,
) {
    for (entity, tgt) in targets.iter() {
        if tgt.is_ready.load() {
            commands.entity(entity).despawn();
        }
    }
}

/*
pub mod graph {

    use super::*;
    use bevy::render::render_graph;

    pub struct AlignmentLinesNode {
        target: Option<Handle<Image>>,
        alignments: QueryState<(
            &'static Handle<AlignmentVertices>,
            &'static GpuAlignmentPolylineMaterial,
        )>,

        ready: bool,
    }

    impl AlignmentLinesNode {
        pub fn new(world: &mut World) -> Self {
            Self {
                target: None,
                alignments: world.query(),
                ready: false,
            }
        }
    }

    impl render_graph::Node for AlignmentLinesNode {
        fn update(&mut self, world: &mut World) {
            self.alignments.update_archetypes(world);

            if !self.ready {
                let pipeline = world.resource::<AlignmentPolylinePipeline>();
                let pipeline_cache = world.resource::<PipelineCache>();

                match pipeline_cache.get_render_pipeline_state(pipeline.pipeline) {
                    bevy::render::render_resource::CachedPipelineState::Ok(_) => {
                        self.ready = true;
                    }
                    bevy::render::render_resource::CachedPipelineState::Err(err) => {
                        panic!("Initializing alignment render pipeline: {err:?}");
                    }
                    _ => {}
                }
            }
        }

        fn run<'w>(
            &self,
            graph: &mut render_graph::RenderGraphContext,
            render_context: &mut bevy::render::renderer::RenderContext<'w>,
            world: &'w World,
        ) -> Result<(), render_graph::NodeRunError> {
            todo!()
        }
    }
}
*/
