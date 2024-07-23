use bevy::{
    ecs::system::lifetimeless::SRes,
    prelude::*,
    render::{
        extract_component::ExtractComponentPlugin,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            CachedRenderPipelineId, PipelineCache, RenderPipelineDescriptor, ShaderType,
            UniformBuffer,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use wgpu::{util::BufferInitDescriptor, ColorWrites, ShaderStages, VertexStepMode};

use crate::{render::color::AlignmentColorScheme, CigarOp};

/*



*/

pub struct AlignmentRendererPlugin;

impl Plugin for AlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RenderAssetPlugin::<GpuAlignmentVertices>::default())
            .add_plugins(ExtractComponentPlugin::<Handle<AlignmentVertices>>::default());
    }

    fn finish(&self, app: &mut App) {
        let mut render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<AlignmentPolylinePipeline>();
    }
}

#[derive(Debug, Component)]
pub struct AlignmentColor {
    pub color_scheme: AlignmentColorScheme,
}

fn create_alignment_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<AlignmentPolylinePipeline>,
    extract_query: bevy::render::Extract<Query<(Entity, &Handle<AlignmentVertices>)>>,
    with_uniforms: Query<(Entity, &Handle<AlignmentVertices>), With<GpuAlignmentPolylineMaterial>>,
) {
    for (entity, handle) in extract_query.iter() {
        if with_uniforms.contains(entity) {
            continue;
        }

        let proj_buffer: UniformBuffer<_> = Mat4::IDENTITY.into();
        let config_buffer: UniformBuffer<_> = GpuAlignmentRenderConfig {
            line_width: 4.0,
            brightness: 1.0,
            _pad0: 0.0,
            _pad1: 0.0,
        }
        .into();

        // TODO: each color scheme should have its own buffer shared by all
        // alignments that use it
        let color_scheme_buffer: UniformBuffer<_> = GpuAlignmentColorScheme::default().into();
        let model_buffer: UniformBuffer<_> = Mat4::IDENTITY.into();

        let group_0 = render_device.create_bind_group(
            None,
            &pipeline.proj_config_layout,
            &BindGroupEntries::sequential((
                proj_buffer.binding().unwrap(),
                config_buffer.binding().unwrap(),
            )),
        );

        let group_1 = render_device.create_bind_group(
            None,
            &pipeline.color_scheme_layout,
            &BindGroupEntries::sequential((color_scheme_buffer.binding().unwrap(),)),
        );

        let group_2 = render_device.create_bind_group(
            None,
            &pipeline.model_layout,
            &BindGroupEntries::sequential((model_buffer.binding().unwrap(),)),
        );

        let material = GpuAlignmentPolylineMaterial {
            proj_buffer,
            config_buffer,
            color_scheme_buffer,
            model_buffer,
            bind_groups: AlignmentPolylineBindGroups {
                group_0,
                group_1,
                group_2,
            },
        };

        commands
            .entity(entity)
            .insert((material, handle.clone_weak()));
    }
}

fn extract_alignments(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    extract_query: bevy::render::Extract<
        Query<(
            Entity,
            &InheritedVisibility,
            &Transform,
            &GlobalTransform,
            &AlignmentColor,
            &Handle<AlignmentVertices>,
        )>,
    >,
    mut extracted: Query<(Entity, &mut GpuAlignmentPolylineMaterial)>,
    // extracted: Query<(Entity, &ExtractedAlignment)>,
    // extracted: Query<(Entity
) {
    let mut values = Vec::with_capacity(*previous_len);

    // for (entity, inherited_visibility, transform, global_transform, handle) in extract_query.iter()
    for row in extract_query.iter() {
        let (entity, _, _, _, color, handle) = row;
        // if !inherited_visibility.get() {
        //     continue;
        // }
        // let transform = transform.compute_matrix();

        // create uniforms; `values` here should contain all the per-alignment data
        // needed by the shader

        values.push((entity, (handle.clone_weak(), ())))
        // values.push((entity, (handle.clone_weak(), PolylineUniform { transform })));
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
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
                            // format: todo!(),
                            // offset: todo!(),
                            // shader_location: todo!(),
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        render_resource::VertexAttribute {
                            // format: todo!(),
                            // offset: todo!(),
                            // shader_location: todo!(),
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        render_resource::VertexAttribute {
                            // format: todo!(),
                            // offset: todo!(),
                            // shader_location: todo!(),
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
    group_0: BindGroup,
    group_1: BindGroup,
    group_2: BindGroup,
}

#[derive(Component)]
struct GpuAlignmentPolylineMaterial {
    proj_buffer: UniformBuffer<Mat4>,
    config_buffer: UniformBuffer<GpuAlignmentRenderConfig>,
    color_scheme_buffer: UniformBuffer<GpuAlignmentColorScheme>,
    model_buffer: UniformBuffer<Mat4>,

    bind_groups: AlignmentPolylineBindGroups,
}

/*
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


        todo!()
    }
}
*/

#[derive(Component, Clone, Copy, PartialEq, Debug)]
pub struct AlignmentRenderConfig {
    pub line_width: f32,
    pub brightness: f32,
}

// #[derive(ShaderType, Component, Clone)]
#[derive(Asset, Debug, PartialEq, Clone, TypePath)]
struct AlignmentPolylineMaterial {
    projection: Mat4,

    config: AlignmentRenderConfig,
    color_scheme: AlignmentColorScheme,
    // model: Mat4,
    // projection: UniformBuffer<Mat4>,
    // config: UniformBuffer<GpuAlignmentConfig>,
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

// #[derive(ShaderType, Component, Clone)]
// struct AlignmentModel {
//     model: Mat4,
// }

// struct GpuAlignmentModel {
//     model:
// }

#[derive(Debug, Default, Asset, Clone, TypePath)]
pub struct AlignmentVertices {
    data: Vec<(Vec2, Vec2, CigarOp)>,
    // data: Vec<u8>,
}

pub struct ExtractedAlignment {
    pub vertices: Handle<AlignmentVertices>,
    pub material: GpuAlignmentPolylineMaterial,
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
