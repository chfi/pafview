use bevy::{
    math::DVec2,
    prelude::*,
    render::render_resource::RawBufferVec,
    tasks::{AsyncComputeTaskPool, Task},
};
use pipeline::PolylineVertices;
use wgpu::BufferUsages;

use crate::app::alignments::layout::SeqPairLayout;

use super::*;

pub struct SampledAlignmentRendererPlugin;

impl Plugin for SampledAlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<BackRenderTarget>::default())
            .add_systems(Startup, spawn_main_sampled_alignment_viewer);

        todo!();

        //
    }
}

#[derive(Component, Default)]
struct SampledAlignmentViewer {
    view: Option<crate::view::View>,

    last_rendered: Option<RenderParams>,
    last_vertex_params: Option<RenderParams>,
}

#[derive(Default)]
struct SampledVertices {
    buffer_data: Vec<VertexData>,
    sampling_params: Option<VertexSamplingParams>,
}

#[derive(Clone, Copy)]
struct VertexSamplingParams {
    view: crate::view::View,
    /// in basepairs per pixel
    scale: f64,
}

fn spawn_main_sampled_alignment_viewer(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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

    let front_color = images.add(front_image);
    let back_color = images.add(back_image);

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

    let front_vertices = PolylineVertices::new();
    let back_vertices = BackVertexBuffer(PolylineVertices::new());

    commands
        .spawn((
            SampledAlignmentViewer::default(),
            front_vertices,
            back_vertices,
            SpriteBundle::default(),
            // SampledVertices::default(),
        ))
        .insert((
            front_color.clone(),
            FrontRenderTarget(RenderTargetImages {
                color: front_color,
                depth: front_depth,
            }),
            BackRenderTarget(RenderTargetImages {
                color: back_color,
                depth: back_depth,
            }),
        ));
}

// initialize vertex buffer(s) and uniforms for viewers
// fn setup_gpu_resources(//
//     viewers: Query<(Entity, &)>,
// ) {
//     // initialize `PolylineVertices` on viewer... also the back buffer!!
//     //
// }

#[derive(Component)]
struct BackVertexBuffer(PolylineVertices);

#[derive(Clone)]
struct RenderTargetImages {
    color: Handle<Image>,
    depth: Handle<Image>,
}

#[derive(Component)]
struct FrontRenderTarget(RenderTargetImages);

#[derive(Component, ExtractComponent, Clone)]
struct BackRenderTarget(RenderTargetImages);

#[derive(Component)]
struct VertexSamplingTask {
    task: Task<SampledVertices>,
    sampling_params: VertexSamplingParams,
}

fn update_alignment_viewer_params(
    viewport: Res<AlignmentViewport>,
    //
    mut viewers: Query<&mut SampledAlignmentViewer>,
) {
    for mut viewer in viewers.iter_mut() {
        viewer.view = Some(viewport.view);
    }

    //
}

fn spawn_vertex_sampling_tasks(
    mut commands: Commands,

    layouts: Res<Assets<SeqPairLayout>>,

    layout_roots: Query<(&Transform, &Handle<SeqPairLayout>)>,

    viewers: Query<(Entity, &SampledAlignmentViewer), Without<VertexSamplingTask>>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    let task_pool = AsyncComputeTaskPool::get();

    for (viewer_ent, viewer) in viewers.iter() {
        let Some(next_view) = viewer.view else {
            continue;
        };

        let bp_per_px = next_view.width() / window.physical_size().x as f64;

        // TODO: spawn task if `next_view` has escaped bounds of the sampling
        // params in `vertices`, or if scale has changed "enough"
        let need_new_vertices = if let Some(sampled_params) = vertices.sampling_params {
            let s_view: crate::view::View = sampled_params.view;

            let view_out_of_bounds = s_view.x_min > next_view.x_max
                || s_view.x_max < next_view.x_min
                || s_view.y_min > next_view.y_max
                || s_view.y_max < next_view.y_min;

            let rel_scale = next_view.width() / s_view.width();
            let beyond_scale_limit = rel_scale < 0.5;

            view_out_of_bounds || beyond_scale_limit
        } else {
            true
        };

        if !need_new_vertices {
            continue;
        }

        // this loop won't work; there should only be one task per viewer
        for (layout_transform, layout_handle) in layout_roots.iter() {
            let Some(layout) = layouts.get(layout_handle) else {
                continue;
            };

            let task: Task<SampledVertices> = task_pool.spawn(async move {
                // TODO build vertices; eventually using rayon
                todo!();
            });

            todo!();
        }
    }

    //
}

fn finish_vertex_sampling_tasks(
    //
    mut commands: Commands,

    mut tasks: Query<(
        Entity,
        &mut VertexSamplingTask,
        &mut pipeline::PolylineVertices,
    )>,
) {
    // move task buffer data into `RawBufferVec`... so not `SampledVertices` here
    //
    // the
}

// copy from `SampledVertices` into GPU buffer...
// fn copy_vertices_to_gpu(
// )

// fn render_sampled_vertices(
//     //
//     mut commands: Commands,
// ) {
//     //
// }

#[derive(Debug)]
enum VertexSamplingError {
    OutOfMemory {
        estimated_extra_bp: Option<u64>,
        successful_target_range: std::ops::Range<u64>,
    },
}
impl std::fmt::Display for VertexSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VertexSamplingError::OutOfMemory {
                estimated_extra_bp,
                successful_target_range,
            } => {
                write!(f, "Cigar vertex buffer full. Successfully sampled range {:?}. Estimated extra {:?} bp",
                    successful_target_range, estimated_extra_bp)
            }
        }
    }
}
impl std::error::Error for VertexSamplingError {}

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct VertexData {
    p0: [f32; 2],
    p1: [f32; 2],
    z: f32,
    color: u32,
}

// samples the `alignment` to produce screen-space
// line segments in `buffer`
fn sample_segments_from_alignment(
    seq_pair_offset: impl Into<[f64; 2]>,
    alignment: &crate::Alignment,
    view: &crate::view::View,
    canvas_size: impl Into<[f32; 2]>,
    buffer: &mut [VertexData],
) -> Result<usize, VertexSamplingError> {
    let [o_x0, o_y0] = seq_pair_offset.into();
    let [c_width, c_height] = canvas_size.into();
    //

    // AI START
    let loc = &alignment.location;
    let tgt_len = loc.target_range.end - loc.target_range.start;

    let al_min = o_x0 + loc.target_range.start as f64;
    let al_max = al_min + tgt_len as f64;

    let cal_min = view.x_min.clamp(al_min, al_max) as u64;
    let cal_max = view.x_max.clamp(al_min, al_max) as u64;
    if cal_min == cal_max {
        return Ok(0);
    }

    let loc_min = cal_min.checked_sub(o_x0 as u64).unwrap_or_default();
    let loc_max = cal_max.checked_sub(o_x0 as u64).unwrap_or_default();

    if loc_min == loc_max {
        return Ok(0);
    }
    // AI END

    // let vis_target_range: std::ops::Range<u64> = todo!();
    let vis_target_range = loc_min..loc_max;
    let target_start = vis_target_range.start;

    let mut buffer_offset = 0;

    let cg_iter = alignment.iter_target_range(vis_target_range);
    let mut cmd_iter = cigar_sampling::CigarScreenPathStrokeIter::new(
        *view,
        UVec2::new(c_width as u32, c_height as u32),
        [o_x0, o_y0],
        cg_iter,
    );

    let bp_per_px = view.width() / c_width as f64;
    // let mut path_start: Option<[u64; 2]> = None;
    let mut path_start = None;

    while let Some(path_cmd) = cmd_iter.emit_next() {
        match path_cmd {
            zeno::Command::MoveTo(p0) => {
                path_start = Some(p0);
            }
            zeno::Command::LineTo(p1) => {
                if let Some(p0) = path_start.as_mut() {
                    buffer[buffer_offset] = VertexData {
                        p0: [p0.x, p0.y],
                        p1: [p1.x, p1.y],
                        z: 0.0,
                        color: 0xFF0000FF,
                    };
                    buffer_offset += 1;
                }
            }
            _ => (),
        }
    }

    /*
    // TODO then iterate the cigar...
    for item in alignment.iter_target_range(vis_target_range) {

        // map to screenspace

        // need to track start of each line to emit the whole segment

        // emit solid line for each consecutive non-indel op
        // - merge/skip indels depending on scale and state

        // if it's a mismatch, and the scale is appropriate, emit a red line at
        // a higher z-level

        // emit into `buffer[buffer_offset]` & increment offset
    }
    */

    Ok(buffer_offset)
}

// vertex buffer for the screen-space triangulated alignment vertices
#[derive(Component)]
struct TriangulatedVertices {
    buffer: wgpu::Buffer,
}

mod pipeline {
    use super::*;
    use bevy::{prelude::*, render::render_resource::RawBufferVec};

    pub(super) struct SampledPolylinePipelinePlugin;

    impl Plugin for SampledPolylinePipelinePlugin {
        fn build(&self, app: &mut App) {
            app.add_plugins(ExtractComponentPlugin::<PolylineVertices>::default());
        }

        fn finish(&self, app: &mut App) {
            // app.init_resource::<PolylinePipeline>();
            let render_app = app.sub_app_mut(RenderApp);
            render_app.init_resource::<PolylinePipeline>();
            // // .add_systems(Render, ())

            // todo!();
        }
    }

    #[derive(Component)]
    pub(super) struct PolylineVertices {
        buffer: RawBufferVec<VertexData>,
        instances: std::ops::Range<u32>,
    }

    #[derive(Component)]
    struct ExtractedVertexBuffer {
        buffer: Buffer,
        instances: std::ops::Range<u32>,
    }

    impl ExtractComponent for PolylineVertices {
        type QueryData = &'static PolylineVertices;

        type QueryFilter = ();

        type Out = ExtractedVertexBuffer;

        fn extract_component(
            item: bevy::ecs::query::QueryItem<'_, Self::QueryData>,
        ) -> Option<Self::Out> {
            let buffer = item.buffer.buffer()?;

            Some(ExtractedVertexBuffer {
                buffer: buffer.clone(),
                instances: item.instances.clone(),
            })
        }
    }

    impl PolylineVertices {
        pub(super) fn new() -> Self {
            Self {
                buffer: RawBufferVec::new(BufferUsages::VERTEX | BufferUsages::COPY_DST),
                instances: 0..0,
            }
        }
    }

    #[derive(Component)]
    pub(super) struct PolylineUniforms {
        config_buffer: UniformBuffer<PolylineConfig>,
        model_buffer: UniformBuffer<Mat4>,

        config_bind_group: BindGroup,
        model_bind_group: BindGroup,
    }

    // #[derive(Component)]
    // pub(super) struct PolylineVertexBuffer {
    //     buffer: Buffer,
    //     instances: std::ops::Range<u32>,
    // }

    /*

    fn copy_vertices_to_gpu(

    ) {

    }

    */

    fn queue_draw(
        mut commands: Commands,

        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        pipeline_cache: Res<PipelineCache>,
        pipeline: Res<PolylinePipeline>,

        gpu_images: Res<RenderAssets<GpuImage>>,

        polylines: Query<
            (
                Entity,
                &PolylineVertices,
                &PolylineUniforms,
                &BackRenderTarget,
            ),
            Without<Rendering>,
        >,
        // gpu_vertices: Res<RenderAssets<GpuAlignmentVertices>>,
        // gpu_materials: Res<RenderAssets<GpuAlignmentPolylineMaterial>>,
    ) {
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
            return;
        };

        // draw the sampled vertices; all that's needed is the vertex buffer and bind group(s)

        for (entity, vertices, uniforms, render_tgt) in polylines.iter() {
            let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sampled Vertices Renderer".into()),
            });

            let Some((tgt_img, depth_img)) = gpu_images
                .get(&render_tgt.0.color)
                .zip(gpu_images.get(&render_tgt.0.depth))
            else {
                continue;
            };

            {
                if vertices.instances.len() == 0 {
                    continue;
                }

                let Some(vx_buffer) = vertices.buffer.buffer() else {
                    continue;
                };

                let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Sampled Vertices Pass".into()),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &tgt_img.texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
                pass.set_bind_group(0, &uniforms.config_bind_group, &[]);
                pass.set_bind_group(1, &uniforms.model_bind_group, &[]);

                pass.set_vertex_buffer(0, wgpu::Buffer::slice(vx_buffer, ..));
                pass.draw(0..6, vertices.instances.clone());
            }
        }
    }

    #[derive(Resource)]
    pub(super) struct PolylinePipeline {
        proj_config_layout: BindGroupLayout,
        // color_scheme_layout: BindGroupLayout,
        model_layout: BindGroupLayout,

        pipeline: CachedRenderPipelineId,

        shader: Handle<Shader>,
    }

    #[derive(ShaderType, Clone, Copy)]
    struct PolylineConfig {
        line_width: f32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

    impl FromWorld for PolylinePipeline {
        fn from_world(world: &mut World) -> Self {
            let render_device = world.resource::<RenderDevice>();

            use bevy::render::render_resource::{self, binding_types};

            let proj_config_layout = render_device.create_bind_group_layout(
                "SampledAlignmentRenderConfig",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::VERTEX,
                    (
                        binding_types::uniform_buffer::<Mat4>(false),
                        binding_types::uniform_buffer::<PolylineConfig>(false),
                    ),
                ),
            );

            // let color_scheme_layout = render_device.create_bind_group_layout(
            //     "AlignmentColorScheme",
            //     &BindGroupLayoutEntries::sequential(
            //         ShaderStages::VERTEX,
            //         (binding_types::uniform_buffer::<GpuAlignmentColorScheme>(
            //             false,
            //         ),),
            //     ),
            // );

            let model_layout = render_device.create_bind_group_layout(
                "SampledAlignmentModel",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::VERTEX,
                    (binding_types::uniform_buffer::<Mat4>(false),),
                ),
            );

            let shader = Shader::from_wgsl(
                include_str!("../../../assets/shaders/lines_vertex_color.wgsl"),
                "internal/shaders/lines_vertex_color.wgsl",
            );
            let shader = world.resource::<AssetServer>().add(shader);
            let pipeline_cache = world.resource::<PipelineCache>();

            let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("Sampled Alignment Render Pipeline".into()),
                layout: vec![
                    proj_config_layout.clone(),
                    // color_scheme_layout.clone(),
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
                                format: wgpu::VertexFormat::Float32,
                                offset: 16,
                                shader_location: 2,
                            },
                            render_resource::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 20,
                                shader_location: 3,
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
                // color_scheme_layout,
                model_layout,
                pipeline,
                shader,
            }
        }
    }
}
