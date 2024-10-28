use bevy::{prelude::*, tasks::Task};
use pipeline::PolylineVertexBuffer;

use crate::app::alignments::layout::SeqPairLayout;

use super::*;

pub struct SampledAlignmentRendererPlugin;

impl Plugin for SampledAlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        todo!();

        //
    }
}

#[derive(Component)]
struct SampledAlignmentViewer {
    view: Option<crate::view::View>,

    last_rendered: Option<RenderParams>,
    last_vertex_params: Option<RenderParams>,
}

fn setup_gpu_resources(//
) {

    //
}

#[derive(Component)]
struct VertexSamplingTask(Task<PolylineVertexBuffer>);

fn spawn_vertex_sampling_tasks(
    //
    layout_roots: Query<(&Transform, &Handle<SeqPairLayout>)>,
) {

    //
}

fn update_vertex_buffer(//
) {

    //
}

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
    use bevy::prelude::*;

    pub(super) struct SampledPolylinePipelinePlugin;

    impl Plugin for SampledPolylinePipelinePlugin {
        fn build(&self, app: &mut App) {
            todo!()
        }

        fn finish(&self, app: &mut App) {
            let render_app = app.sub_app_mut(RenderApp);
            render_app.init_resource::<PolylinePipeline>();
            // .add_systems(Render, ())

            todo!();
        }
    }

    #[derive(Component)]
    pub(super) struct PolylineVertexBuffer {
        vertex_buffer: std::sync::Arc<wgpu::Buffer>,
        instances: std::ops::Range<u32>,
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
