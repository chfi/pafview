use std::sync::atomic::{AtomicBool, AtomicU8};

use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
    utils::tracing,
};
use pipeline::{PolylineConfig, PolylineModel, PolylineProjection, PolylineVertices};
use wgpu::BufferUsages;

use crate::app::alignments::layout::SeqPairLayout;

use super::*;

pub struct SampledAlignmentRendererPlugin;

impl Plugin for SampledAlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<BackRenderTarget>::default())
            .add_plugins(ExtractComponentPlugin::<RenderOperation>::default())
            .add_plugins(pipeline::SampledPolylinePipelinePlugin)
            .add_systems(Startup, spawn_main_sampled_alignment_viewer)
            .add_systems(
                Update,
                (
                    update_alignment_viewer_params,
                    spawn_vertex_sampling_tasks,
                    (
                        update_line_width,
                        update_vertex_transform,
                        update_viewer_sprite_visibility,
                        update_viewer_sprite_transform,
                        update_projection,
                    ),
                )
                    .chain(),
            )
            .add_systems(
                PreUpdate,
                (
                    finish_render_operation,
                    finish_vertex_sampling_tasks,
                    trigger_render_operation,
                    resize_alignment_viewer_back_image,
                )
                    .chain(),
            );
        // .add_systems(PostUpdate, (trigger_render_operation,).chain());

        //
    }
}

#[derive(Component, Default)]
struct SampledAlignmentViewer {
    view: Option<crate::view::View>,

    last_rendered: Option<RenderParams>,

    last_sampled_at: Option<std::time::Instant>,
    last_rendered_at: Option<std::time::Instant>,
}

struct SampledVertices {
    buffer_data: Vec<VertexData>,
    sampling_params: VertexSamplingParams,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
struct VertexSamplingParams {
    view: crate::view::View,
    canvas_size: UVec2,
}

fn spawn_main_sampled_alignment_viewer(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 1,
    };

    let mut image = Image {
        texture_descriptor: wgpu::TextureDescriptor {
            label: "SampledViewer Color 1".into(),
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
    let mut back_image = front_image.clone();
    back_image.texture_descriptor.label = "SampledView Color 2".into();

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
    // let back_vertices = BackVertexBuffer(PolylineVertices::new());

    commands
        .spawn((
            SampledAlignmentViewer::default(),
            front_vertices,
            // back_vertices,
            SpriteBundle::default(),
            PolylineProjection {
                proj: Mat4::IDENTITY,
            },
            PolylineConfig::new(5.0),
            PolylineModel {
                model: Mat4::IDENTITY,
            }, // SampledVertices::default(),
            RenderLayers::layer(1),
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

// #[derive(Component)]
// struct BackVertexBuffer(PolylineVertices);

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
}

fn resize_alignment_viewer_back_image(
    mut images: ResMut<Assets<Image>>,
    // mut viewers: Query<&mut BackRenderTarget, (With<SampledAlignmentViewer>, Without<RenderOperation>)>,
    mut viewers: Query<
        (&mut SampledAlignmentViewer, &BackRenderTarget),
        (Without<RenderOperation>),
    >,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    let win_size = window.physical_size();

    let extent = wgpu::Extent3d {
        width: win_size.x,
        height: win_size.y,
        depth_or_array_layers: 1,
    };

    for (mut viewer, render_tgt) in viewers.iter_mut() {
        let mut resized = false;
        for img_handle in [&render_tgt.0.color, &render_tgt.0.depth] {
            if let Some(img) = images.get_mut(img_handle) {
                if img.size() != win_size {
                    img.resize(extent);
                    resized = true;
                }
            }
        }
        if resized {
            viewer.last_rendered = None;
        }
    }
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

#[tracing::instrument(skip_all)]
fn spawn_vertex_sampling_tasks(
    mut commands: Commands,

    alignments: Res<crate::Alignments>,
    layouts: Res<Assets<SeqPairLayout>>,

    layout_roots: Query<(&Transform, &Handle<SeqPairLayout>)>,

    viewers: Query<
        (
            Entity,
            &SampledAlignmentViewer,
            Option<&VertexSamplingParams>,
        ),
        Without<VertexSamplingTask>,
    >,

    windows: Query<&Window>,
    frame_count: Res<bevy::core::FrameCount>,
    // keyboard: Res<ButtonInput<KeyCode>>,
) {
    // to give the window time to resize etc.
    if frame_count.0 < 3 {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    let canvas_size_u = window.physical_size();
    let canvas_size = canvas_size_u.as_vec2();

    let task_pool = AsyncComputeTaskPool::get();

    // let need_new_vertices = keyboard.just_pressed(KeyCode::Enter);

    for (viewer_ent, viewer, last_params) in viewers.iter() {
        let Some(next_view) = viewer.view else {
            continue;
        };

        if let Some(last_time) = viewer.last_sampled_at {
            if last_time.elapsed().as_millis() < 100 {
                continue;
            }
        }

        // TODO: spawn task if `next_view` has escaped bounds of the sampling
        // params in `vertices`, or if scale has changed "enough"
        let need_new_vertices = if let Some(sampled_params) = last_params.as_ref() {
            let s_view: crate::view::View = sampled_params.view;

            // s_view != next_view || canvas_size_u != sampled_params.canvas_size
            // || bp_per_px != sampled_params.scale()

            let view_out_of_bounds = s_view.x_min > next_view.x_max
                || s_view.x_max < next_view.x_min
                || s_view.y_min > next_view.y_max
                || s_view.y_max < next_view.y_min;

            let rel_scale = next_view.width() / s_view.width();
            let beyond_scale_limit = rel_scale < 0.5 || rel_scale > 2.0;

            if view_out_of_bounds || beyond_scale_limit {
                dbg!((view_out_of_bounds, beyond_scale_limit));
            }

            view_out_of_bounds || beyond_scale_limit
        } else {
            true
        };

        if !need_new_vertices {
            continue;
        }

        let placed_layouts = layout_roots
            .iter()
            .filter_map(|(tform, handle)| {
                let layout = layouts.get(handle)?.clone();
                Some((*tform, layout))
            })
            .collect::<Vec<_>>();

        let alignments_vec = alignments.alignments.clone();
        let alignment_ixs = alignments.indices.clone();

        let params = VertexSamplingParams {
            view: next_view,
            canvas_size: canvas_size_u,
            // scale: bp_per_px,
        };

        let task = task_pool.spawn(async move {
            use rayon::prelude::*;

            let (data_send, data_recv) = crossbeam::channel::unbounded::<VertexData>();

            let t0 = std::time::Instant::now();
            let alignments = placed_layouts.par_iter().flat_map(|(_tform, layout)| {
                layout
                    .layout_qbvh
                    .tiles_in_rect(params.view.center(), params.view.size() * 0.5)
                    .into_par_iter()
                    .filter_map(|seq_pair| {
                        let aabb = layout.aabbs.get(&seq_pair)?;
                        let al_ixs = alignment_ixs.get(&(seq_pair.target, seq_pair.query))?;
                        let seq_pair_offset = [aabb.mins.x, aabb.mins.y];

                        Some((seq_pair_offset, al_ixs))
                    })
                    .flat_map(|(offset, al_indices)| {
                        let al_vec = &alignments_vec;
                        al_indices.par_iter().filter_map(move |ix| {
                            let al = al_vec.get(*ix)?;
                            Some((offset, al))
                        })
                    })
            });

            alignments.for_each_with(
                (data_send, Vec::<VertexData>::new()),
                |(send, ref mut vx_data), (seq_pair_offset, alignment)| {
                    vx_data.clear();
                    if let Err(_err) = sample_segments_from_alignment(
                        seq_pair_offset,
                        alignment,
                        &next_view,
                        canvas_size,
                        vx_data,
                    ) {
                        // log
                    } else {
                        vx_data.iter().for_each(|&data| {
                            send.send(data).unwrap();
                        });
                    }
                },
            );

            let vertex_data = data_recv.iter().collect::<Vec<_>>();

            println!("sampled vertices in {} ms", t0.elapsed().as_millis());

            SampledVertices {
                buffer_data: vertex_data,
                sampling_params: params,
            }
        });

        tracing::info!("spawning vertex sampling task");
        // println!("spawned vertex sampling task: {params:?}");

        commands
            .entity(viewer_ent)
            .insert((params, VertexSamplingTask { task }));
    }

    //
}

#[tracing::instrument(skip_all)]
fn finish_vertex_sampling_tasks(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    //
    mut commands: Commands,

    mut viewers: Query<
        (
            Entity,
            &mut SampledAlignmentViewer,
            &mut VertexSamplingTask,
            &mut pipeline::PolylineVertices,
            Has<RenderOperation>,
            // &mut PolylineModel,
        ),
        // Without<RenderOperation>,
    >,
) {
    // move task buffer data into `RawBufferVec`... so not `SampledVertices` here
    //
    // the

    for (viewer_ent, mut viewer, mut task, mut vertices, has_render_op) in viewers.iter_mut() {
        if !task.task.is_finished() {
            continue;
        }

        let Some(mut result) = bevy::tasks::block_on(bevy::tasks::poll_once(&mut task.task)) else {
            commands.entity(viewer_ent).remove::<VertexSamplingTask>();
            continue;
        };

        std::mem::swap(vertices.buffer.values_mut(), &mut result.buffer_data);

        let inst_count = vertices.buffer.values().len();

        dbg!(has_render_op);

        // println!("sampled {} vertices", vertices.buffer.values().len());
        // println!("{:#?}", vertices.buffer.values());
        commands
            .entity(viewer_ent)
            .insert(result.sampling_params)
            .remove::<VertexSamplingTask>();

        viewer.last_sampled_at = Some(std::time::Instant::now());

        vertices.instances = 0..inst_count as u32;
        {
            let span = info_span!("Vertex buffer write");
            let _guard = span.enter();
            info!("vertices.buffer.reserve({inst_count})");
            vertices.buffer.reserve(inst_count, &render_device);
            info!("vertices.buffer.write_buffer({inst_count})");
            vertices.buffer.write_buffer(&render_device, &render_queue);
        }

        // viewer.last_vertex_params =

        // model.model = Mat4::IDENTITY;
    }
}

fn update_line_width(
    app_config: Res<crate::AppConfig>,
    mut viewers: Query<&mut PolylineConfig, With<SampledAlignmentViewer>>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let win_size = window.physical_size().as_vec2();

    for mut config in viewers.iter_mut() {
        let width = app_config.alignment_line_width / win_size.x;
        config.line_width = width;
    }
}

fn update_projection(
    mut viewers: Query<(
        // Entity,
        // &VertexSamplingTask,
        &SampledAlignmentViewer,
        // &mut pipeline::PolylineVertices,
        &mut PolylineProjection,
    )>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    let size = window.physical_size().as_vec2();

    for (_viewer, mut proj) in viewers.iter_mut() {
        // if let Some(view) = viewer.view {
        // let hw = size.x * 0.5;
        // let hh = size.y * 0.5;
        // println!("updating projection with {size:?}");
        let proj_uv =
            ultraviolet::projection::orthographic_wgpu_dx(0.0, size.x, size.y, 0.0, 0.1, 10.0);
        // ultraviolet::projection::orthographic_wgpu_dx(0.0, size.x, 0.0, size.y, 0.1, 10.0);
        let mat = Mat4::from_cols_array(proj_uv.as_array());
        proj.proj = mat;
    }
}

#[tracing::instrument(skip_all)]
fn update_vertex_transform(
    mut viewers: Query<(
        &SampledAlignmentViewer,
        &VertexSamplingParams,
        // &PolylineVertices,
        &mut PolylineModel,
    )>,
    mut last_scale: Local<Vec3>,
) {
    for (viewer, sampled, mut model) in viewers.iter_mut() {
        let Some(next_view) = viewer.view else {
            continue;
        };

        let win_size = sampled.canvas_size.as_vec2();
        let last_view = sampled.view;
        let old_mid = last_view.center();
        let new_mid = next_view.center();

        let world_delta = new_mid - old_mid;
        let norm_delta = world_delta / next_view.size();

        let w_rat = last_view.width() / next_view.width();
        let h_rat = last_view.height() / next_view.height();

        let w_rat_ = next_view.width() / last_view.width();
        let h_rat_ = next_view.height() / last_view.height();
        let screen_delta =
            norm_delta.to_f32() * [w_rat_ as f32 * win_size.x, h_rat_ as f32 * win_size.y].as_uv();
        let mut center = Transform::from_translation(Vec3::new(win_size.x, win_size.y, 0.0) * 0.5);
        let translate =
            Transform::from_translation(Vec3::new(-screen_delta.x, screen_delta.y, 0.0));
        let scale_vec = Vec3::new(w_rat as f32, h_rat as f32, 1.0);
        let scale = Transform::from_scale(scale_vec);
        println!("{scale_vec:?} vs {last_scale:?}");
        *last_scale = scale_vec;

        let mut transform = center.mul_transform(scale);
        center.translation *= -1.0;
        transform = transform.mul_transform(center);
        transform = transform.mul_transform(translate);

        model.model = transform.compute_matrix();
    }
}

fn update_viewer_sprite_visibility(mut viewers: Query<(&mut Visibility, &SampledAlignmentViewer)>) {
    for (mut vis, viewer) in viewers.iter_mut() {
        let Some(bp_per_px) = viewer.last_rendered.map(|p| p.scale()) else {
            continue;
        };

        // println!("setting visibility");
        if bp_per_px < 1.0 {
            *vis = Visibility::Hidden;
        } else {
            *vis = Visibility::Inherited;
        }
    }
}

fn update_viewer_sprite_transform(
    mut viewers: Query<(
        &SampledAlignmentViewer,
        &VertexSamplingParams,
        &mut Transform,
        &mut Sprite,
    )>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    // let win_size = window.resolution.size();
    let dpi_scale = window.resolution.scale_factor();

    for (viewer, _vertex_params, _transform, mut sprite) in viewers.iter_mut() {
        let Some(rendered) = viewer.last_rendered else {
            continue;
        };

        let img_size = rendered.canvas_size.as_vec2();
        sprite.custom_size = Some(img_size / dpi_scale);

        // NB: sprite transform disabled as it makes things jumpy right now
        /*
        let Some(next_view) = viewer.view else {
            continue;
        };

        let last_view = rendered.view;

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
        */
    }
}

#[derive(Debug, Clone, Component, ExtractComponent)]
struct RenderOperation {
    view: crate::view::View,
    canvas_size: UVec2,
    finished: Arc<AtomicU8>,
}

impl RenderOperation {
    const STATE_READY: u8 = 0;
    const STATE_SUBMITTED: u8 = 1;
    const STATE_FINISHED: u8 = 2;
    const STATE_ERROR: u8 = 3;
}

#[tracing::instrument(skip_all)]
fn trigger_render_operation(
    mut commands: Commands,

    viewers: Query<
        (Entity, &SampledAlignmentViewer),
        (Without<RenderOperation>, With<VertexSamplingParams>),
    >,
    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let canvas_size = window.physical_size();

    for (viewer_ent, viewer) in viewers.iter() {
        let Some(view) = viewer.view else {
            continue;
        };

        // if let Some(last_time) = viewer.last_rendered_at {
        //     if last_time.elapsed().as_millis() < 20 {
        //         continue;
        //     }
        // }

        let need_render = Some(view) != viewer.last_rendered.map(|p| p.view);

        if !need_render {
            continue;
        }
        // println!("triggering re-render");

        commands.entity(viewer_ent).insert(RenderOperation {
            view,
            canvas_size,
            finished: Arc::new(0.into()),
            // finished: Arc::new(false.into()),
        });
    }
}

#[tracing::instrument(skip_all)]
fn finish_render_operation(
    mut commands: Commands,
    mut viewers: Query<(
        Entity,
        &mut SampledAlignmentViewer,
        &RenderOperation,
        &mut Handle<Image>,
        &mut FrontRenderTarget,
        &mut BackRenderTarget,
    )>,
) {
    for (viewer_ent, mut viewer, render_op, mut sprite_img, mut front_tgts, mut back_tgts) in
        viewers.iter_mut()
    {
        let render_state = render_op
            .finished
            .load(std::sync::atomic::Ordering::Relaxed);
        if render_state < RenderOperation::STATE_FINISHED {
            continue;
        } else if render_state == RenderOperation::STATE_ERROR {
            // TODO: maybe want to do something more here, but not sure
            commands.entity(viewer_ent).remove::<RenderOperation>();
            continue;
        }
        /*
        if !render_op
            .finished
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            continue;
        }
        */

        viewer.last_rendered = Some(RenderParams {
            view: render_op.view,
            canvas_size: render_op.canvas_size,
        });
        viewer.last_rendered_at = Some(std::time::Instant::now());
        std::mem::swap(&mut front_tgts.0, &mut back_tgts.0);
        *sprite_img = front_tgts.0.color.clone_weak();
        // println!("rendering complete");

        commands.entity(viewer_ent).remove::<RenderOperation>();
    }
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
#[tracing::instrument(skip_all)]
fn sample_segments_from_alignment(
    seq_pair_offset: impl Into<[f64; 2]>,
    alignment: &crate::Alignment,
    view: &crate::view::View,
    canvas_size: impl Into<[f32; 2]>,
    buffer: &mut Vec<VertexData>,
    // buffer: &mut [VertexData],
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

    let mut buffer_offset = 0;

    while let Some(path_cmd) = cmd_iter.emit_next() {
        match path_cmd {
            zeno::Command::MoveTo(p0) => {
                path_start = Some(p0);
            }
            zeno::Command::LineTo(p1) => {
                if let Some(p0) = path_start.as_mut() {
                    let half_bp = (0.5 / bp_per_px) as f32;
                    let vertex = VertexData {
                        p0: [p0.x + half_bp, p0.y + half_bp],
                        p1: [p1.x + half_bp, p1.y + half_bp],
                        z: 0.5,
                        color: 0xFF000000, // ABGR
                    };
                    buffer.push(vertex);
                    buffer_offset += 1;
                }
            }
            _ => (),
        }
    }

    Ok(buffer_offset)
}

mod pipeline {
    use super::*;
    use bevy::render::{
        extract_component::{ComponentUniforms, DynamicUniformIndex, UniformComponentPlugin},
        render_resource::RawBufferVec,
    };

    pub(super) struct SampledPolylinePipelinePlugin;

    impl Plugin for SampledPolylinePipelinePlugin {
        fn build(&self, app: &mut App) {
            app.add_plugins(ExtractComponentPlugin::<PolylineVertices>::default())
                .add_plugins((
                    ExtractComponentPlugin::<PolylineModel>::default(),
                    UniformComponentPlugin::<PolylineModel>::default(),
                ))
                .add_plugins((
                    ExtractComponentPlugin::<PolylineConfig>::default(),
                    UniformComponentPlugin::<PolylineConfig>::default(),
                ))
                .add_plugins((
                    ExtractComponentPlugin::<PolylineProjection>::default(),
                    UniformComponentPlugin::<PolylineProjection>::default(),
                ));
        }

        fn finish(&self, app: &mut App) {
            let render_app = app.sub_app_mut(RenderApp);

            render_app
                .init_resource::<PolylinePipeline>()
                .add_systems(Render, queue_draw.in_set(RenderSet::Render));
        }
    }

    #[derive(Component)]
    pub(super) struct PolylineVertices {
        pub(super) buffer: RawBufferVec<VertexData>,
        pub(super) instances: std::ops::Range<u32>,
    }

    #[derive(Component)]
    pub struct ExtractedVertexBuffer {
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

    #[derive(Resource)]
    pub(super) struct PolylinePipeline {
        proj_config_layout: BindGroupLayout,
        model_layout: BindGroupLayout,

        pipeline: CachedRenderPipelineId,
        shader: Handle<Shader>,
    }

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent)]
    pub(super) struct PolylineModel {
        pub(super) model: Mat4,
    }

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent)]
    pub(super) struct PolylineConfig {
        pub(super) line_width: f32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
    }

    impl PolylineConfig {
        pub(super) fn new(line_width: f32) -> Self {
            Self {
                line_width,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }
        }
    }

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent)]
    pub(super) struct PolylineProjection {
        pub(super) proj: Mat4,
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
                        binding_types::uniform_buffer::<Mat4>(true),
                        binding_types::uniform_buffer::<PolylineConfig>(true),
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
                    (binding_types::uniform_buffer::<Mat4>(true),),
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
                        array_stride: std::mem::size_of::<VertexData>() as u64,
                        // array_stride: 6 * std::mem::size_of::<u32>() as u64,
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

    fn queue_draw(
        mut commands: Commands,

        render_device: Res<RenderDevice>,
        render_queue: Res<RenderQueue>,
        pipeline_cache: Res<PipelineCache>,
        pipeline: Res<PolylinePipeline>,

        gpu_images: Res<RenderAssets<GpuImage>>,

        projections: Res<ComponentUniforms<PolylineProjection>>,
        configs: Res<ComponentUniforms<PolylineConfig>>,
        models: Res<ComponentUniforms<PolylineModel>>,

        polylines: Query<(
            Entity,
            &ExtractedVertexBuffer,
            // &PolylineVertices,
            (
                &DynamicUniformIndex<PolylineProjection>,
                &DynamicUniformIndex<PolylineConfig>,
                &DynamicUniformIndex<PolylineModel>,
            ),
            // &PolylineModel,
            // &PolylineConfi
            // &PolylineBindGroups,
            &RenderOperation,
            &BackRenderTarget,
        )>,
        // gpu_vertices: Res<RenderAssets<GpuAlignmentVertices>>,
        // gpu_materials: Res<RenderAssets<GpuAlignmentPolylineMaterial>>,
    ) {
        let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline) else {
            return;
        };

        // draw the sampled vertices; all that's needed is the vertex buffer and bind group(s)
        // dbg!();

        // if polylines.is_empty() {
        // println!("nothing to render!");
        // }

        for (entity, vertices, uniform_indices, render_op, render_tgt) in polylines.iter() {
            let render_state = render_op
                .finished
                .load(std::sync::atomic::Ordering::Relaxed);

            if render_state != RenderOperation::STATE_READY {
                // skip as rendering has already begun
                continue;
            }

            let mut cmds = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sampled Vertices Renderer".into()),
            });

            let Some((tgt_img, depth_img)) = gpu_images
                .get(&render_tgt.0.color)
                .zip(gpu_images.get(&render_tgt.0.depth))
            else {
                dbg!();
                render_op.finished.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            // println!("rendering to image size {:?}", tgt_img.size);

            if vertices.instances.len() == 0 {
                dbg!();
                render_op.finished.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            }

            let vx_buffer = &vertices.buffer;
            // let Some(vx_buffer) = vertices.buffer.buffer() else {
            //     dbg!();
            //     continue;
            // };

            // create bind groups

            let Some((proj_binding, cfg_binding)) = projections
                .uniforms()
                .binding()
                .zip(configs.uniforms().binding())
            else {
                dbg!();
                render_op.finished.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            let Some(model_binding) = models.uniforms().binding() else {
                dbg!();
                render_op.finished.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            render_op.finished.store(
                RenderOperation::STATE_SUBMITTED,
                std::sync::atomic::Ordering::Relaxed,
            );

            let group_0 = render_device.create_bind_group(
                None,
                &pipeline.proj_config_layout,
                &BindGroupEntries::sequential((proj_binding, cfg_binding)),
            );
            let group_1 = render_device.create_bind_group(
                None,
                &pipeline.model_layout,
                &BindGroupEntries::sequential((model_binding,)),
            );

            let (proj_ix, cfg_ix, model_ix) = uniform_indices;

            {
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

                pass.set_pipeline(render_pipeline);
                pass.set_bind_group(0, &group_0, &[proj_ix.index(), cfg_ix.index()]);
                pass.set_bind_group(1, &group_1, &[model_ix.index()]);

                pass.set_vertex_buffer(0, wgpu::Buffer::slice(vx_buffer, ..));
                pass.draw(0..6, vertices.instances.clone());
            }

            render_queue.0.submit([cmds.finish()]);
            // dbg!();

            // start render

            let finished = render_op.finished.clone();
            render_queue.0.on_submitted_work_done(move || {
                // finished.store(true, std::sync::atomic::Ordering::Relaxed);
                finished.store(
                    RenderOperation::STATE_FINISHED,
                    std::sync::atomic::Ordering::Relaxed,
                );
            });
        }
    }
}
