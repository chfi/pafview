use std::sync::atomic::AtomicU8;

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
        app.register_type::<SampledAlignmentViewer>()
            .register_type::<RenderParams>()
            .register_type::<RenderOperation>();

        app.add_plugins(ExtractComponentPlugin::<BackRenderTarget>::default())
            .add_plugins(ExtractComponentPlugin::<RenderOperation>::default())
            .add_plugins(pipeline::SampledPolylinePipelinePlugin)
            .add_systems(Startup, spawn_main_sampled_alignment_viewer)
            .add_systems(
                Update,
                (
                    finish_vertex_sampling_tasks,
                    update_alignment_viewer_params,
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
                    // finish_vertex_sampling_tasks,
                    // spawn_vertex_sampling_tasks,
                    finish_render_operation,
                    spawn_vertex_sampling_tasks,
                    // trigger_render_operation,
                    resize_alignment_viewer_back_image,
                )
                    .chain(),
            );

        // app.add_plugins(debug::DebugPlugin);

        // app.add_systems(Update, viz_image_handle);
        // .add_systems(PostUpdate, finish_vertex_sampling_tasks);
        app.add_systems(PostUpdate, trigger_render_operation);

        //
    }
}

#[derive(Component, Default, Reflect)]
struct SampledAlignmentViewer {
    view: Option<crate::view::View>,

    last_rendered: Option<RenderParams>,
    last_rendered_sampling_params: Option<VertexSamplingParams>,

    last_sampled_at: Option<std::time::Instant>,
    last_rendered_at: Option<std::time::Instant>,
}

struct SampledVertices {
    buffer_data: Vec<VertexData>,
    sampling_params: VertexSamplingParams,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Reflect)]
struct VertexSamplingParams {
    view: crate::view::View,
    canvas_size: Vec2,
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

    commands
        .spawn((
            SampledAlignmentViewer::default(),
            PolylineVertices::new(),
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
    mut viewers: Query<(&mut SampledAlignmentViewer, &BackRenderTarget), Without<RenderOperation>>,

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

            // if view_out_of_bounds || beyond_scale_limit {
            //     dbg!((view_out_of_bounds, beyond_scale_limit));
            // }

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
            canvas_size,
            // scale: bp_per_px,
        };

        let task = task_pool.spawn(async move {
            use rayon::prelude::*;

            let (data_send, data_recv) = crossbeam::channel::unbounded::<VertexData>();

            let t0 = std::time::Instant::now();
            // TODO: actually use the root transform
            let alignments = placed_layouts.par_iter().flat_map(|(transform, layout)| {
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
            futures_time::task::sleep(futures_time::time::Duration::from_millis(500)).await;

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
            .insert(VertexSamplingTask { task });
        // .insert((params, VertexSamplingTask { task }));
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
            // Option<&RenderOperation>,
            // Has<RenderOperation>,
            // &mut PolylineModel,
        ),
        // Without<RenderOperation>,
    >,
) {
    // move task buffer data into `RawBufferVec`... so not `SampledVertices` here
    //
    // the

    for (viewer_ent, mut viewer, mut task, mut vertices) in viewers.iter_mut() {
        /*
        if let Some(state) =
            render_op.map(|s| s.finished.load(std::sync::atomic::Ordering::Relaxed))
        {
            if state == RenderOperation::STATE_READY {
                continue;
            }
            // if state < RenderOperation::STATE_FINISHED {
            //     continue;
            // }
        }
        */

        if !task.task.is_finished() {
            continue;
        }

        let Some(mut result) = bevy::tasks::block_on(bevy::tasks::poll_once(&mut task.task)) else {
            commands.entity(viewer_ent).remove::<VertexSamplingTask>();
            continue;
        };

        std::mem::swap(vertices.buffer.values_mut(), &mut result.buffer_data);

        let inst_count = vertices.buffer.values().len();

        // dbg!(has_render_op);

        // println!("sampled {} vertices", vertices.buffer.values().len());
        // println!("{:#?}", vertices.buffer.values());
        commands
            .entity(viewer_ent)
            .insert(result.sampling_params)
            .remove::<VertexSamplingTask>();

        viewer.last_sampled_at = Some(std::time::Instant::now());

        vertices.instances = 0..inst_count as u32;
        vertices.params = Some(result.sampling_params);
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
    mut viewers: Query<
        (
            &SampledAlignmentViewer,
            // &VertexSamplingParams,
            &PolylineVertices,
            &mut PolylineModel,
        ),
        // Without<RenderOperation>,
    >,
    mut last_scale: Local<Vec3>,
) {
    for (viewer, vertices, mut model) in viewers.iter_mut() {
        let Some(next_view) = viewer.view else {
            continue;
        };

        let Some(sampled) = vertices.params else {
            continue;
        };

        let win_size = sampled.canvas_size;
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
        &PolylineVertices,
        &mut Transform,
        &mut Sprite,
    )>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let dpi_scale = window.scale_factor();

    for (viewer, _vertices, mut transform, mut sprite) in viewers.iter_mut() {
        let Some(rendered) = viewer.last_rendered else {
            continue;
        };

        let img_size = rendered.canvas_size.as_vec2();
        sprite.custom_size = Some(img_size / dpi_scale);

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

            let screen_delta = norm_delta.to_f32() * [img_size.x, img_size.y].as_uv() / dpi_scale;

            *transform =
                Transform::from_translation(Vec3::new(-screen_delta.x, -screen_delta.y, 0.0))
                    .with_scale(Vec3::new(w_rat as f32, h_rat as f32, 1.0));
        }
    }
}

#[derive(Debug, Clone, Component, ExtractComponent, Reflect)]
struct RenderOperation {
    view: crate::view::View,
    canvas_size: UVec2,
    vertex_params: VertexSamplingParams,
    #[reflect(ignore)]
    state: Arc<AtomicU8>,
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

    viewers: Query<(Entity, &SampledAlignmentViewer, &PolylineVertices), Without<RenderOperation>>,
    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let canvas_size = window.physical_size();

    for (viewer_ent, viewer, vertices) in viewers.iter() {
        let Some(vx_params) = vertices.params else {
            continue;
        };

        let Some(view) = viewer.view else {
            continue;
        };

        if let Some(last_time) = viewer.last_rendered_at {
            if last_time.elapsed().as_millis() < 20 {
                continue;
            }
        }

        let need_render = Some(view) != viewer.last_rendered.map(|p| p.view)
            || Some(vx_params) != viewer.last_rendered_sampling_params;

        if !need_render {
            // dbg!();
            continue;
        }

        // println!("triggering re-render");

        commands.entity(viewer_ent).insert(RenderOperation {
            view,
            canvas_size,
            vertex_params: vx_params,
            state: Arc::new(0.into()),
            // finished: Arc::new(false.into()),
        });
        // dbg!();
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
        let render_state = render_op.state.load(std::sync::atomic::Ordering::Relaxed);
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
        viewer.last_rendered_sampling_params = Some(render_op.vertex_params);
        viewer.last_rendered_at = Some(std::time::Instant::now());
        std::mem::swap(&mut front_tgts.0, &mut back_tgts.0);
        *sprite_img = front_tgts.0.color.clone_weak();

        println!("rendering complete");

        commands.entity(viewer_ent).remove::<RenderOperation>();
    }
}

#[derive(Debug)]
#[allow(unused)]
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
    let seq_pair_offset @ [o_x0, o_y0] = seq_pair_offset.into();
    let canvas_size @ [c_width, c_height] = canvas_size.into();
    let screen_dims = Vec2::new(c_width, c_height);
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

    let bp_per_px = view.width() / c_width as f64;

    let (y0, y1) = if loc.query_strand.is_rev() {
        (loc.query_range.end, loc.query_range.start)
    } else {
        (loc.query_range.start, loc.query_range.end)
    };

    use bevy::math::DVec2;

    let al_start = DVec2::new(o_x0 + loc.target_range.start as f64, o_y0 + y0 as f64);
    let al_end = DVec2::new(o_x0 + loc.target_range.end as f64, o_y0 + y1 as f64);

    let al_screen_start = view.map_world_to_screen(screen_dims, al_start);
    let al_screen_end = view.map_world_to_screen(screen_dims, al_end);

    let mut buffer_offset = 0;

    if alignment.cigar.is_empty() {
        buffer.push(VertexData {
            p0: *al_screen_start.as_array(),
            p1: *al_screen_end.as_array(),
            z: 0.5,
            color: 0xFF000000,
        });
        buffer_offset += 1;
    } else {
        let cg_iter = alignment.iter_target_range(vis_target_range);
        buffer_offset +=
            sample_alignment_iterator(cg_iter, seq_pair_offset, view, canvas_size, buffer);
        /*
        let mut cmd_iter = cigar_sampling::CigarScreenPathStrokeIter::new(
            *view,
            UVec2::new(c_width as u32, c_height as u32),
            [o_x0, o_y0],
            cg_iter,
        );

        // let mut path_start: Option<[u64; 2]> = None;
        let mut path_start = None;

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
        */
    }

    Ok(buffer_offset)
}

fn sample_alignment_iterator(
    iter: crate::paf::AlignmentIter,
    seq_pair_offset: [f64; 2],
    view: &crate::view::View,
    canvas_size: [f32; 2],
    buffer: &mut Vec<VertexData>,
) -> usize {
    let [x_o, y_o] = seq_pair_offset;

    let bp_per_px = view.width() / canvas_size[0] as f64;

    let mut open_match_world: Option<[f64; 2]> = None;
    let mut last_item: Option<crate::paf::AlignmentIterItem> = None;

    fn mk_segment<P: Into<[f32; 2]>>(p0: P, p1: P) -> VertexData {
        VertexData {
            p0: p0.into(),
            p1: p1.into(),
            z: 0.5,
            color: 0xFF000000,
        }
    }

    let buffer_start_len = buffer.len();

    for item in iter {
        let item_len = item.op_count as f64;

        let xs = item.target_seq_range();
        let ys = item.query_seq_range();

        let x0 = xs.start;
        let x1 = xs.end;

        let mut y0 = ys.start;
        let mut y1 = ys.end;

        if item.strand().is_rev() {
            std::mem::swap(&mut y0, &mut y1);
        }

        if item.op.is_match_or_mismatch() {
            if item_len > bp_per_px {
                if let Some(w_prev) = open_match_world {
                    // this item is big enough to be visible on its own and
                    // we've already opened a line segment

                    // do nothing?
                } else {
                    // this item is big enough to be visible and we haven't opened a line segment
                    open_match_world = Some([x0 as f64 + x_o, y0 as f64 + y_o]);
                }
            } else {
                if let Some(w_prev) = open_match_world {
                    // this item is small, and there's already an open
                    // line segment

                    // do nothing?
                } else {
                    // this item is small, but could be the start of
                    // a line segment continued by the next items
                    open_match_world = Some([x0 as f64 + x_o, y0 as f64 + y_o]);
                }
            }
        } else {
            if let Some(w0) = open_match_world {
                if item_len > bp_per_px {
                    // open match, and this indel would be visible,
                    // so emit a line segment

                    let p0 = view.map_world_to_screen(canvas_size, w0);

                    let w1 = [x0 as f64 + x_o, y0 as f64 + y_o];
                    let p1 = view.map_world_to_screen(canvas_size, w1);

                    let segment = mk_segment(p0, p1);
                    if buffer.len() - buffer_start_len < 10 {
                        let i = buffer.len() - buffer_start_len;
                        println!("{i} {segment:?}\nlast: {last_item:?}");
                    }
                    buffer.push(segment);
                    open_match_world = None;
                } else {
                    // open match, but this indel would be invisible
                    // do nothing?
                }
            } else {
                // no line segment has been opened, and this is an indel,
                // so there's nothing to do
            }
        }

        last_item = Some(item);
    }

    if let Some(last) = last_item {
        let last_x0 = last.target_seq_range().start as f64 + x_o;
        let last_x1 = last.target_seq_range().end as f64 + x_o;
        let (last_y0, last_y1) = {
            let y_min = last.query_seq_range().start as f64 + y_o;
            let y_max = last.query_seq_range().end as f64 + y_o;

            if last.strand().is_rev() {
                (y_max, y_min)
            } else {
                (y_min, y_max)
            }
        };

        if let Some(w0) = open_match_world {
            let p0 = view.map_world_to_screen(canvas_size, w0);
            let p1 = if last.op.is_match_or_mismatch() {
                // emit segment [w0, last.op.end]
                view.map_world_to_screen(canvas_size, [last_x1, last_y1])
            } else {
                // emit segment [w0, last.op.start]
                view.map_world_to_screen(canvas_size, [last_x0, last_y0])
            };
            buffer.push(mk_segment(p0, p1));
        } else {
            if last.op.is_match_or_mismatch() {
                // emit segment [last.op.start, last.op.end]
                let w0 = [last_x0, last_y0];
                let p0 = view.map_world_to_screen(canvas_size, w0);
                let w1 = [last_x1, last_y1];
                let p1 = view.map_world_to_screen(canvas_size, w1);
                buffer.push(mk_segment(p0, p1));
            }
        }
    }

    let vx_count = buffer.len() - buffer_start_len;
    vx_count
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
        pub(super) params: Option<VertexSamplingParams>,
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
            let mut buffer = RawBufferVec::new(BufferUsages::VERTEX | BufferUsages::COPY_DST);
            buffer.set_label(Some("PolylineVertices Buffer"));
            Self {
                buffer,
                instances: 0..0,
                params: None,
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

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent, Reflect)]
    pub(super) struct PolylineModel {
        pub(super) model: Mat4,
    }

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent, Reflect)]
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

    #[derive(ShaderType, Clone, Copy, Component, ExtractComponent, Reflect)]
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
            Option<&ExtractedVertexBuffer>,
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

        for (_entity, vertices, uniform_indices, render_op, render_tgt) in polylines.iter() {
            let Some(vertices) = vertices.as_ref() else {
                render_op.state.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            let render_state = render_op.state.load(std::sync::atomic::Ordering::Relaxed);

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
                render_op.state.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            // println!("rendering to image size {:?}", tgt_img.size);

            if vertices.instances.len() == 0 {
                render_op.state.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            }

            let vx_buffer = &vertices.buffer;
            // create bind groups

            let Some((proj_binding, cfg_binding)) = projections
                .uniforms()
                .binding()
                .zip(configs.uniforms().binding())
            else {
                render_op.state.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            let Some(model_binding) = models.uniforms().binding() else {
                render_op.state.store(
                    RenderOperation::STATE_ERROR,
                    std::sync::atomic::Ordering::Relaxed,
                );
                continue;
            };

            render_op.state.store(
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

            // start render

            let finished = render_op.state.clone();
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

/*
mod debug {
    use super::*;

    pub(super) struct DebugPlugin;

    impl Plugin for DebugPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup_debug_display)
                .add_systems(Update, update_debug_display.after(super::update_projection));
        }
    }

    #[derive(Resource)]
    struct DebugRootNode {
        root: Entity,
        model_text: Entity,
    }

    fn setup_debug_display(mut commands: Commands) {
        let mut model = Entity::PLACEHOLDER;

        let root = commands
            .spawn(NodeBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    bottom: Val::Px(100.0),
                    ..default()
                },
                background_color: Color::srgb(0.65, 0.65, 0.65).into(),
                ..default()
            })
            .with_children(|parent| {
                // left vertical fill (border)

                parent
                    .spawn(NodeBundle {
                        style: Style {
                            width: Val::Px(300.0),
                            height: Val::Px(200.0),
                            // border: UiRect::all(Val::Px(2.)),
                            ..default()
                        },
                        // background_color: Color::srgb(0.65, 0.65, 0.65).into(),
                        ..default()
                    })
                    .with_children(|parent| {
                        model = parent
                            .spawn(TextBundle {
                                text: Text::from_section(
                                    "",
                                    TextStyle {
                                        // font_size: 10.0,
                                        ..default()
                                    },
                                ),
                                ..default()
                            })
                            .id();
                    });
            })
            .id();
        commands.insert_resource(DebugRootNode {
            root,
            model_text: model,
        });
    }

    fn update_debug_display(
        debug_root: Res<DebugRootNode>,
        viewers: Query<(
            &Handle<Image>,
            &FrontRenderTarget,
            &BackRenderTarget,
            &PolylineModel,
        )>,
        mut ui_bg: Query<&mut BackgroundColor, With<Node>>,
        mut ui_text: Query<&mut Text>,
    ) {
        let Ok(mut bg) = ui_bg.get_mut(debug_root.root) else {
            return;
        };

        for (img, front, back, model) in viewers.iter() {
            let f = &front.0.color;
            let b = &back.0.color;
            let min = f.min(b);
            let color = if img == min {
                Color::linear_rgba(1.0, 0.0, 0.0, 1.0)
            } else {
                Color::linear_rgba(0.0, 0.0, 1.0, 1.0)
            };

            bg.0 = color;

            if let Ok(mut ui_text) = ui_text.get_mut(debug_root.model_text) {
                let mat = &model.model;
                let x0 = mat.x_axis.x;
                let y1 = mat.y_axis.y;
                let wxy = mat.w_axis.xy();
                let wx = wxy.x;
                let wy = wxy.y;

                ui_text.sections[0].value = format!("x: {x0}\ny: {y1}\nw_xy: [{wx}, {wy}]");
            }
        }
    }
}
*/
