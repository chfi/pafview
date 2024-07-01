use std::borrow::Cow;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use ultraviolet::Mat4;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::SurfaceConfiguration;
use wgpu::Texture;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

use crate::render::batch::ColorSchemeBuffers;

use self::exact::CpuRasterizerBindGroups;
use self::exact::CpuViewRasterizerEgui;
use self::lines::LineColorSchemePipeline;

pub mod batch;
pub mod color;
pub mod exact;
pub mod gui;
pub mod lines;
pub mod thread;

pub use gui::EguiRenderer;
pub use lines::LinePipeline;

pub struct PafRenderer {
    pub line_pipeline: LinePipeline,
    pub line_color_scheme_pipeline: LineColorSchemePipeline,
    pub color_scheme_buffers: ColorSchemeBuffers,
    msaa_samples: u32,

    pub line_width: f32,

    grid_data: Option<(wgpu::Buffer, wgpu::Buffer, std::ops::Range<u32>)>,

    active_task: Option<PafDrawTask>,
    draw_states: [PafDrawState; 2],

    image_renderer: ImageRenderer,
    image_bind_groups: [ImageRendererBindGroups; 2],
    exact_image_bind_groups: exact::CpuRasterizerBindGroups,

    // temporary (may be used for grid later still)
    identity_uniform: wgpu::Buffer,
    identity_bind_group: wgpu::BindGroup,

    last_rendered_view: Option<crate::view::View>,
}

impl PafRenderer {
    const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    // TODO should probably be configurable, but for now this is the easy way
    // to make it consistent
    const SCALE_LIMIT_BP_PER_PX: f64 = 1.0;

    pub fn new(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        msaa_samples: u32,
        alignment_colors: &color::PafColorSchemes,
        // match_vertices: wgpu::Buffer,
        // match_colors: wgpu::Buffer,
        // match_instances: std::ops::Range<u32>,
    ) -> Self {
        log::warn!("initializing PafRenderer");
        let line_pipeline = LinePipeline::new(&device, Self::COLOR_FORMAT, msaa_samples);

        let line_color_scheme_pipeline =
            LineColorSchemePipeline::new(&device, Self::COLOR_FORMAT, msaa_samples);

        let init_state = || PafDrawState::init(device, &line_pipeline.bind_group_layout_0);

        let draw_states = [init_state(), init_state()];

        let image_renderer = ImageRenderer::new(device, swapchain_format);

        let image_bind_groups = [
            ImageRendererBindGroups::new(device, &image_renderer.bind_group_layout_0),
            ImageRendererBindGroups::new(device, &image_renderer.bind_group_layout_0),
        ];

        // let identity_uniform = device.create_buffer_init(&BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[mat], usage: () });

        let identity_mat = Mat4::identity();

        let identity_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[identity_mat]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let identity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &line_pipeline.bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: identity_uniform.as_entire_binding(),
            }],
        });

        let exact_image_bind_groups = CpuRasterizerBindGroups::new(device, &image_renderer);

        let color_scheme_buffers = ColorSchemeBuffers::from_color_schemes(
            device,
            &line_color_scheme_pipeline,
            alignment_colors,
        );

        Self {
            line_pipeline,
            line_color_scheme_pipeline,
            color_scheme_buffers,
            msaa_samples,

            line_width: 5.0,

            // match_vertices,
            // match_colors,
            // match_instances,
            grid_data: None,

            active_task: None,
            draw_states,

            image_renderer,
            image_bind_groups,

            exact_image_bind_groups,

            identity_uniform,
            identity_bind_group,

            last_rendered_view: None,
        }
    }

    pub fn set_grid(
        &mut self,
        grid_data: Option<(wgpu::Buffer, wgpu::Buffer, std::ops::Range<u32>)>,
    ) {
        self.grid_data = grid_data;
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        app: &crate::PafViewerApp,
        cpu_rasterizer: &mut CpuViewRasterizerEgui,
        alignment_colors: &color::PafColorSchemes,
        match_data: &batch::MatchDrawBatchData,
        view: &crate::view::View,
        window_dims: [u32; 2],
        swapchain_view: &TextureView,
        encoder: &mut CommandEncoder,
    ) {
        let front_image_dims = self.draw_states[0]
            .draw_set
            .as_ref()
            .map(|set| {
                let size = set.framebuffers.color_texture.size();
                [size.width, size.height]
            })
            .unwrap_or([0, 0]);

        if self.last_rendered_view != Some(*view) || front_image_dims != window_dims {
            if view.bp_per_pixel(window_dims[0]) > Self::SCALE_LIMIT_BP_PER_PX {
                self.submit_draw_matches(
                    device,
                    queue,
                    alignment_colors,
                    &app.app_config,
                    match_data,
                    view,
                    window_dims,
                );
            } else {
                cpu_rasterizer.draw_into_wgpu_texture(device, queue, window_dims, app, view);
                self.last_rendered_view = Some(*view);
            }
        }

        if view.bp_per_pixel(window_dims[0]) > Self::SCALE_LIMIT_BP_PER_PX {
            self.draw_front_image(device, queue, view, window_dims, swapchain_view, encoder);
        } else {
            let Some((_texture, texture_view)) = &cpu_rasterizer.last_wgpu_texture else {
                return;
            };

            self.exact_image_bind_groups.create_bind_groups(
                device,
                &self.image_renderer,
                texture_view,
            );
            let bind_group = &mut self.exact_image_bind_groups.image_bind_groups[0];
            bind_group.update_uniform(queue);
            self.image_renderer
                .draw(bind_group, swapchain_view, encoder);
        }
    }

    fn draw_front_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // alignment_colors: &color::PafColorSchemes,
        view: &crate::view::View,
        window_dims: [u32; 2],
        swapchain_view: &TextureView,
        encoder: &mut CommandEncoder,
    ) {
        if view.bp_per_pixel(window_dims[0]) <= Self::SCALE_LIMIT_BP_PER_PX
            || self.draw_states[0].draw_set.is_none()
        {
            self.image_renderer.clear(swapchain_view, encoder);
            return;
        }

        if let Some(set) = self.draw_states[0].draw_set.as_ref() {
            self.image_renderer.create_bind_groups(
                device,
                &mut self.image_bind_groups[0],
                &set.framebuffers.color_view,
            );

            self.image_bind_groups[0].update_uniform(queue);

            self.image_renderer
                .draw(&self.image_bind_groups[0], swapchain_view, encoder);
        }
    }

    fn submit_draw_matches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        alignment_colors: &color::PafColorSchemes,
        config: &crate::config::AppConfig,
        match_data: &batch::MatchDrawBatchData,
        view: &crate::view::View,
        window_dims: [u32; 2],
    ) {
        // if there's an active task and it has completed, swap it to the front
        let task_complete = self.active_task.as_ref().is_some_and(|t| t.is_complete());
        if task_complete {
            self.active_task = None;
            self.draw_states.swap(0, 1);
            self.image_bind_groups.swap(0, 1);
            self.last_rendered_view = Some(*view);
            debug_assert!(self.draw_states[0].draw_set.is_some());
        }

        let task_running = self.active_task.is_some();

        let is_up_to_date = self.draw_states[0]
            .draw_set
            .as_ref()
            .is_some_and(|set| set.params.matches_view_and_dims(view, window_dims));

        if !task_running && !is_up_to_date {
            // if the new view and/or window dims differ from the front buffer,
            // and if there is no active task, queue a new task using the back buffer

            // recreate buffers if present & size mismatch
            if let Some(mut set) = self.draw_states[1].draw_set.take() {
                if set.params.window_dims != window_dims {
                    log::warn!("recreating framebuffers");
                    set.recreate_framebuffers(
                        device,
                        &self.line_pipeline.bind_group_layout_0,
                        Self::COLOR_FORMAT,
                        self.msaa_samples,
                        window_dims,
                    )
                }
                self.draw_states[1].draw_set = Some(set);
            }

            // create buffers if not initialized
            if self.draw_states[1].draw_set.is_none() {
                log::warn!("initializing PafDrawSet");
                self.draw_states[1].draw_set = Some(PafDrawSet::new(
                    device,
                    &self.line_pipeline.bind_group_layout_0,
                    Self::COLOR_FORMAT,
                    self.msaa_samples,
                    window_dims,
                ));
            }

            let params = PafDrawParams::from_view_and_dims(view, window_dims);
            self.draw_states[1].update_uniforms(
                queue,
                view,
                window_dims,
                config.alignment_line_width,
                config.grid_line_width,
            );

            let Some(draw_set) = self.draw_states[1].draw_set.as_mut() else {
                unreachable!();
            };

            draw_set.params = params;

            let uniforms = &self.draw_states[1].uniforms;
            // render pass & encoder
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PafRenderer Lines"),
            });

            // Self::draw_frame_tiled(
            //     match_data,
            //     &self.line_pipeline,
            //     draw_set,
            //     uniforms,
            //     &self.identity_bind_group,
            //     &self.grid_data,
            //     &mut encoder,
            // );

            Self::draw_frame_tiled_color_schemes(
                &self.line_pipeline,
                &self.line_color_scheme_pipeline,
                &self.color_scheme_buffers,
                alignment_colors,
                match_data,
                draw_set,
                uniforms,
                &self.identity_bind_group,
                &self.grid_data,
                &mut encoder,
            );

            // Self::draw_frame(
            //     &self.line_pipeline,
            //     draw_set,
            //     uniforms,
            //     &self.identity_bind_group,
            //     &self.grid_data,
            //     &self.match_vertices,
            //     &self.match_colors,
            //     instances,
            //     &mut encoder,
            // );

            let task = PafDrawTask::new();
            self.active_task = Some(task.clone());

            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done(move || {
                task.complete
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            });
        }
    }

    fn draw_frame(
        line_pipeline: &LinePipeline,
        params: &PafDrawSet,
        uniforms: &PafUniforms,
        identity_uniform: &wgpu::BindGroup,
        grid_data: &Option<(wgpu::Buffer, wgpu::Buffer, std::ops::Range<u32>)>,
        match_vertices: &wgpu::Buffer,
        match_colors: &wgpu::Buffer,
        match_instances: std::ops::Range<u32>,
        encoder: &mut CommandEncoder,
    ) {
        let attch = if let Some(msaa_view) = &params.framebuffers.msaa_view {
            wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(&params.framebuffers.color_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: &params.framebuffers.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(attch)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&line_pipeline.pipeline);

        if let Some((grid_vertices, grid_colors, grid_instances)) = grid_data {
            rpass.set_bind_group(0, &uniforms.grid_bind_group, &[]);
            rpass.set_bind_group(1, identity_uniform, &[]);
            rpass.set_vertex_buffer(0, grid_vertices.slice(..));
            rpass.set_vertex_buffer(1, grid_colors.slice(..));
            rpass.draw(0..6, grid_instances.clone());
        }

        rpass.set_bind_group(0, &uniforms.line_bind_group, &[]);
        rpass.set_bind_group(1, identity_uniform, &[]);
        rpass.set_vertex_buffer(0, match_vertices.slice(..));
        rpass.set_vertex_buffer(1, match_colors.slice(..));
        rpass.draw(0..6, match_instances);
    }
}

#[derive(Debug)]
struct PafTextures {
    color_texture: Texture,
    color_view: TextureView,
    msaa_view: Option<TextureView>,
}

impl PafTextures {
    fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        size: [u32; 2],
        msaa_samples: u32,
    ) -> Self {
        let label_prefix = format!("PafTextures {size:?} MSAA:{msaa_samples}");

        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{label_prefix} Color")),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{label_prefix} Color View")),
            ..Default::default()
        });

        let msaa_view = if msaa_samples > 1 {
            Some(create_multisampled_framebuffer(
                device,
                size,
                color_format,
                msaa_samples,
            ))
        } else {
            None
        };

        Self {
            color_texture,
            color_view,
            msaa_view,
        }
    }
}

#[derive(Clone)]
struct PafDrawTask {
    complete: Arc<AtomicBool>,
}

impl PafDrawTask {
    fn new() -> Self {
        Self {
            complete: AtomicBool::new(false).into(),
        }
    }

    fn is_complete(&self) -> bool {
        self.complete.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PafDrawParams {
    target_range: std::ops::RangeInclusive<f64>,
    query_range: std::ops::RangeInclusive<f64>,
    window_dims: [u32; 2],
}

impl PafDrawParams {
    fn from_view_and_dims(view: &crate::view::View, window_dims: [u32; 2]) -> Self {
        Self {
            target_range: view.x_range(),
            query_range: view.y_range(),
            window_dims,
        }
    }

    fn matches_view_and_dims(&self, view: &crate::view::View, window_dims: [u32; 2]) -> bool {
        self.window_dims == window_dims
            && self.target_range == view.x_range()
            && self.query_range == view.y_range()
    }
}

struct PafUniforms {
    proj_uniform: wgpu::Buffer,
    conf_uniform: wgpu::Buffer,
    line_bind_group: wgpu::BindGroup,

    grid_conf_uniform: wgpu::Buffer,
    grid_bind_group: wgpu::BindGroup,
}

struct PafDrawState {
    draw_set: Option<PafDrawSet>,
    uniforms: PafUniforms,
}

impl PafDrawState {
    fn init(device: &wgpu::Device, bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        // let projection = view.to_mat4();
        let mat = Mat4::identity();
        let proj_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[mat]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // NB: the values here don't matter since they'll get immediately updated
        //     by self.update_uniforms()
        let conf = [5f32, 0f32, 0.0, 0.0];
        let conf_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&conf),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: conf_uniform.as_entire_binding(),
                },
            ],
        });

        let conf = [1f32, 1.0f32, 0.0, 0.0];
        let grid_conf_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&conf),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: proj_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_conf_uniform.as_entire_binding(),
                },
            ],
        });

        Self {
            draw_set: None,
            uniforms: PafUniforms {
                proj_uniform,
                conf_uniform,
                line_bind_group,

                grid_conf_uniform,
                grid_bind_group,
            },
        }
    }

    fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view: &crate::view::View,
        window_dims: [u32; 2],
        alignment_line_width: f32,
        grid_line_width: f32,
    ) {
        let proj = view.to_mat4();
        queue.write_buffer(
            &self.uniforms.proj_uniform,
            0,
            bytemuck::cast_slice(&[proj]),
        );

        let line_brightness = 0.0;
        // px / window width
        // let line_width: f32 = 5.0 / 1000.0;
        // let line_width: f32 = 15.0 / 1000.0;
        let match_line_width: f32 = alignment_line_width / window_dims[0] as f32;
        queue.write_buffer(
            &self.uniforms.conf_uniform,
            0,
            bytemuck::cast_slice(&[match_line_width, line_brightness, 0.0, 0.0]),
        );

        let grid_line_width: f32 = grid_line_width / window_dims[0] as f32;
        let line_brightness = 0.0;
        queue.write_buffer(
            &self.uniforms.grid_conf_uniform,
            0,
            bytemuck::cast_slice(&[grid_line_width, line_brightness, 0.0, 0.0]),
        );
    }
}

#[derive(Debug)]
struct PafDrawSet {
    params: PafDrawParams,
    framebuffers: PafTextures,
    // uniforms: PafUniforms,
}

impl PafDrawSet {
    fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        msaa_samples: u32,
        // view: crate::view::View,
        // params: PafDrawParams,
        window_dims: [u32; 2],
    ) -> Self {
        let framebuffers = PafTextures::new(device, color_format, window_dims, msaa_samples);
        let params = PafDrawParams {
            target_range: 0.0..=0.0,
            query_range: 0.0..=0.0,
            window_dims,
        };
        // let params = PafDrawParams::from_view_and_dims(view, window_dims);

        Self {
            params,
            framebuffers,
        }
    }

    fn recreate_framebuffers(
        &mut self,
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
        msaa_samples: u32,
        window_dims: [u32; 2],
    ) {
        if self.params.window_dims == window_dims {
            return;
        }

        self.params.window_dims = window_dims;
        self.params.query_range = 0.0..=0.0;
        self.params.target_range = 0.0..=0.0;

        self.framebuffers = PafTextures::new(device, color_format, window_dims, msaa_samples);
    }
}

struct ImageRendererBindGroups {
    uniform: wgpu::Buffer,
    bind_group_0: wgpu::BindGroup,

    color_view_id: Option<wgpu::Id<wgpu::TextureView>>,
    bind_group_1: Option<wgpu::BindGroup>,
}

impl ImageRendererBindGroups {
    fn new(device: &wgpu::Device, bind_group_0_layout: &wgpu::BindGroupLayout) -> Self {
        let mat = Mat4::identity();
        let uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[mat]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Image Renderer - Bind Group 0"),
            layout: bind_group_0_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        Self {
            uniform,
            bind_group_0,
            color_view_id: None,
            bind_group_1: None,
        }
    }

    fn update_uniform(
        &self,
        queue: &wgpu::Queue,
        // old_params: &PafDrawParams,
        // new_params: &PafDrawParams,
    ) {
        // TODO update uniform based on old & new view
        let matrix = Mat4::identity();
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&[matrix]));
    }
}

struct ImageRenderer {
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,

    sampler: wgpu::Sampler,
}

impl ImageRenderer {
    fn new(
        device: &wgpu::Device,
        // color_format: wgpu::TextureFormat,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sample_image.wgsl"))),
        });

        let bind_group_entries_0 = [wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];

        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Image Renderer - Vertex uniform"),
                entries: &bind_group_entries_0,
            });

        let bind_group_entries_1 = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Image Renderer - Frag uniform"),
                entries: &bind_group_entries_1,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Image Renderer"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Image Renderer"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            bind_group_layout_0,
            bind_group_layout_1,
            pipeline_layout,
            pipeline,
            sampler,
        }
    }

    fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        img_group: &mut ImageRendererBindGroups,
        color_view: &wgpu::TextureView,
    ) {
        let is_up_to_date = img_group
            .color_view_id
            .is_some_and(|id| id == color_view.global_id());

        if is_up_to_date {
            return;
        }

        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
        ];

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Image Renderer"),
            layout: &self.bind_group_layout_1,
            entries: &entries,
        });

        img_group.color_view_id = Some(color_view.global_id());
        img_group.bind_group_1 = Some(bind_group);
    }

    fn clear(&self, swapchain_view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        let attch = wgpu::RenderPassColorAttachment {
            view: swapchain_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        };

        // creating and dropping render pass to clear swapchain image
        let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Image Renderer"),
            color_attachments: &[Some(attch)],
            ..Default::default()
        });
    }

    fn draw(
        &self,
        bind_group: &ImageRendererBindGroups,
        swapchain_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some(bind_group_1) = &bind_group.bind_group_1 else {
            log::warn!(
                "Attempted to draw using the sampled image renderer but there's no texture bind group"
            );
            return;
        };

        let attch = wgpu::RenderPassColorAttachment {
            view: swapchain_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Image Renderer"),
            color_attachments: &[Some(attch)],
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group.bind_group_0, &[]);
        rpass.set_bind_group(1, bind_group_1, &[]);
        rpass.draw(0..6, 0..1);
    }
}

pub fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    dims: [u32; 2],
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::TextureView {
    let [width, height] = dims;
    let multisampled_texture_extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: multisampled_texture_extent,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}
