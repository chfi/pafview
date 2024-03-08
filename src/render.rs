use bytemuck::{Pod, Zeroable};
use egui::{Context, ViewportBuilder};
use egui_wgpu::Renderer;
use egui_wgpu::ScreenDescriptor;
use egui_winit::EventResponse;
use egui_winit::State;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::SurfaceConfiguration;
use wgpu::Texture;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::{
    event::{Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use std::io::prelude::*;

use anyhow::anyhow;

pub struct LinePipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl LinePipeline {
    pub fn new(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        //
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("lines.wgsl"))),
        });

        let proj = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let conf = wgpu::BindGroupLayoutEntry { binding: 1, ..proj };
        let layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[proj, conf],
        };

        let bind_group_layout = device.create_bind_group_layout(&layout_desc);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 4 * std::mem::size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        // wgpu::VertexAttribute {
                        //     format: wgpu::VertexFormat::Uint32,
                        //     offset: 16,
                        //     shader_location: 2,
                        // },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            // multisample: wgpu::MultisampleState::default(),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
        });

        Self {
            bind_group_layout,
            pipeline_layout,
            pipeline,
        }
    }
}

pub struct ShortMatchPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub pipeline: wgpu::RenderPipeline,
}

impl ShortMatchPipeline {
    pub fn new(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        //
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("short.wgsl"))),
        });

        let proj = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let conf = wgpu::BindGroupLayoutEntry { binding: 1, ..proj };
        let layout_desc = wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[proj, conf],
        };

        let bind_group_layout = device.create_bind_group_layout(&layout_desc);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 4 * std::mem::size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        // wgpu::VertexAttribute {
                        //     format: wgpu::VertexFormat::Uint32,
                        //     offset: 16,
                        //     shader_location: 2,
                        // },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            // multisample: wgpu::MultisampleState::default(),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                // mask: todo!(),
                alpha_to_coverage_enabled: true,
                ..Default::default()
            },
            multiview: None,
        });

        Self {
            bind_group_layout,
            pipeline_layout,
            pipeline,
        }
    }
}

pub struct EguiRenderer {
    pub context: Context,
    state: State,
    renderer: Renderer,

    msaa_framebuffer: Option<TextureView>,
}

impl EguiRenderer {
    pub fn new(
        device: &Device,
        surface_config: &SurfaceConfiguration,
        output_color_format: TextureFormat,
        output_depth_format: Option<TextureFormat>,
        msaa_samples: u32,
        window: &Window,
    ) -> EguiRenderer {
        let egui_context = Context::default();

        let viewport = ViewportBuilder::default();

        let viewport_id = egui_context.viewport_id();

        let egui_state = egui_winit::State::new(
            egui_context.clone(),
            viewport_id,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            output_color_format,
            output_depth_format,
            msaa_samples,
        );

        let msaa_framebuffer = if msaa_samples > 1 {
            Some(create_multisampled_framebuffer(
                &device,
                &surface_config,
                msaa_samples,
            ))
        } else {
            None
        };

        EguiRenderer {
            context: egui_context,
            state: egui_state,
            renderer: egui_renderer,
            msaa_framebuffer,
        }
    }

    pub fn resize(&mut self, device: &Device, config: &SurfaceConfiguration, msaa_samples: u32) {
        self.msaa_framebuffer = if msaa_samples > 1 {
            Some(create_multisampled_framebuffer(
                &device,
                &config,
                msaa_samples,
            ))
        } else {
            None
        };
    }

    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn draw(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        window: &Window,
        window_surface_view: &TextureView,
        screen_descriptor: ScreenDescriptor,
        run_ui: impl FnOnce(&Context),
    ) {
        // self.state.set_pixels_per_point(window.scale_factor() as f32);
        let raw_input = self.state.take_egui_input(&window);
        let full_output = self.context.run(raw_input, |ui| {
            run_ui(&self.context);
        });

        self.state
            .handle_platform_output(&window, full_output.platform_output);

        let tris = self
            .context
            .tessellate(full_output.shapes, self.context.pixels_per_point());
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(&device, &queue, *id, &image_delta);
        }
        self.renderer
            .update_buffers(&device, &queue, encoder, &tris, &screen_descriptor);

        let attch = if let Some(msaa_framebuffer) = &self.msaa_framebuffer {
            wgpu::RenderPassColorAttachment {
                view: msaa_framebuffer,
                resolve_target: Some(window_surface_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: window_surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(attch)],
            depth_stencil_attachment: None,
            label: Some("egui main render pass"),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.renderer.render(&mut rpass, &tris, &screen_descriptor);
        drop(rpass);
        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }
    }
}

pub fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    sample_count: u32,
) -> wgpu::TextureView {
    let multisampled_texture_extent = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: multisampled_texture_extent,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}

pub struct PafRenderer {
    line_pipeline: LinePipeline,
    msaa_samples: u32,

    match_vertices: wgpu::Buffer,

    active_task: Option<PafDrawTask>,
    draw_sets: [Option<PafDrawSet>; 2],
}

impl PafRenderer {
    pub fn new(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        msaa_samples: u32,
        match_vertices: wgpu::Buffer,
    ) -> Self {
        let line_pipeline = LinePipeline::new(&device, swapchain_format, msaa_samples);

        Self {
            line_pipeline,
            msaa_samples,
            match_vertices,
            active_task: None,
            draw_sets: [None, None],
        }
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: crate::view::View,
        window_dims: [u32; 2],
        swapchain_view: &TextureView,
    ) {
        // if there's an active task and it has completed, swap it to the front
        let task_complete = self.active_task.as_ref().is_some_and(|t| t.is_complete());
        if task_complete {
            self.active_task = None;
            self.draw_sets.swap(0, 1);
            debug_assert!(self.draw_sets[0].is_some());
        }

        let task_running = self.active_task.is_some();

        let need_update = self.draw_sets[0]
            .as_ref()
            .is_some_and(|set| !set.params.matches_view_and_dims(view, window_dims));

        if !task_running && need_update {
            // if the new view and/or window dims differ from the front buffer,
            // and if there is no active task, queue a new task using the back buffer

            let create_draw_set = if let Some(set) = &self.draw_sets[1] {
                set.params.window_dims != window_dims
            } else {
                true
            };

            if create_draw_set {
                let params = PafDrawParams::from_view_and_dims(view, window_dims);

                self.draw_sets[1] = Some(PafDrawSet {
                    params,
                    framebuffers: todo!(),
                    line_bind_group: todo!(),
                });
            }

            let Some(draw_set) = &self.draw_sets[1] else {
                unreachable!();
            };

            let draw_set = if self.draw_sets[1].is_none() {
                // create new framebuffers & bind group
                todo!();
            } else {
                // use old framebuffers & bind group if size matches, otherwise recreate
                todo!();
            };

            // render pass & encoder
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PafRenderer Lines"),
            });

            //

            queue.submit([encoder.finish()]);
            queue.on_submitted_work_done(|| {
                //
            });
        }

        /*
        if let Some(task) = &self.active_task {
            // if window_dims == task.window_dims &&

        } else {
        };
        */

        todo!();
    }

    fn draw_frame(
        &self,
        params: &PafDrawSet,
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

        rpass.set_pipeline(&self.line_pipeline.pipeline);
        rpass.set_bind_group(0, &params.line_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.match_vertices.slice(..));
        rpass.draw(0..6, match_instances);
    }
}

struct PafTextures {
    color_texture: Texture,
    color_view: TextureView,
    msaa_view: Option<TextureView>,
}

struct PafDrawTask {
    complete: Arc<AtomicBool>,
}

impl PafDrawTask {
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
    fn from_view_and_dims(view: crate::view::View, window_dims: [u32; 2]) -> Self {
        Self {
            target_range: view.x_range(),
            query_range: view.y_range(),
            window_dims,
        }
    }

    fn matches_view_and_dims(&self, view: crate::view::View, window_dims: [u32; 2]) -> bool {
        self.window_dims == window_dims
            && self.target_range == view.x_range()
            && self.query_range == view.y_range()
    }
}

pub struct PafDrawSet {
    params: PafDrawParams,
    framebuffers: PafTextures,
    line_bind_group: wgpu::BindGroup,
}

pub struct ImageRendererBindGroup {
    color_view_id: Option<wgpu::Id<wgpu::TextureView>>,
    bind_group: Option<wgpu::BindGroup>,
}

pub struct ImageRenderer {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,

    sampler: wgpu::Sampler,
}

impl ImageRenderer {
    pub fn new(
        device: &wgpu::Device,
        // color_format: wgpu::TextureFormat,
        swapchain_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sample_image.wgsl"))),
        });

        let bind_group_entries = [
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Image Renderer"),
            entries: &bind_group_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
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
            bind_group_layout,
            pipeline_layout,
            pipeline,
            sampler,
        }
    }

    fn create_bind_groups(
        &self,
        device: &wgpu::Device,
        img_group: &mut ImageRendererBindGroup,
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
            layout: &self.bind_group_layout,
            entries: &entries,
        });

        img_group.color_view_id = Some(color_view.global_id());
        img_group.bind_group = Some(bind_group);
    }

    fn draw(
        &self,
        bind_group: &ImageRendererBindGroup,
        swapchain_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some(bind_group) = &bind_group.bind_group else {
            log::warn!(
                "Attempted to draw using the sampled image renderer but there's no bind group"
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
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }
}
