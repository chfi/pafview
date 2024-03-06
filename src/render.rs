use bytemuck::{Pod, Zeroable};
use egui::{Context, ViewportBuilder};
use egui_wgpu::Renderer;
use egui_wgpu::ScreenDescriptor;
use egui_winit::EventResponse;
use egui_winit::State;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::SurfaceConfiguration;
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
                resolve_target: Some(&window_surface_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: &window_surface_view,
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

    match_vertices: Option<wgpu::Buffer>,
}

impl PafRenderer {
    fn draw_frame(
        &self,
        params: &PafDrawSet,
        match_instances: std::ops::Range<u32>,
        encoder: &mut CommandEncoder,
    ) {
        let Some(match_buffer) = &self.match_vertices else {
            log::warn!("PafRenderer has no vertex buffer");
            return;
        };

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
        rpass.set_vertex_buffer(0, match_buffer.slice(..));
        rpass.draw(0..6, match_instances);
    }
}

struct PafTextures {
    color_view: TextureView,
    msaa_view: Option<TextureView>,
}

pub struct PafDrawSet {
    framebuffers: PafTextures,

    line_bind_group: wgpu::BindGroup,
}
