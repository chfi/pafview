use egui::Context;
use egui_wgpu::Renderer;
use egui_wgpu::ScreenDescriptor;
use egui_winit::EventResponse;
use egui_winit::State;

use wgpu::SurfaceConfiguration;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::{event::WindowEvent, window::Window};

use super::create_multisampled_framebuffer;

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

        egui_context.set_visuals(egui::Visuals::light());

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
                [surface_config.width, surface_config.height],
                surface_config.format,
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
                [config.width, config.height],
                config.format,
                msaa_samples,
            ))
        } else {
            None
        };
    }

    pub fn handle_input(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn initialize(
        &mut self,
        //
        window: &Window,
    ) {
        // self.context.run(raw_
        let raw_input = self.state.take_egui_input(window);
        self.context.begin_frame(raw_input);
        let full_output = self.context.end_frame();
        // let full_output = self.context.run(raw_input, |ui| {
        // run_ui(&self.context);
        // });
        self.state
            .handle_platform_output(&window, full_output.platform_output);
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
        let full_output = self.context.run(raw_input, |_ui| {
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
