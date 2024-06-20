use annotations::{draw::AnnotationPainter, physics::LabelPhysics, AnnotationGuiHandler};
use bimap::BiMap;
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use egui_wgpu::ScreenDescriptor;
use grid::AlignmentGrid;
use paf::Alignments;
use regions::SelectionHandler;
use rustc_hash::FxHashMap;
use sequences::Sequences;
use std::{
    borrow::Cow,
    str::FromStr,
    sync::{atomic::AtomicBool, Arc},
};
use ultraviolet::{DVec2, Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{ElementState, Event, MouseButton, WindowEvent},
    event_loop::{EventLoop, EventLoopBuilder},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use std::io::prelude::*;

use anyhow::anyhow;

use pafview::paf;
use pafview::pixels;
use pafview::{cigar, AppEvent};
use pafview::{math_conv, PafViewerApp};

use pafview::config;
use pafview::config::AppConfig;

use pafview::annotations;
use pafview::cli;
use pafview::grid;
use pafview::gui;
use pafview::regions;
use pafview::render;
use pafview::sequences;
use pafview::view;

use cigar::*;
use paf::PafLine;
use pixels::*;
use render::*;
use view::View;

use pafview::{
    annotations::AnnotationStore, gui::AppWindowStates, paf::parse_paf_line, sequences::SeqId,
};

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AlignedSeq {
    // name of the given sequence
    name: String,
    // its length
    len: u64,
    // its start offset in the global all-to-all alignment matrix
    #[deprecated]
    offset: u64,
}

pub fn main() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
    }

    let args = pafview::cli::Cli::parse();

    // Load PAF and optional FASTA
    let (alignments, sequences) = pafview::paf::load_input_files(&args)?;

    println!("drawing {} alignments", alignments.pairs.len());

    let alignment_grid = AlignmentGrid::from_alignments(&alignments, sequences.names().clone());
    // let alignment_grid = AlignmentGrid::from_axes(&alignments, sequences.names().clone(), x_axis, y_axis);
    // let alignment_grid = AlignmentGrid {
    //     x_axis,
    //     y_axis,
    //     sequence_names: sequences.names().clone(),
    // };

    let app_config = config::load_app_config().unwrap_or_default();

    let app = PafViewerApp {
        app_config,
        alignments: Arc::new(alignments),
        alignment_grid: Arc::new(alignment_grid),
        sequences,
        // paf_input: todo!(),
        annotations: AnnotationStore::default(),
    };

    start_window(app);

    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: [f32; 2],
    p1: [f32; 2],
    // color: u32,
}

async fn run(event_loop: EventLoop<AppEvent>, window: Window, mut app: PafViewerApp) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    let instance = wgpu::Instance::default();

    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let sample_count = 4;
    // let sample_count = 1;

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let (grid_buffer, grid_color_buffer, grid_instances) = {
        // let input = &app.paf_input;
        let x_axis = &app.alignment_grid.x_axis;
        let y_axis = &app.alignment_grid.y_axis;
        let instances = x_axis.tile_count() + y_axis.tile_count() + 4;
        let mut lines: Vec<LineVertex> = Vec::with_capacity(instances);

        let x_max = x_axis.total_len as f32;
        let y_max = y_axis.total_len as f32;

        // X
        for x_u in x_axis.offsets() {
            let x = x_u as f32;
            lines.push(LineVertex {
                p0: [x, 0f32],
                p1: [x, y_max],
            });
        }
        lines.push(LineVertex {
            p0: [x_max, 0f32],
            p1: [x_max, y_max],
            // color,
        });

        // Y
        for y_u in y_axis.offsets() {
            let y = y_u as f32;
            lines.push(LineVertex {
                p0: [0f32, y],
                p1: [x_max, y],
            });
        }

        lines.push(LineVertex {
            p0: [0f32, y_max],
            p1: [x_max, y_max],
            // color,
        });

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let color_data = vec![0xFF000000u32; lines.len()];

        let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&color_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, color_buffer, 0..instances as u32)
    };

    /*
    let (match_buffer, match_color_buffer, match_instances) = {
        let input = &app.paf_input;
        let instance_count = input.total_matches();

        let mut lines: Vec<LineVertex> = Vec::with_capacity(instance_count);
        let mut colors: Vec<u32> = Vec::with_capacity(instance_count);

        // let color = 0x00000000;

        for line in &input.processed_lines {
            // for &[from, to] in line.match_edges.iter() {
            for (&[from, to], &is_match) in std::iter::zip(&line.match_edges, &line.match_is_match)
            {
                lines.push(LineVertex {
                    p0: [from.x as f32, from.y as f32].into(),
                    p1: [to.x as f32, to.y as f32].into(),
                });

                let col_match = 0xFF000000;
                // let col_mismatch = 0xFFFFFFFF;
                let col_mismatch = 0xFF0000FF;
                // let col_mismatch = 0xFFFFFFFF;

                if is_match {
                    colors.push(col_match);
                } else {
                    colors.push(col_mismatch);
                }
            }
        }

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, color_buffer, 0..instance_count as u32)
    };
    */

    let initial_view = View {
        x_min: 0.0,
        x_max: app.alignment_grid.x_axis.total_len as f64,
        y_min: 0.0,
        y_max: app.alignment_grid.y_axis.total_len as f64,
    };

    let mut app_view = View {
        x_min: 0.0,
        x_max: app.alignment_grid.x_axis.total_len as f64,
        y_min: 0.0,
        y_max: app.alignment_grid.y_axis.total_len as f64,
    };

    // let mut app_view = View {
    //     x_min: 30667021.681820594,
    //     y_min: 28324452.4998267,
    //     x_max: 30667091.79049289,
    //     y_max: 28324528.035320908,
    // };

    // let mut app_view = View {
    //     x_min: 500.0,
    //     x_max: 1200.0,
    //     y_min: 198_000.0,
    //     y_max: 191_500.0,
    // };

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let mut paf_renderer = PafRenderer::new(
        &device,
        config.format,
        sample_count,
        // match_buffer,
        // match_color_buffer,
        // match_instances,
    );

    let match_draw_data = render::batch::MatchDrawBatchData::from_alignments(
        &device,
        &paf_renderer.line_pipeline.bind_group_layout_1,
        &app.alignment_grid,
        &app.alignments,
    );

    paf_renderer.set_grid(Some((grid_buffer, grid_color_buffer, grid_instances)));

    let mut egui_renderer = EguiRenderer::new(&device, &config, swapchain_format, None, 1, &window);
    // egui_renderer.initialize(&window);

    let mut label_physics = LabelPhysics::default();
    label_physics.heightfields =
        annotations::physics::AlignmentHeightFields::from_alignments(&app.alignments);
    let mut labels_to_prepare: Vec<annotations::AnnotationId> = Vec::new();

    // egui_renderer.context.run(
    // egui_renderer.context

    // let mut cpu_rasterizer = egui_renderer.context.fonts(|fonts| {
    //
    //     exact::CpuViewRasterizerEgui::initialize(fonts)
    // });
    let mut cpu_rasterizer = exact::CpuViewRasterizerEgui::initialize();

    let mut window_states = AppWindowStates::new(&app.annotations);

    let mut roi_gui = gui::regions::RegionsOfInterestGui::default();

    let mut annotation_painter = AnnotationPainter::default();

    let event_loop_proxy = event_loop.create_proxy();

    let args = pafview::cli::Cli::parse();

    if let Some(bed_path) = &args.bed {
        event_loop_proxy
            .send_event(AppEvent::LoadAnnotationFile {
                path: bed_path.into(),
            })
            .unwrap();
    }

    let quit_signal = Arc::new(AtomicBool::from(false));

    // {
    let signal = quit_signal.clone();
    ctrlc::set_handler(move || {
        println!("signaling exit!");
        signal.store(true, std::sync::atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
    // }

    // TODO build this on a separate thread
    // let rstar_match = spatial::RStarMatches::from_paf(&input);

    let mut selection_handler = SelectionHandler::default();

    let mut mouse_down = false;
    let mut last_pos = None;
    let mut delta = DVec2::new(0.0, 0.0);

    let mut delta_scale = 1.0;

    let mut last_frame = std::time::Instant::now();

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter);

            if let Event::LoopExiting = event {
                println!("closing!");

                if let Err(err) = config::save_app_config(&app.app_config) {
                    log::error!("Error saving app configuration to file: {err:?}");
                }
            }

            if let Event::UserEvent(event) = event {
                match event {
                    AppEvent::LoadAnnotationFile { path } => {
                        // TODO check file extension maybe, other logic
                        match app.annotations.load_bed_file(
                            &app.alignment_grid,
                            &mut annotation_painter,
                            &path,
                        ) {
                            Ok(list_id) => {
                                let annot_ids =
                                    app.annotations.list_by_id(list_id).into_iter().flat_map(
                                        |list| {
                                            list.records
                                                .iter()
                                                .enumerate()
                                                .map(|(record_id, _)| (list_id, record_id))
                                        },
                                    );

                                labels_to_prepare.extend(annot_ids);
                                // label_physics.prepare_annotations(
                                //     &app.alignment_grid,
                                //     &app.annotations,
                                //     annot_ids,
                                //     fonts,
                                //     annotation_painter,
                                // );

                                log::info!("Loaded BED file `{path:?}`");
                            }
                            Err(err) => {
                                log::error!("Error loading BED file at path `{path:?}`: {err:?}")
                            }
                        }
                    }
                    AppEvent::RequestSelection { target } => {
                        selection_handler.set_target_if_not_selecting(target);
                    }
                }

                return;
            }

            if let Event::AboutToWait = event {
                if quit_signal.load(std::sync::atomic::Ordering::SeqCst) {
                    println!("received quit signal");
                    target.exit();
                }

                let result = device.poll(wgpu::Maintain::Poll);
                if let wgpu::MaintainResult::SubmissionQueueEmpty = result {
                    if delta.x != 0.0 || delta.y != 0.0 || delta_scale != 1.0 {
                        println!("redraw!");
                        window.request_redraw();
                    }
                }
            }

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                let resp = egui_renderer.handle_input(&window, &event);
                if resp.repaint {
                    window.request_redraw();
                }
                if resp.consumed {
                    return;
                }
                // TODO block further input when appropriate (e.g. don't pan when dragging over a window)
                match event {
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            match state {
                                winit::event::ElementState::Pressed => mouse_down = true,
                                winit::event::ElementState::Released => {
                                    last_pos = None;
                                    mouse_down = false
                                }
                            }
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if let PhysicalKey::Code(KeyCode::F12) = event.physical_key {
                            if event.state != ElementState::Pressed {
                                return;
                            }
                            app_view = initial_view;

                            /*
                            // if event.state
                            let [w, h]: [u32; 2] = window.inner_size().into();
                            let path = "screenshot.png";
                            log::info!("taking screenshot");
                            match write_png(&app.paf_input, &app_view, w as usize, h as usize, path)
                            {
                                Ok(_) => log::info!("wrote screenshot to {path}"),
                                Err(e) => log::info!("error writing screenshot: {e:?}"),
                            }
                            */
                        }
                    }
                    WindowEvent::MouseWheel { delta, phase, .. } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => {
                            delta_scale = 1.0 - y as f64 * 0.01;
                        }
                        winit::event::MouseScrollDelta::PixelDelta(xy) => {
                            delta_scale = 1.0 - xy.y * 0.001;
                        }
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        let pos = DVec2::new(position.x, position.y);
                        if mouse_down {
                            // TODO make panning 1-to-1
                            // let vwidth = 2.0 / projection[0][0];
                            // let vheight = 2.0 / projection[1][1];
                            if let Some(last) = last_pos {
                                delta = (pos - last) * DVec2::new(1.0, -1.0);
                            }
                            last_pos = Some(pos);
                        }
                    }
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size

                        let old_size = [config.width, config.height];

                        if new_size.width > 0 && new_size.height > 0 {
                            app_view = app_view.resize_for_window_size(old_size, new_size);
                        }

                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        // msaa_framebuffer = create_multisampled_framebuffer(
                        //     &device,
                        //     [config.width, config.height],
                        //     config.format,
                        //     sample_count,
                        // );
                        egui_renderer.resize(&device, &config, 1);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let delta_t = last_frame.elapsed().as_secs_f64();
                        // log::info!("delta_t: {} ms", last_frame.elapsed().as_millis());

                        let win_size: [u32; 2] = window.inner_size().into();

                        let screen_size = egui_renderer.context.screen_rect().size();

                        if delta.x != 0.0 || delta.y != 0.0 {
                            let dx = -delta.x * app_view.width() / win_size[0] as f64;
                            let dy = -delta.y * app_view.height() / win_size[1] as f64;
                            app_view.translate(dx, dy);
                        }

                        if delta_scale != 1.0 {
                            if let Some(pos) = egui_renderer.context.pointer_latest_pos() {
                                let [px, py]: [f32; 2] = pos.into();

                                let x = px / screen_size.x;
                                let y = py / screen_size.y;

                                app_view.zoom_with_focus([x as f64, y as f64], delta_scale);
                            }
                        }

                        app_view.apply_limits(win_size);

                        delta = DVec2::new(0.0, 0.0);
                        delta_scale = 1.0;

                        let viewport = {
                            // TODO: move all logic to Viewport; for now just construct each frame
                            // should take menu bar into account
                            let center = DVec2::new(
                                (app_view.x_max + app_view.x_min) * 0.5,
                                (app_view.y_max + app_view.y_min) * 0.5,
                            );
                            let size = DVec2::new(
                                app_view.x_max - app_view.x_min,
                                app_view.y_max - app_view.y_min,
                            );
                            view::Viewport::new(center, size, [0.0, 0.0], screen_size)
                        };

                        let debug_painter = egui_renderer.context.debug_painter();

                        label_physics.update_anchors(
                            &debug_painter,
                            &app.alignment_grid,
                            &viewport,
                        );
                        // label_physics.update_labels(
                        //     &debug_painter,
                        //     &app.alignment_grid,
                        //     &app.annotations,
                        //     &mut annotation_painter,
                        //     &viewport,
                        // );
                        label_physics.update_labels_new(
                            &debug_painter,
                            &app.alignment_grid,
                            &app.annotations,
                            &mut annotation_painter,
                            &viewport,
                        );
                        label_physics.step(&app.alignment_grid, delta_t as f32, &viewport);

                        last_frame = std::time::Instant::now();

                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let frame_view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        paf_renderer.draw(
                            &device,
                            &queue,
                            &app,
                            &mut cpu_rasterizer,
                            &match_draw_data,
                            &app_view,
                            win_size,
                            &frame_view,
                            &mut encoder,
                        );

                        egui_renderer.draw(
                            &device,
                            &queue,
                            &mut encoder,
                            &window,
                            &frame_view,
                            ScreenDescriptor {
                                size_in_pixels: window.inner_size().into(),
                                pixels_per_point: window.scale_factor() as f32,
                            },
                            |ctx| {
                                // cpu_rasterizer.draw_and_display_view_layer(ctx, &app, &app_view);

                                // crate::annotations::label_layout::ebug_window(
                                //     ctx, &app, &app_view,
                                // );

                                pafview::gui::goto::goto_region_window(
                                    ctx,
                                    &mut window_states.goto_region_open,
                                    &app.alignment_grid,
                                    &mut app_view,
                                );

                                #[cfg(debug_assertions)]
                                annotations::physics::debug::label_physics_debug_window(
                                    ctx,
                                    &mut window_states.label_physics_debug_open,
                                    &app,
                                    &label_physics,
                                    &viewport,
                                );

                                selection_handler.run(ctx, &mut app_view);
                                // regions::paf_line_debug_aabbs(&input, ctx, &app_view);
                                // annotations::draw_annotation_test_window(
                                //     &app.seq_names,
                                //     &app.paf_input,
                                //     ctx,
                                //     &app_view,
                                // );

                                // gui::debug::line_width_control(ctx, &mut paf_renderer);

                                gui::MenuBar::show(ctx, &app, &mut window_states);

                                // exact_render_view_dbg.show(ctx, &app, win_size, &app_view);
                                // exact_render_dbg.show(ctx, &app);

                                roi_gui.show_window(
                                    ctx,
                                    &app,
                                    &event_loop_proxy,
                                    &mut annotation_painter,
                                    &mut app_view,
                                    &mut window_states,
                                );

                                // gui::view_controls(
                                //     ctx,
                                //     &app.alignment_grid,
                                //     &app.seq_names,
                                //     &app.paf_input,
                                //     &mut app_view,
                                //     &mut window_states,
                                // );

                                // annot_gui_handler.show_annotation_list(
                                //     ctx,
                                //     &app,
                                //     &mut window_states,
                                // );
                                // annot_gui_handler.draw_annotations(ctx, &app, &mut app_view);

                                gui::draw_cursor_position_rulers(
                                    &app.alignment_grid,
                                    &app.alignment_grid.sequence_names,
                                    ctx,
                                    &app_view,
                                );

                                annotation_painter.draw(ctx, &app_view);
                            },
                        );

                        queue.submit(Some(encoder.finish()));

                        // prepare newly loaded annotations as labels in the physics system
                        if labels_to_prepare.len() > 0 {
                            println!(
                                "preparing {} annotations for display",
                                labels_to_prepare.len()
                            );
                            egui_renderer.context.fonts(|fonts| {
                                label_physics.prepare_annotations(
                                    &app.alignment_grid,
                                    &app.annotations,
                                    labels_to_prepare.drain(..),
                                    fonts,
                                    &mut annotation_painter,
                                );
                            })
                        }

                        // handle label physics (TODO threading!)

                        frame.present();
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

fn start_window(app: PafViewerApp) {
    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event()
        .build()
        .unwrap();
    // let event_loop = EventLoop::<AppEvent>::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new().with_title("PAFView");

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(run(event_loop, window, app));
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(run(event_loop, window, app));
    }
}

/*
pub fn write_png(
    input: &PafInput,
    view: &crate::view::View,
    width: usize,
    height: usize,
    out_path: impl AsRef<std::path::Path>,
) -> anyhow::Result<()> {
    use line_drawing::XiaolinWu;
    // let width = width * 4;
    // let height = height * 4;

    let mut px_bytes = vec![0u8; width * height * 4];

    for channels in bytemuck::cast_slice_mut::<_, [u8; 4]>(&mut px_bytes) {
        for c in channels {
            *c = 255;
        }
    }

    let w = width as i64;
    let h = height as i64;

    let screen_dims = [w as f32, h as f32];

    /*
    let line = XiaolinWu::<f32, i64>::new([100f32, 100.0].into(), [400f32, 400.0].into());

    for ((x, y), val) in line {
        if x < 0 || x >= w || y < 0 || y >= h {
            continue;
        }

        let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut px_bytes);
        let ix = x + y * w;

        let val = (255.0 * (1.0 - val)) as u8;
        let [rgb @ .., a] = &mut pixels[ix as usize];
        *a = 255;
        for chan in rgb {
            *chan = val;
        }
    }
    */

    println!("screen_dims: {w}, {h}");

    let matches = input
        .processed_lines
        .iter()
        .flat_map(|l| l.match_edges.iter());

    let mut min_x = std::i64::MAX;
    let mut min_y = min_x;
    let mut max_x = std::i64::MIN;
    let mut max_y = max_x;
    let mut px_count = 0;

    for &[from, to] in matches {
        let s_from = view.map_world_to_screen(screen_dims, from);
        let s_to = view.map_world_to_screen(screen_dims, to);

        let start: [f32; 2] = s_from.into();
        let end: [f32; 2] = s_to.into();

        let line = XiaolinWu::<f32, i64>::new(start.into(), end.into());

        // for ((x, y), val) in line {
        // let mut seen = false;
        for ((x, y), val) in line {
            // x & y are in screen coordinates
            if x < 0 || x >= w || y < 0 || y >= h {
                continue;
            }
            px_count += 1;

            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            // if !seen {
            //     println!("{x}, {y}");
            //     seen = true;
            // }

            let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut px_bytes);
            let ix = x + y * w;

            let val = (255.0 * (1.0 - val)) as u8;
            let [rgb @ .., a] = &mut pixels[ix as usize];
            *a = 255;
            for chan in rgb {
                *chan = val;
            }
        }
    }

    println!("min_x: {min_x}\tmin_y: {min_y}");
    println!("max_x: {max_x}\tmax_y: {max_y}");
    println!("touched pixels {px_count} times");

    lodepng::encode32_file(out_path, &px_bytes, width, height)?;

    Ok(())
}
*/
