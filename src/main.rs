use annotations::{draw::AnnotationPainter, AnnotationGuiHandler};
use bimap::BiMap;
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use egui_wgpu::ScreenDescriptor;
use grid::AlignmentGrid;
use paf::Alignments;
use regions::SelectionHandler;
use rustc_hash::FxHashMap;
use sequences::Sequences;
use std::{borrow::Cow, str::FromStr, sync::Arc};
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

pub mod cigar;
pub mod paf;
pub mod pixels;

mod annotations;
mod cli;
mod grid;
mod gui;
mod regions;
mod render;
mod sequences;
mod view;

pub use cigar::*;
pub use paf::{PafInput, PafLine};
pub use pixels::*;
use render::*;
use view::View;

use crate::{
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

pub fn test_main() -> anyhow::Result<()> {
    let mut args = std::env::args();
    let paf_path = args.nth(1).ok_or(anyhow!("Path to PAF not provided"))?;

    // let reader = std::fs::File::open(&
    // let paf_input = PafInput::read_paf_file(&paf_path)?;

    Ok(())
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

    let args = crate::cli::Cli::parse();

    // Load PAF and optional FASTA
    let (alignments, sequences) = crate::paf::load_input_files(&args.paf, args.fasta)?;

    println!("drawing {} alignments", alignments.pairs.len());

    // construct AlignmentGrid
    let mut targets = alignments
        .pairs
        .values()
        .map(|al| (al.target_id, al.location.target_total_len))
        .collect::<Vec<_>>();
    targets.sort_by_key(|(_, l)| *l);
    targets.dedup_by_key(|(id, _)| *id);
    let x_axis = grid::GridAxis::from_index_and_lengths(targets);
    let mut queries = alignments
        .pairs
        .values()
        .map(|al| (al.query_id, al.location.query_total_len))
        .collect::<Vec<_>>();
    queries.sort_by_key(|(_, l)| *l);
    queries.dedup_by_key(|(id, _)| *id);
    let y_axis = grid::GridAxis::from_index_and_lengths(queries);

    println!(
        "X axis {} tiles, total len {}",
        x_axis.tile_count(),
        x_axis.total_len
    );
    println!(
        "Y axis {} tiles, total len {}",
        y_axis.tile_count(),
        y_axis.total_len
    );

    let alignment_grid = AlignmentGrid {
        x_axis,
        y_axis,
        sequence_names: sequences.names().clone(),
    };

    // TODO replace PafInput everywhere...

    let seq_names = sequences.names().clone();

    let app = PafViewerApp {
        alignments,
        alignment_grid,
        sequences,
        // paf_input: todo!(),
        seq_names,
        annotations: AnnotationStore::default(),
    };

    start_window(app);

    Ok(())
}

/*
pub fn old_main() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
    }

    let mut args = std::env::args();
    let paf_path = args.nth(1).ok_or(anyhow!("Path to PAF not provided"))?;

    let bed_path = args.next();

    // parse paf
    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

    let mut target_names = FxHashMap::default();
    let mut query_names = FxHashMap::default();

    let mut targets = Vec::<AlignedSeq>::new();
    let mut queries = Vec::<AlignedSeq>::new();

    for line in reader.lines() {
        let line = line?;
        if let Some(paf_line) = parse_paf_line(line.split('\t')) {
            let query_name = paf_line.query_name.to_string();
            let target_name = paf_line.tgt_name.to_string();

            if !query_names.contains_key(&query_name) {
                query_names.insert(query_name.clone(), queries.len());
                queries.push(AlignedSeq {
                    name: query_name,
                    len: paf_line.query_seq_len,
                    ..AlignedSeq::default()
                });
            }

            if !target_names.contains_key(&target_name) {
                target_names.insert(target_name.clone(), targets.len());
                targets.push(AlignedSeq {
                    name: target_name,
                    len: paf_line.tgt_seq_len,
                    ..AlignedSeq::default()
                });
            }
        }
    }

    targets.sort_by_key(|seq| std::cmp::Reverse(seq.len));
    queries.sort_by_key(|seq| std::cmp::Reverse(seq.len));

    let mut all_seqs = targets.iter().chain(&queries).cloned().collect::<Vec<_>>();
    all_seqs.sort_by_cached_key(|seq| (std::cmp::Reverse(seq.len), seq.name.clone()));
    all_seqs.dedup_by_key(|seq| (std::cmp::Reverse(seq.len), seq.name.clone()));

    let process_aligned = |names: &mut FxHashMap<String, usize>, aseqs: &mut [AlignedSeq]| {
        let mut offset = 0;

        for (new_id, seq) in aseqs.iter_mut().enumerate() {
            if let Some(id) = names.get_mut(&seq.name) {
                *id = new_id;
            }
            seq.offset = offset;
            offset += seq.len;
        }

        offset
    };

    let target_len = process_aligned(&mut target_names, &mut targets);
    let query_len = process_aligned(&mut query_names, &mut queries);

    let seq_names = target_names
        .iter()
        .chain(&query_names)
        .map(|(n, i)| (n.clone(), SeqId(*i)))
        .collect::<bimap::BiMap<_, _>>();
    let seq_names = Arc::new(seq_names);

    let x_axis = grid::GridAxis::from_sequences(&seq_names, &targets);
    let y_axis = grid::GridAxis::from_sequences(&seq_names, &queries);

    // process matches
    let mut processed_lines = Vec::new();

    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

    let mut pair_line_ix = FxHashMap::default();

    for line in reader.lines() {
        let line = line?;
        let Some(paf_line) = parse_paf_line(line.split('\t')) else {
            continue;
        };

        let target = seq_names.get_by_left(paf_line.tgt_name);
        let query = seq_names.get_by_left(paf_line.query_name);

        let (Some(target_i), Some(query_i)) = (target, query) else {
            continue;
        };

        // let origin = {
        //     let x0 = x_axis.sequence_offset(*target_i).unwrap();
        //     let y0 = y_axis.sequence_offset(*query_i).unwrap();
        //     // let x0 = &targets[*target_i].offset;
        //     // let y0 = &queries[*query_i].offset;

        //     let x = x0 + paf_line.tgt_seq_start;
        //     let y = if paf_line.strand_rev {
        //         y0 + paf_line.query_seq_end
        //     } else {
        //         y0 + paf_line.query_seq_start
        //     };
        //     [x as u64, y as u64]
        // };

        pair_line_ix.insert((*target_i, *query_i), processed_lines.len());
        processed_lines.push(ProcessedCigar::from_line_local(&seq_names, &paf_line)?);
        // processed_lines.push(ProcessedCigar::from_line(&seq_names, &paf_line, origin)?);

        // process_cigar(&paf_line, origin, &mut match_edges)?;
        // process_cigar_compress(&paf_line, origin, target_len, query_len, &mut match_edges)?;
    }

    let paf_input = PafInput {
        queries,
        targets,
        pair_line_ix,
        processed_lines,
    };

    let mut annotations = AnnotationStore::default();

    println!("sum target len: {target_len}");
    println!("sum query len: {query_len}");
    let total_matches: usize = paf_input
        .processed_lines
        .iter()
        .map(|l| l.match_edges.len())
        .sum();
    println!("drawing {} matches", total_matches);

    let alignment_grid = AlignmentGrid {
        x_axis,
        y_axis,
        sequence_names: seq_names.clone(),
    };

    let app = PafViewerApp {
        alignment_grid,
        paf_input,
        seq_names,
        annotations,
    };

    start_window(app);

    Ok(())
}
*/

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: [f32; 2],
    p1: [f32; 2],
    // color: u32,
}

#[derive(Clone)]
pub enum AppEvent {
    LoadAnnotationFile { path: std::path::PathBuf },
    // AnnotationShapeDisplay {
    //     shape_id: annotations::draw::AnnotShapeId,
    //     enable: Option<bool>,
    // },

    // idk if this is a good idea but worth a try
    RequestSelection { target: regions::SelectionTarget },
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

    let mut app_view = View {
        x_min: 0.0,
        x_max: app.alignment_grid.x_axis.total_len as f64,
        y_min: 0.0,
        y_max: app.alignment_grid.y_axis.total_len as f64,
    };

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

    // egui_renderer.context.run(
    // egui_renderer.context

    // let mut cpu_rasterizer = egui_renderer.context.fonts(|fonts| {
    //
    //     exact::CpuViewRasterizerEgui::initialize(fonts)
    // });
    let mut cpu_rasterizer = exact::CpuViewRasterizerEgui::initialize();

    let mut window_states = AppWindowStates::new(&app.annotations);

    let mut annot_gui_handler = AnnotationGuiHandler::default();

    let mut roi_gui = gui::regions::RegionsOfInterestGui::default();

    let mut annotation_painter = AnnotationPainter::default();

    let event_loop_proxy = event_loop.create_proxy();

    if let Some(bed_path) = std::env::args().nth(2) {
        if let Ok(path) = std::path::PathBuf::from_str(&bed_path) {
            event_loop_proxy.send_event(AppEvent::LoadAnnotationFile { path });
        }
    }

    // TODO build this on a separate thread
    // let rstar_match = spatial::RStarMatches::from_paf(&input);

    let mut selection_handler = SelectionHandler::default();

    // let mut exact_render_dbg = exact::ExactRenderDebug::default();
    // let mut exact_render_view_dbg = exact::ExactRenderViewDebug::default();

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

            if let Event::UserEvent(event) = event {
                match event {
                    AppEvent::LoadAnnotationFile { path } => {
                        // TODO check file extension maybe, other logic
                        match app.annotations.load_bed_file(
                            &app.alignment_grid,
                            &mut annotation_painter,
                            &path,
                        ) {
                            Ok(_) => {
                                log::info!("Loaded BED file `{path:?}`");
                            }
                            Err(err) => {
                                log::error!("Error loading BED file at path `{path:?}`: {err:?}")
                            }
                        }
                    }
                    AppEvent::RequestSelection { target } => {
                        // ignore if someone's already waiting for a region selection
                        if !selection_handler.has_active_selection_request() {
                            selection_handler.selection_target = Some(target);
                        }
                    }
                }

                return;
            }

            if let Event::AboutToWait = event {
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

                                annot_gui_handler.show_annotation_list(
                                    ctx,
                                    &app,
                                    &mut window_states,
                                );
                                annot_gui_handler.draw_annotations(ctx, &app, &mut app_view);

                                gui::draw_cursor_position_rulers(
                                    &app.alignment_grid,
                                    &app.seq_names,
                                    ctx,
                                    &app_view,
                                );

                                annotation_painter.draw(ctx, &app_view);
                            },
                        );

                        queue.submit(Some(encoder.finish()));
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

struct PafViewerApp {
    alignments: Alignments,
    alignment_grid: AlignmentGrid,
    sequences: Sequences,

    // paf_input: PafInput,
    #[deprecated]
    seq_names: Arc<bimap::BiMap<String, SeqId>>,
    annotations: AnnotationStore,
}
