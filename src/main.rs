use bytemuck::{Pod, Zeroable};
use egui_wgpu::ScreenDescriptor;
use regions::SelectionHandler;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use ultraviolet::{DVec2, Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use std::io::prelude::*;

use anyhow::anyhow;

mod gui;
mod regions;
mod render;
mod spatial;
mod view;

use render::*;
use view::View;

struct PafInput {
    queries: Vec<AlignedSeq>,
    targets: Vec<AlignedSeq>,
    target_len: usize,
    query_len: usize,

    match_edges: Vec<[DVec2; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PafLine<S> {
    query_name: S,
    query_seq_len: usize,
    query_seq_start: usize,
    query_seq_end: usize,

    tgt_name: S,
    tgt_seq_len: usize,
    tgt_seq_start: usize,
    tgt_seq_end: usize,

    strand_rev: bool,
    cigar: S,
}

#[derive(Debug, Default, Clone)]
struct AlignedSeq {
    // name of the given sequence
    name: String,
    // its length
    len: usize,
    // its rank among other seqs in the query or target set
    rank: usize,
    // its start offset in the global all-to-all alignment matrix
    offset: usize,
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
// struct AlignedSeq {
//     name: String,
//     len: usize,
// }

#[derive(Default)]
struct NameCache {
    query_names: FxHashMap<String, usize>,
    target_names: FxHashMap<String, usize>,
}

fn parse_paf_line<'a>(mut fields: impl Iterator<Item = &'a str>) -> Option<PafLine<&'a str>> {
    let (query_name, query_seq_len, query_seq_start, query_seq_end) =
        parse_name_range(&mut fields)?;
    let strand = fields.next()?;
    let (tgt_name, tgt_seq_len, tgt_seq_start, tgt_seq_end) = parse_name_range(&mut fields)?;

    let cigar = fields.skip(3).find_map(|s| s.strip_prefix("cg:Z:"))?;

    Some(PafLine {
        query_name,
        query_seq_len,
        query_seq_start,
        query_seq_end,

        tgt_name,
        tgt_seq_len,
        tgt_seq_start,
        tgt_seq_end,

        strand_rev: strand == "-",
        cigar,
    })
}

fn parse_name_range<'a>(
    mut fields: impl Iterator<Item = &'a str>,
) -> Option<(&'a str, usize, usize, usize)> {
    let name = fields.next()?;
    let len = fields.next()?.parse().ok()?;
    let start = fields.next()?.parse().ok()?;
    let end = fields.next()?.parse().ok()?;
    Some((name, len, start, end))
}

pub fn write_png(
    input: &PafInput,
    width: usize,
    height: usize,
    out_path: impl AsRef<std::path::Path>,
) -> anyhow::Result<()> {
    use line_drawing::XiaolinWu;
    let mut px_bytes = vec![0u8; width * height * 4];

    let view = View {
        x_min: 0.0,
        y_min: 0.0,
        x_max: input.target_len as f64,
        y_max: input.query_len as f64,
    };

    let w = width as i64;
    let h = height as i64;

    let screen_dims = [w as f32, h as f32];

    let matches = &input.match_edges;

    for &[from, to] in matches {
        let s_from = view.map_world_to_screen(screen_dims, from);
        let s_to = view.map_world_to_screen(screen_dims, to);

        let start: [f32; 2] = s_from.into();
        let end: [f32; 2] = s_to.into();

        let line = XiaolinWu::<f32, i64>::new(start.into(), end.into());

        println!("   match: ({from:?}, {to:?})");
        println!("  screen: ({s_from:?}, {s_to:?})");

        for ((x, y), val) in line {
            println!(": {x}, {y}");
            // x & y are in screen coordinates
            if x < 0 || x >= w || y < 0 || y >= h {
                continue;
            }

            let pixels: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut px_bytes);
            let ix = x + y * w;

            let val = (255.0 * val) as u8;
            let [rgb @ .., a] = &mut pixels[ix as usize];
            *a = 255;
            for chan in rgb {
                *chan = val;
            }
        }
    }

    lodepng::encode32_file(out_path, &px_bytes, width, height)?;
    // let mut writer = std::fs::File::new(out_path).map(std::io::BufWriter::new)?;
    // let mut encoder = png::Encoder::new(writer,

    // let mut writer = std::io::Bu

    Ok(())
}

pub fn main() -> anyhow::Result<()> {
    let mut args = std::env::args();
    let paf_path = args.nth(1).ok_or(anyhow!("Path to PAF not provided"))?;

    // parse paf
    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

    let mut names = NameCache::default();

    let mut queries = Vec::<AlignedSeq>::new();
    let mut targets = Vec::<AlignedSeq>::new();

    for line in reader.lines() {
        let line = line?;
        if let Some(paf_line) = parse_paf_line(line.split('\t')) {
            let query_name = paf_line.query_name.to_string();
            let target_name = paf_line.tgt_name.to_string();

            names.query_names.insert(query_name.clone(), queries.len());
            names
                .target_names
                .insert(target_name.clone(), targets.len());

            queries.push(AlignedSeq {
                name: query_name,
                len: paf_line.query_seq_len,
                ..AlignedSeq::default()
            });
            targets.push(AlignedSeq {
                name: target_name,
                len: paf_line.tgt_seq_len,
                ..AlignedSeq::default()
            });
        }
    }

    let process_aligned = |aseqs: &mut [AlignedSeq]| {
        let mut by_len = aseqs.iter().map(|a| a.len).enumerate().collect::<Vec<_>>();
        by_len.sort_by_key(|(_id, len)| *len);

        let mut offset = 0;

        for (rank, (i, _len)) in by_len.into_iter().enumerate() {
            let entry = &mut aseqs[i];
            entry.rank = rank;
            entry.offset = offset;
            offset += entry.len;
        }

        offset
    };

    let target_len = process_aligned(&mut targets);
    let query_len = process_aligned(&mut queries);

    // process matches
    let mut match_edges = Vec::new();

    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

    for line in reader.lines() {
        let line = line?;
        let Some(paf_line) = parse_paf_line(line.split('\t')) else {
            continue;
        };

        let target = names.target_names.get(paf_line.tgt_name);
        let query = names.query_names.get(paf_line.query_name);

        let (Some(target_i), Some(query_i)) = (target, query) else {
            continue;
        };

        let origin = {
            let x0 = &targets[*target_i].offset;
            let y0 = &queries[*query_i].offset;

            let x = x0 + paf_line.tgt_seq_start;
            let y = if paf_line.strand_rev {
                y0 + paf_line.query_seq_end
            } else {
                y0 + paf_line.query_seq_start
            };
            [x, y]
        };

        process_cigar(&paf_line, origin, target_len, query_len, &mut match_edges)?;
        // process_cigar_compress(&paf_line, origin, target_len, query_len, &mut match_edges)?;
    }

    let paf_input = PafInput {
        queries,
        targets,
        target_len,
        query_len,
        match_edges,
    };

    println!("sum target len: {target_len}");
    println!("sum query len: {query_len}");
    println!("drawing {} matches", paf_input.match_edges.len());

    let mut min = DVec2::broadcast(std::f64::MAX);
    let mut max = DVec2::broadcast(std::f64::MIN);

    for &[from, to] in paf_input.match_edges.iter() {
        min = min.min_by_component(from).min_by_component(to);
        max = max.max_by_component(from).max_by_component(to);
    }

    println!("AABB: min {min:?}, max {max:?}");

    start_window(names, paf_input);

    Ok(())
}

// Cigar ops "projected" into a subset, with matches/mismatches combined
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OpProj {
    Match(usize),
    Del(usize),
    Ins(usize),
}

impl OpProj {
    fn count(&self) -> usize {
        match *self {
            OpProj::Match(a) => a,
            OpProj::Del(a) => a,
            OpProj::Ins(a) => a,
        }
    }

    fn with_count(&self, count: usize) -> Self {
        use OpProj::*;
        match *self {
            Match(_) => Match(count),
            Del(_) => Del(count),
            Ins(_) => Ins(count),
        }
    }

    fn is_same_op(&self, other: Self) -> bool {
        use OpProj::*;
        match (*self, other) {
            (Match(_), Match(_)) => true,
            (Del(_), Del(_)) => true,
            (Ins(_), Ins(_)) => true,
            _ => false,
        }
    }
}

fn process_cigar(
    paf: &PafLine<&str>,
    origin: [usize; 2],
    target_len: usize,
    query_len: usize,
    match_edges: &mut Vec<[DVec2; 2]>,
) -> anyhow::Result<()> {
    let ops = paf
        .cigar
        .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
        .filter_map(|opstr| {
            let count = opstr[..opstr.len() - 1].parse::<usize>().ok()?;
            let op = opstr.as_bytes()[opstr.len() - 1] as char;
            Some((op, count))
        });

    let [mut target_pos, mut query_pos] = origin;

    for (op, count) in ops {
        match op {
            'M' | '=' | 'X' => {
                let x = target_pos;
                let y = query_pos;

                {
                    let x0 = x as f64 / target_len as f64;
                    let y0 = y as f64 / query_len as f64;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f64 / target_len as f64;
                    let y1 = (y + count) as f64 / query_len as f64;

                    match_edges.push([DVec2::new(x0, y0), DVec2::new(x1, y1)]);
                }

                // update query pos & target pos
                target_pos += count;
                if paf.strand_rev {
                    query_pos = query_pos.checked_sub(count).unwrap_or_default()
                } else {
                    query_pos += count;
                }
            }
            'D' => {
                target_pos += count;
            }
            'I' => {
                if paf.strand_rev {
                    query_pos = query_pos.checked_sub(count).unwrap_or_default()
                } else {
                    query_pos += count;
                }
            }
            _ => (),
        }
    }

    Ok(())
}

fn process_cigar_compress(
    paf: &PafLine<&str>,
    origin: [usize; 2],
    target_len: usize,
    query_len: usize,
    match_edges: &mut Vec<[DVec2; 2]>,
) -> anyhow::Result<()> {
    use OpProj::*;

    let mut ops = paf
        .cigar
        .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
        .filter_map(|opstr| {
            let count = opstr[..opstr.len() - 1].parse::<usize>().ok()?;
            let op = opstr.as_bytes()[opstr.len() - 1] as char;
            match op {
                'M' | '=' | 'X' => Some(Match(count)),
                'D' => Some(Del(count)),
                'I' => Some(Ins(count)),
                _ => None,
            }
        });
    // .filter(|op| {
    //     let lim = 150;
    //     match op {
    //         Match(_) => true,
    //         Del(c) | Ins(c) => *c > lim,
    //     }
    // });

    let mut compressed_ops = Vec::new();

    let mut last_op = ops.next().ok_or(anyhow!("Empty CIGAR!"))?;

    for op in ops {
        if last_op.is_same_op(op) {
            // combine
            last_op = last_op.with_count(last_op.count() + op.count());
        } else {
            // emit last op
            compressed_ops.push(last_op);
            last_op = op;
        }
    }
    compressed_ops.push(last_op);

    let [mut target_pos, mut query_pos] = origin;

    for op in compressed_ops {
        match op {
            Match(count) => {
                let x = target_pos;
                let y = query_pos;

                {
                    let x0 = x as f64 / target_len as f64;
                    let y0 = y as f64 / query_len as f64;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f64 / target_len as f64;
                    let y1 = (y + count) as f64 / query_len as f64;

                    match_edges.push([DVec2::new(x0, y0), DVec2::new(x1, y1)]);
                }

                // update query pos & target pos
                target_pos += count;
                if paf.strand_rev {
                    query_pos = query_pos.checked_sub(count).unwrap_or_default()
                } else {
                    query_pos += count;
                }
            }
            Del(count) => {
                target_pos += count;
            }
            Ins(count) => {
                if paf.strand_rev {
                    query_pos = query_pos.checked_sub(count).unwrap_or_default()
                } else {
                    query_pos += count;
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: [f32; 2],
    p1: [f32; 2],
    // color: u32,
}

async fn run(event_loop: EventLoop<()>, window: Window, name_cache: NameCache, input: PafInput) {
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

    let line_pipeline = LinePipeline::new(&device, swapchain_format, sample_count);
    let short_pipeline = ShortMatchPipeline::new(&device, swapchain_format, sample_count);

    let (grid_buffer, grid_instances) = {
        let instances = input.targets.len() + input.queries.len();
        let mut lines: Vec<LineVertex> = Vec::with_capacity(instances);

        let mut targets_sort = input
            .targets
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.len))
            .collect::<Vec<_>>();
        let mut queries_sort = input
            .queries
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.len))
            .collect::<Vec<_>>();
        targets_sort.sort_by_key(|(_, l)| *l);
        queries_sort.sort_by_key(|(_, l)| *l);

        let x_max = input.target_len as f32;
        let y_max = input.query_len as f32;

        let color = 0x00000000;

        // X
        for t in input.targets.iter() {
            let x = t.offset as f32;
            lines.push(LineVertex {
                p0: [x, 0f32],
                p1: [x, y_max],
                // color,
            });
        }

        // Y
        for q in input.queries.iter() {
            let y = q.offset as f32;
            lines.push(LineVertex {
                p0: [0f32, y],
                p1: [x_max, y],
                // color,
            });
        }

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, 0..instances as u32)
    };

    let (match_buffer, match_instances) = {
        let instances = input.match_edges.len();
        let mut lines: Vec<LineVertex> = Vec::with_capacity(instances);

        let color = 0x00000000;

        for &[from, to] in input.match_edges.iter() {
            let x0 = from.x * input.target_len as f64;
            let y0 = from.y * input.query_len as f64;
            let x1 = to.x * input.target_len as f64;
            let y1 = to.y * input.query_len as f64;

            lines.push(LineVertex {
                p0: [x0 as f32, y0 as f32].into(),
                p1: [x1 as f32, y1 as f32].into(),
                // color,
            });
        }

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, 0..instances as u32)
    };

    let mut app_view = View {
        x_min: 0.0,
        x_max: input.target_len as f64,
        y_min: 0.0,
        y_max: input.query_len as f64,
    };

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let mut paf_renderer = PafRenderer::new(
        &device,
        config.format,
        sample_count,
        match_buffer,
        match_instances,
    );

    let mut egui_renderer = EguiRenderer::new(&device, &config, swapchain_format, None, 1, &window);

    // let mut msaa_framebuffer = create_multisampled_framebuffer(
    //     &device,
    //     [config.width, config.height],
    //     config.format,
    //     sample_count,
    // );

    let rstar_match = spatial::RStarMatches::from_paf(&input);

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
            let _ = (&instance, &adapter, &line_pipeline, &short_pipeline);

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

                        let win_size: [u32; 2] = window.inner_size().into();

                        // if delta.x != 0.0 || delta.y != 0.0 || delta_scale != 1.0 {
                        if delta.x != 0.0 || delta.y != 0.0 {
                            let dx = -delta.x * app_view.width() / win_size[0] as f64;
                            let dy = -delta.y * app_view.height() / win_size[1] as f64;
                            app_view.translate(dx, dy);
                        }
                        if delta_scale != 1.0 {
                            if let Some(pos) = egui_renderer.context.pointer_latest_pos() {
                                let [w, h] = win_size;
                                let [px, py]: [f32; 2] = pos.into();

                                let x = px / w as f32;
                                let y = py / h as f32;

                                app_view.zoom_with_focus([x as f64, y as f64], delta_scale);
                                // app_view.scale_around_point([x as f64, y as f64], delta_scale);
                            }

                            // let screen_pt = app_view.scale_around_center(delta_scale);
                            // let projection = app_view.to_mat4();
                            // queue.write_buffer(
                            //     &proj_uniform,
                            //     0,
                            //     bytemuck::cast_slice(&[projection]),
                            // );
                        }
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
                            &app_view,
                            win_size,
                            &frame_view,
                            &mut encoder,
                        );

                        /*
                        {
                            let attch = if sample_count == 1 {
                                wgpu::RenderPassColorAttachment {
                                    view: &frame_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }
                            } else {
                                wgpu::RenderPassColorAttachment {
                                    view: &msaa_framebuffer,
                                    resolve_target: Some(&frame_view),
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                        store: wgpu::StoreOp::Discard,
                                    },
                                }
                            };

                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(attch)],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            // rpass.set_pipeline(&line_pipeline.pipeline);
                            // rpass.set_bind_group(0, &line_bind_group, &[]);
                            rpass.set_pipeline(&short_pipeline.pipeline);
                            rpass.set_bind_group(0, &short_bind_group, &[]);

                            // first draw grid
                            // rpass.set_vertex_buffer(0, grid_buffer.slice(..));
                            // rpass.draw(0..6, grid_instances.clone());

                            // then matches
                            rpass.set_vertex_buffer(0, match_buffer.slice(..));
                            rpass.draw(0..6, match_instances.clone());
                        }
                        */

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
                                selection_handler.run(ctx, &mut app_view);
                                gui::draw_cursor_position_rulers(&input, ctx, &app_view);

                                // gui::view_controls(&name_cache, &input, &mut app_view, ctx);
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

pub fn start_window(name_cache: NameCache, input: PafInput) {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
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
        env_logger::init();
        pollster::block_on(run(event_loop, window, name_cache, input));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window, name_cache, input));
    }
}
