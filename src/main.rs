use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use std::io::prelude::*;

use anyhow::anyhow;

mod render;

use render::*;

struct PafInput {
    queries: Vec<AlignedSeq>,
    targets: Vec<AlignedSeq>,
    target_len: usize,
    query_len: usize,

    match_edges: Vec<[Vec2; 2]>,
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
            let qi = names.query_names.len();
            let ti = names.target_names.len();

            let query_name = paf_line.query_name.to_string();
            let target_name = paf_line.query_name.to_string();

            names.query_names.insert(query_name.clone(), qi);
            names.target_names.insert(target_name.clone(), ti);

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

        let target_i = names.target_names.get(paf_line.tgt_name).unwrap();
        let query_i = names.query_names.get(paf_line.query_name).unwrap();

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

    let mut min = Vec2::broadcast(std::f32::MAX);
    let mut max = Vec2::broadcast(std::f32::MIN);

    for &[from, to] in paf_input.match_edges.iter() {
        min = min.min_by_component(from).min_by_component(to);
        max = max.max_by_component(from).max_by_component(to);
    }

    println!("AABB: min {min:?}, max {max:?}");

    start_window(paf_input);

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
    match_edges: &mut Vec<[Vec2; 2]>,
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

                // TODO output match
                {
                    let x0 = x as f32 / target_len as f32;
                    let y0 = y as f32 / query_len as f32;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f32 / target_len as f32;
                    let y1 = (y + count) as f32 / query_len as f32;

                    match_edges.push([Vec2::new(x0, y0), Vec2::new(x1, y1)]);
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
    match_edges: &mut Vec<[Vec2; 2]>,
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
                    let x0 = x as f32 / target_len as f32;
                    let y0 = y as f32 / query_len as f32;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f32 / target_len as f32;
                    let y1 = (y + count) as f32 / query_len as f32;

                    match_edges.push([Vec2::new(x0, y0), Vec2::new(x1, y1)]);
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

fn create_multisampled_framebuffer(
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

async fn run(event_loop: EventLoop<()>, window: Window, input: PafInput) {
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
            let x0 = from.x * input.target_len as f32;
            let y0 = from.y * input.query_len as f32;
            let x1 = to.x * input.target_len as f32;
            let y1 = to.y * input.query_len as f32;

            lines.push(LineVertex {
                p0: [x0, y0].into(),
                p1: [x1, y1].into(),
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

    let xbl = input.target_len as f32 / 16.0;
    let ybl = input.query_len as f32 / 16.0;

    let mut projection = ultraviolet::projection::orthographic_wgpu_dx(
        // 14.0 * xbl,
        // 15.0 * xbl,
        // 14.0 * ybl,
        // 15.0 * ybl,
        // 0.0,
        // input.target_len as f32 / 4.0,
        // 0.0,
        // input.query_len as f32 / 4.0,
        0.0,
        input.target_len as f32,
        0.0,
        input.query_len as f32,
        0.1,
        10.0,
    );

    let (proj_uniform, line_conf_uniform, short_conf_uniform) = {
        let proj_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[projection]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // px / window width
        let line_width: f32 = 5.0 / 1000.0;
        // let line_width: f32 = 15.0 / 1000.0;
        let conf = [line_width, 0.0, 0.0, 0.0];
        let line_conf_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&conf),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let frag_width = 0.5 / size.width as f32;
        let frag_height = 0.5 / size.height as f32;

        let conf = [frag_width, frag_height, 0.0, 0.0];
        let short_conf_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&conf),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        (proj_uniform, line_conf_uniform, short_conf_uniform)
    };

    let line_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &line_pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: proj_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: line_conf_uniform.as_entire_binding(),
            },
        ],
    });

    let short_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &short_pipeline.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: proj_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: short_conf_uniform.as_entire_binding(),
            },
        ],
    });

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let mut msaa_framebuffer = create_multisampled_framebuffer(&device, &config, sample_count);

    let mut mouse_down = false;
    let mut last_pos = Vec2::new(0.0, 0.0);
    let mut delta = Vec2::new(0.0, 0.0);

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
                match event {
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            match state {
                                winit::event::ElementState::Pressed => mouse_down = true,
                                winit::event::ElementState::Released => mouse_down = false,
                            }
                        }
                    }
                    WindowEvent::MouseWheel { delta, phase, .. } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => {
                            delta_scale = 1.0 + y * 0.01;
                        }
                        winit::event::MouseScrollDelta::PixelDelta(xy) => {
                            delta_scale = 1.0 + xy.y as f32 * 0.001;
                        }
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        let pos = Vec2::new(position.x as f32, position.y as f32);
                        if mouse_down {
                            // TODO make panning 1-to-1
                            // let vwidth = 2.0 / projection[0][0];
                            // let vheight = 2.0 / projection[1][1];
                            delta = (pos - last_pos) * Vec2::new(1.0, -1.0);
                        }

                        last_pos = pos;
                    }
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        msaa_framebuffer =
                            create_multisampled_framebuffer(&device, &config, sample_count);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let delta_t = last_frame.elapsed().as_secs_f64();

                        if delta.x != 0.0 || delta.y != 0.0 || delta_scale != 1.0 {
                            projection = Mat4::from_nonuniform_scale(Vec3::new(
                                delta_scale,
                                delta_scale,
                                1.0,
                            )) * Mat4::from_translation(
                                delta_t as f32 * Vec3::new(delta.x, delta.y, 0.0),
                            ) * projection;
                            queue.write_buffer(
                                &proj_uniform,
                                0,
                                bytemuck::cast_slice(&[projection]),
                            );
                        }
                        delta = Vec2::new(0.0, 0.0);
                        delta_scale = 1.0;

                        last_frame = std::time::Instant::now();

                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });
                        {
                            let attch = if sample_count == 1 {
                                wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }
                            } else {
                                wgpu::RenderPassColorAttachment {
                                    view: &msaa_framebuffer,
                                    resolve_target: Some(&view),
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
                            rpass.set_pipeline(&line_pipeline.pipeline);
                            rpass.set_bind_group(0, &line_bind_group, &[]);
                            // rpass.set_pipeline(&short_pipeline.pipeline);
                            // rpass.set_bind_group(0, &short_bind_group, &[]);

                            // first draw grid
                            // rpass.set_vertex_buffer(0, grid_buffer.slice(..));
                            // rpass.draw(0..6, grid_instances.clone());

                            // then matches
                            rpass.set_vertex_buffer(0, match_buffer.slice(..));
                            rpass.draw(0..6, match_instances.clone());
                        }

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

pub fn start_window(input: PafInput) {
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
        pollster::block_on(run(event_loop, window, input));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window, input));
    }
}
