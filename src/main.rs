use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use std::borrow::Cow;
use ultraviolet::UVec2;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

use std::io::prelude::*;

use anyhow::anyhow;

struct PafInput {
    queries: Vec<AlignedSeq>,
    targets: Vec<AlignedSeq>,
    target_len: usize,
    query_len: usize,

    matches: Vec<()>,
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
    let mut matches = Vec::new();

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

        let ops = paf_line
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

                    // update query pos & target pos
                    if paf_line.strand_rev {
                        query_pos = query_pos.checked_sub(count).unwrap_or_default()
                    } else {
                        query_pos += count;
                    }
                }
                'D' => {
                    target_pos += count;
                }
                'I' => {
                    if paf_line.strand_rev {
                        query_pos = query_pos.checked_sub(count).unwrap_or_default()
                    } else {
                        query_pos += count;
                    }
                }
                _ => (),
            }
        }
    }

    let paf_input = PafInput {
        queries,
        targets,
        target_len,
        query_len,
        matches,
    };

    println!("sum target len: {target_len}");
    println!("sum query len: {query_len}");

    start_window(paf_input);

    Ok(())
}

#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: [f32; 2],
    p1: [f32; 2],
    color: u32,
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
                required_features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
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

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 5 * std::mem::size_of::<u32>() as u64,
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
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 16,
                        shader_location: 2,
                    },
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
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

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
            // for (i, _) in targets_sort.into_iter().take(1) {
            // let t = &input.targets[i];
            let x = t.offset as f32;
            lines.push(LineVertex {
                p0: [x, 0f32],
                p1: [x, y_max],
                color,
            });
        }

        // Y
        for q in input.queries.iter() {
            // for (i, _) in queries_sort.into_iter().take(1) {
            // let q = &input.queries[i];
            let y = q.offset as f32;
            lines.push(LineVertex {
                p0: [0f32, y],
                p1: [x_max, y],
                color,
            });
        }

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&lines),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        (buffer, 0..instances as u32)
    };

    let mut projection = ultraviolet::projection::orthographic_wgpu_dx(
        0.0,
        input.target_len as f32 / 10.0,
        0.0,
        input.query_len as f32 / 10.0,
        0.1,
        10.0,
    );

    let (proj_uniform, conf_uniform) = {
        let proj_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[projection]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // px / window width
        let line_width: f32 = 10.0 / 1000.0;
        let conf = [line_width, 0.0, 0.0, 0.0];
        let conf_uniform = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&conf),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        (proj_uniform, conf_uniform)
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
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

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
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
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &bind_group, &[]);

                            // first draw grid
                            rpass.set_vertex_buffer(0, grid_buffer.slice(..));
                            rpass.draw(0..6, grid_instances.clone());
                            // rpass.draw(0..6, 0..2);
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
