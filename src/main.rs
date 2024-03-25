use annotations::AnnotationGuiHandler;
use bimap::BiMap;
use bytemuck::{Pod, Zeroable};
use egui_wgpu::ScreenDescriptor;
use grid::AlignmentGrid;
use regions::SelectionHandler;
use rustc_hash::FxHashMap;
use std::borrow::Cow;
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

mod annotations;
mod grid;
mod gui;
mod regions;
mod render;
mod spatial;
mod view;

use render::*;
use view::View;

use crate::{annotations::AnnotationStore, gui::AppWindowStates};

struct PafInput {
    queries: Vec<AlignedSeq>,
    targets: Vec<AlignedSeq>,
    target_len: u64,
    query_len: u64,

    // match_edges: Vec<[DVec2; 2]>,
    processed_lines: Vec<ProcessedCigar>,
}

impl PafInput {
    fn total_matches(&self) -> usize {
        self.processed_lines
            .iter()
            .map(|l| l.match_edges.len())
            .sum()
    }
}

struct ProcessedCigar {
    target_id: usize,
    target_offset: u64,
    target_len: u64,

    query_id: usize,
    query_offset: u64,
    query_len: u64,

    match_edges: Vec<[DVec2; 2]>,
    match_is_match: Vec<bool>,
    match_offsets: Vec<[u64; 2]>,

    aabb_min: DVec2,
    aabb_max: DVec2,
}

impl ProcessedCigar {
    fn from_line(
        seq_names: &BiMap<String, usize>,
        paf_line: &PafLine<&str>,
        origin: [u64; 2],
    ) -> anyhow::Result<Self> {
        let ops = paf_line
            .cigar
            .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
            .filter_map(|opstr| {
                let count = opstr[..opstr.len() - 1].parse::<u64>().ok()?;
                let op = opstr.as_bytes()[opstr.len() - 1] as char;
                Some((op, count))
            });

        let [mut target_pos, mut query_pos] = origin;

        let target_id = *seq_names
            .get_by_left(paf_line.tgt_name)
            .ok_or_else(|| anyhow!("Target sequence `{}` not found", paf_line.tgt_name))?;
        let query_id = *seq_names
            .get_by_left(paf_line.query_name)
            .ok_or_else(|| anyhow!("Query sequence `{}` not found", paf_line.query_name))?;

        let mut match_edges = Vec::new();
        let mut match_offsets = Vec::new();
        let mut match_is_match = Vec::new();

        let mut aabb_min = DVec2::broadcast(std::f64::MAX);
        let mut aabb_max = DVec2::broadcast(std::f64::MIN);

        for (op, count) in ops {
            match op {
                'M' | '=' | 'X' => {
                    let x = target_pos;
                    let y = query_pos;

                    {
                        let x0 = x as f64;
                        let y0 = y as f64;

                        let x_end = if paf_line.strand_rev {
                            x.checked_sub(count).unwrap_or_default()
                        } else {
                            x + count
                        };
                        let x1 = x_end as f64;
                        let y1 = (y + count) as f64;

                        let p0 = DVec2::new(x0, y0);
                        let p1 = DVec2::new(x1, y1);

                        aabb_min = aabb_min.min_by_component(p0).min_by_component(p1);
                        aabb_max = aabb_max.max_by_component(p0).max_by_component(p1);

                        match_edges.push([p0, p1]);
                        match_offsets.push([target_pos, query_pos]);

                        match_is_match.push(op == 'M' || op == '=');
                    }

                    target_pos += count;
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

        let target_len = target_pos - origin[0];
        let query_len = query_pos - origin[1];

        Ok(Self {
            target_id,
            target_offset: origin[0],
            target_len,

            query_id,
            query_offset: origin[1],
            query_len,

            match_edges,
            match_offsets,
            match_is_match,

            aabb_min,
            aabb_max,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PafLine<S> {
    query_name: S,
    query_seq_len: u64,
    query_seq_start: u64,
    query_seq_end: u64,

    tgt_name: S,
    tgt_seq_len: u64,
    tgt_seq_start: u64,
    tgt_seq_end: u64,

    strand_rev: bool,
    cigar: S,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct AlignedSeq {
    // name of the given sequence
    name: String,
    // its length
    len: u64,
    // its start offset in the global all-to-all alignment matrix
    #[deprecated]
    offset: u64,
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
) -> Option<(&'a str, u64, u64, u64)> {
    let name = fields.next()?;
    let len = fields.next()?.parse().ok()?;
    let start = fields.next()?.parse().ok()?;
    let end = fields.next()?.parse().ok()?;
    Some((name, len, start, end))
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

    let seq_names = target_names.into_iter().collect::<bimap::BiMap<_, _>>();

    let x_axis = grid::GridAxis::from_sequences(&seq_names, &targets);
    let y_axis = x_axis.clone();

    // process matches
    let mut processed_lines = Vec::new();

    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

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

        let origin = {
            let x0 = x_axis.sequence_offset(*target_i).unwrap();
            let y0 = y_axis.sequence_offset(*query_i).unwrap();
            // let x0 = &targets[*target_i].offset;
            // let y0 = &queries[*query_i].offset;

            let x = x0 + paf_line.tgt_seq_start;
            let y = if paf_line.strand_rev {
                y0 + paf_line.query_seq_end
            } else {
                y0 + paf_line.query_seq_start
            };
            [x as u64, y as u64]
        };

        processed_lines.push(ProcessedCigar::from_line(&seq_names, &paf_line, origin)?);

        // process_cigar(&paf_line, origin, &mut match_edges)?;
        // process_cigar_compress(&paf_line, origin, target_len, query_len, &mut match_edges)?;
    }

    let paf_input = PafInput {
        queries,
        targets,
        target_len,
        query_len,
        processed_lines,
    };

    let mut annotations = AnnotationStore::default();

    if let Some(bed_path) = bed_path {
        log::info!("Loading BED file `{bed_path}`");
        match annotations.load_bed_file(&seq_names, &bed_path) {
            Ok(_) => {
                log::info!("Loaded BED file `{bed_path}`");
            }
            Err(err) => log::error!("Error loading BED file at path `{bed_path}`: {err:?}"),
        }
    }

    println!("sum target len: {target_len}");
    println!("sum query len: {query_len}");
    let total_matches: usize = paf_input
        .processed_lines
        .iter()
        .map(|l| l.match_edges.len())
        .sum();
    println!("drawing {} matches", total_matches);

    let alignment_grid = AlignmentGrid { x_axis, y_axis };

    let app = PafViewerApp {
        alignment_grid,
        paf_input,
        seq_names,
        annotations,
    };

    start_window(app);

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
                    let x0 = x as f64;
                    let y0 = y as f64;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f64;
                    let y1 = (y + count) as f64;

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
                    let x0 = x as f64;
                    let y0 = y as f64;

                    let x_end = if paf.strand_rev {
                        x.checked_sub(count).unwrap_or_default()
                    } else {
                        x + count
                    };
                    let x1 = x_end as f64;
                    let y1 = (y + count) as f64;

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

#[derive(Clone, PartialEq, PartialOrd)]
pub enum AppEvent {
    LoadAnnotationFile { path: std::path::PathBuf },
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
        let input = &app.paf_input;
        let instances = input.targets.len() + input.queries.len() + 2;
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

        let x_axis = &app.alignment_grid.x_axis;
        let y_axis = &app.alignment_grid.y_axis;

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

    let mut app_view = View {
        x_min: 0.0,
        x_max: app.paf_input.target_len as f64,
        y_min: 0.0,
        y_max: app.paf_input.query_len as f64,
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
        match_color_buffer,
        match_instances,
    );

    paf_renderer.set_grid(Some((grid_buffer, grid_color_buffer, grid_instances)));

    let mut window_states = AppWindowStates::new(&app.annotations);

    let mut egui_renderer = EguiRenderer::new(&device, &config, swapchain_format, None, 1, &window);

    let mut annot_gui_handler = AnnotationGuiHandler::default();

    let mut roi_gui = gui::regions::RegionsOfInterestGui::default();

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

            if let Event::UserEvent(event) = &event {
                match event {
                    AppEvent::LoadAnnotationFile { path } => {
                        // TODO check file extension maybe, other logic
                        match app.annotations.load_bed_file(&app.seq_names, &path) {
                            Ok(_) => {
                                log::info!("Loaded BED file `{path:?}`");
                            }
                            Err(err) => {
                                log::error!("Error loading BED file at path `{path:?}`: {err:?}")
                            }
                        }
                    }
                }
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

                            // if event.state
                            let [w, h]: [u32; 2] = window.inner_size().into();
                            let path = "screenshot.png";
                            log::info!("taking screenshot");
                            match write_png(&app.paf_input, &app_view, w as usize, h as usize, path)
                            {
                                Ok(_) => log::info!("wrote screenshot to {path}"),
                                Err(e) => log::info!("error writing screenshot: {e:?}"),
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
                                // regions::paf_line_debug_aabbs(&input, ctx, &app_view);
                                // annotations::draw_annotation_test_window(
                                //     &app.seq_names,
                                //     &app.paf_input,
                                //     ctx,
                                //     &app_view,
                                // );

                                gui::MenuBar::show(ctx, &app, &mut window_states);

                                roi_gui.show_window(
                                    ctx,
                                    &mut app.annotations,
                                    &app.alignment_grid,
                                    &app.seq_names,
                                    &app.paf_input,
                                    &mut app_view,
                                    &mut window_states,
                                );

                                gui::view_controls(
                                    ctx,
                                    &app.alignment_grid,
                                    &app.seq_names,
                                    &app.paf_input,
                                    &mut app_view,
                                    &mut window_states,
                                );

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
    alignment_grid: AlignmentGrid,

    paf_input: PafInput,

    seq_names: bimap::BiMap<String, usize>,

    annotations: AnnotationStore,
}
