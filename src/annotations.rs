use std::{
    borrow::Borrow,
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{anyhow, Result};
use bimap::BiMap;
use egui::Galley;
use rustc_hash::FxHashMap;
use ultraviolet::{DVec2, Vec2};

use crate::{gui::AppWindowStates, PafInput, PafViewerApp};

#[derive(Default)]
pub struct AnnotationStore {
    annotation_sources: BiMap<String, usize>,
    annotation_lists: Vec<RecordList>,
    // annotation_sources: FxHashMap<PathBuf, RecordList>,
}

impl AnnotationStore {
    pub fn source_names_iter<'a>(&'a self) -> impl Iterator<Item = (usize, &'a str)> {
        (0..self.annotation_lists.len()).filter_map(|i| {
            Some((
                i,
                self.annotation_sources
                    .get_by_right(&i)
                    .map(|s| s.as_str())?,
            ))
        })
    }

    pub fn list_by_id(&self, id: usize) -> Option<&RecordList> {
        self.annotation_lists.get(id)
    }

    pub fn list_by_name(&self, name: &str) -> Option<&RecordList> {
        let id = self.annotation_sources.get_by_left(name)?;
        self.annotation_lists.get(*id)
    }

    pub fn load_bed_file(
        &mut self,
        seq_names: &BiMap<String, usize>,
        bed_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        use std::io::prelude::*;
        use std::io::BufReader;

        let bed_reader = std::fs::File::open(bed_path.as_ref()).map(BufReader::new)?;

        let mut record_list: Vec<Record> = Vec::new();

        for line in bed_reader.lines() {
            let line = line?;
            let fields = line.trim().split('\t').collect::<Vec<_>>();

            if fields.len() < 4 {
                continue;
            }

            let seq_name = *fields
                .get(0)
                .ok_or(anyhow!("Sequence name missing from BED row"))?;
            let seq_id = *seq_names
                .get_by_left(seq_name)
                .ok_or(anyhow!("Could not find sequence `{seq_name}`"))?;

            let parse_range = |from: usize, to: usize| -> Option<std::ops::Range<u64>> {
                let start = fields.get(from);
                let end = fields.get(to);
                start.zip(end).and_then(|(s, e)| {
                    let s = s.parse().ok()?;
                    let e = e.parse().ok()?;
                    Some(s..e)
                })
            };

            let seq_range = parse_range(1, 2).ok_or_else(|| {
                anyhow!(
                    "Could not parse `{:?}`, `{:?}` as interval",
                    fields.get(1),
                    fields.get(2)
                )
            })?;

            let label = fields.get(3).unwrap().to_string();

            let rev_strand = fields.get(5).map(|&s| s == "-").unwrap_or(false);

            let thick_range = parse_range(6, 7);

            let color = fields.get(8).and_then(|rgb| {
                let mut rgb = rgb.split(',');

                let mut next_chan =
                    || -> Option<u8> { rgb.next().and_then(|s: &str| s.parse().ok()) };

                let r = next_chan()?;
                let g = next_chan()?;
                let b = next_chan()?;

                Some(egui::Color32::from_rgb(r, g, b))
            });

            let color = if let Some(color) = color {
                color
            } else {
                let [r, g, b] = string_hash_color(&label);
                egui::Rgba::from_rgb(r, g, b).into()
            };

            record_list.push(Record {
                seq_id,
                seq_range,
                color,
                label,
            });
        }
        let source = bed_path.as_ref().to_owned();

        let source_str = source
            .as_os_str()
            .to_str()
            .map(String::from)
            .unwrap_or_else(|| "<Error>".to_string());

        self.annotation_sources
            .insert(source_str, self.annotation_lists.len());
        self.annotation_lists.push(RecordList {
            records: record_list,
        });

        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.annotation_lists.is_empty()
    }
}

pub struct RecordList {
    pub records: Vec<Record>,
}

pub struct Record {
    pub seq_id: usize,
    pub seq_range: std::ops::Range<u64>,

    pub color: egui::Color32,
    pub label: String,
}

#[derive(Default)]
pub struct AnnotationGuiHandler {
    // target_enabled: Vec<bool>,
    // query_enabled: Vec<bool>,

    // draw_target_regions: FxHashMap<String,
    record_states: FxHashMap<(usize, usize), AnnotationState>,
}

struct AnnotationState {
    draw_target_region: bool,
    draw_query_region: bool,

    galley: Option<Arc<Galley>>,
    seq_region: std::ops::RangeInclusive<f64>,
}

impl AnnotationGuiHandler {
    pub fn show_annotation_list(
        &mut self,
        ctx: &egui::Context,
        app: &PafViewerApp,
        window_state: &mut AppWindowStates,
    ) {
        let Some(annot_list_open) = window_state.annotation_list_open.as_mut() else {
            return;
        };

        egui::Window::new("Annotations")
            .open(annot_list_open)
            .show(&ctx, |ui| {
                // TODO scrollable & filterable list of annotation records

                // if ui.button("Show annotations").clicked() {
                for (_file_path, &list_id) in app.annotations.annotation_sources.iter() {
                    let list = &app.annotations.annotation_lists[list_id];

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (record_id, record) in list.records.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!("{}", record.label));

                                let target_btn = ui.button("Target");
                                let query_btn = ui.button("Query");

                                let state = self.get_region_state(app, list_id, record_id);

                                if target_btn.clicked() {
                                    state.draw_target_region = !state.draw_target_region;
                                    log::info!(
                                        "drawing {} target region: {}\t(region {:?})",
                                        record.label,
                                        state.draw_target_region,
                                        state.seq_region
                                    );
                                }

                                if query_btn.clicked() {
                                    state.draw_query_region = !state.draw_query_region;
                                    log::info!(
                                        "drawing {} query region: {}\t(region {:?})",
                                        record.label,
                                        state.draw_query_region,
                                        state.seq_region
                                    );
                                }
                            });
                        }
                    });
                }
            });
    }

    pub fn draw_annotations(
        &mut self,
        ctx: &egui::Context,
        app: &PafViewerApp,
        view: &crate::view::View,
    ) {
        let screen_size = ctx.screen_rect().size();

        let id_str = "annotation-regions-painter";
        let id = egui::Id::new(id_str);

        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, id));

        for ((list_id, record_id), state) in self.record_states.iter_mut() {
            let record = &app.annotations.annotation_lists[*list_id].records[*record_id];

            let color = egui::Rgba::from(record.color).multiply(0.5);

            let extra = 0.03 * app.paf_input.target_len as f64;

            if state.galley.is_none() && (state.draw_target_region || state.draw_query_region) {
                let mut job = egui::text::LayoutJob::default();
                job.append(
                    &record.label,
                    0.0,
                    egui::text::TextFormat {
                        font_id: egui::FontId::monospace(12.0),
                        color: egui::Color32::PLACEHOLDER,
                        ..Default::default()
                    },
                );
                let galley = painter.layout_job(job);
                state.galley = Some(galley);
            }

            if state.draw_target_region {
                let x0 = *state.seq_region.start();
                let x1 = *state.seq_region.end();
                let s0 = view.map_world_to_screen(screen_size, DVec2::new(x0, -extra));
                let s1 = view.map_world_to_screen(
                    screen_size,
                    DVec2::new(x1, extra + app.paf_input.target_len as f64),
                );

                let y0 = s0.y.min(s1.y);
                let y1 = s0.y.max(s1.y);
                let mut rect = egui::Rect::from_x_y_ranges(s0.x..=s1.x, y0..=y1);

                if rect.width() < 0.5 {
                    rect.set_width(1.0);
                }

                painter.rect_filled(rect, 0.0, color);

                let mut label_pos = rect.left_top();
                label_pos.y = label_pos.y.max(25.0);

                if let Some(galley) = state.galley.as_ref() {
                    painter.galley(label_pos, galley.clone(), egui::Color32::BLACK);
                }
            }

            if state.draw_query_region {
                let y0 = *state.seq_region.start();
                let y1 = *state.seq_region.end();
                let s0 = view.map_world_to_screen(screen_size, DVec2::new(-extra, y0));
                let s1 = view.map_world_to_screen(
                    screen_size,
                    DVec2::new(extra + app.paf_input.target_len as f64, y1),
                );

                let y0 = s0.y.min(s1.y);
                let y1 = s0.y.max(s1.y);
                let mut rect = egui::Rect::from_x_y_ranges(s0.x..=s1.x, y0..=y1);

                if rect.height() < 0.5 {
                    rect.set_height(1.0);
                }

                painter.rect_filled(rect, 0.0, color);

                let mut label_pos = rect.left_top();
                label_pos.x = label_pos.x.max(5.0);

                if let Some(galley) = state.galley.as_ref() {
                    painter.galley(label_pos, galley.clone(), egui::Color32::BLACK);
                }
            }
        }
    }

    fn get_region_state(
        &mut self,
        app: &PafViewerApp,
        annotation_list_id: usize,
        record_id: usize,
    ) -> &mut AnnotationState {
        let key = (annotation_list_id, record_id);

        let axis = &app.alignment_grid.x_axis;

        if !self.record_states.contains_key(&key) {
            let record = &app.annotations.annotation_lists[annotation_list_id].records[record_id];

            let seq_offset = axis.sequence_offset(record.seq_id).unwrap();

            let start_w = (seq_offset + record.seq_range.start) as f64;
            let end_w = (seq_offset + record.seq_range.end) as f64;

            self.record_states.insert(
                key,
                AnnotationState {
                    draw_target_region: false,
                    draw_query_region: false,
                    galley: None,
                    seq_region: start_w..=end_w,
                },
            );
        }

        let Some(state) = self.record_states.get_mut(&key) else {
            unreachable!();
        };

        state
    }

    // fn enable_annotation_region(
    //     &mut self,
    //     app: &PafViewerApp,
    //     annotation_list_id: usize,
    //     record_id: usize,
    //     draw_target_region: Option<bool>,
    //     draw_query_region: Option<bool>,
    // ) {
    //     let state = self.get_region_state(app, annotation_list_id, record_id);

    //     state.draw_target_region = draw_target_region.unwrap_or(state.draw_target_region);
    //     state.draw_query_region = draw_query_region.unwrap_or(state.draw_query_region);
    // }
}

pub fn hashed_rgb(name: &str) -> [u8; 3] {
    // use sha2::Digest;
    // use sha2::Sha256;

    // let mut hasher = Sha256::new();
    // hasher.update(name.as_bytes());
    // let hash = hasher.finalize();
    // let r = hash[24];
    // let g = hash[8];
    // let b = hash[16];

    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::default();
    name.as_bytes().hash(&mut hasher);
    let hash = hasher.finish().to_ne_bytes();

    let r = hash[0];
    let g = hash[1];
    let b = hash[2];

    [r, g, b]
}

pub fn string_hash_color_f32(input: &str) -> [f32; 3] {
    let [s_r, s_g, s_b] = hashed_rgb(input);

    let r_f = (s_r as f32) / std::u8::MAX as f32;
    let g_f = (s_g as f32) / std::u8::MAX as f32;
    let b_f = (s_b as f32) / std::u8::MAX as f32;

    let sum = r_f + g_f + b_f;

    [r_f / sum, g_f / sum, b_f / sum]
}

pub fn string_hash_color_alt(path_name: &str) -> [f32; 3] {
    string_hash_color_f32(path_name)
}

pub fn string_hash_color(path_name: &str) -> [f32; 3] {
    let [path_r, path_g, path_b] = hashed_rgb(path_name);

    let r_f = (path_r as f32) / std::u8::MAX as f32;
    let g_f = (path_g as f32) / std::u8::MAX as f32;
    let b_f = (path_b as f32) / std::u8::MAX as f32;

    let sum = r_f + g_f + b_f;

    let r_f = r_f / sum;
    let g_f = g_f / sum;
    let b_f = b_f / sum;

    let f = (1.0 / r_f.max(g_f).max(b_f)).min(1.5);

    let r_u = (255. * (r_f * f).min(1.0)).round();
    let g_u = (255. * (g_f * f).min(1.0)).round();
    let b_u = (255. * (b_f * f).min(1.0)).round();

    let r_f = (r_u as f32) / std::u8::MAX as f32;
    let g_f = (g_u as f32) / std::u8::MAX as f32;
    let b_f = (b_u as f32) / std::u8::MAX as f32;

    [r_f, g_f, b_f]
}

pub fn find_matches_for_target_range(
    seq_names: &BiMap<String, usize>,
    input: &PafInput,
    // target_name: &str,
    target_id: usize,
    target_range: std::ops::Range<u64>,
) -> Vec<(usize, Vec<[DVec2; 2]>)> {
    // let Some(&target_id) = seq_names.get(target_name) else {
    //     return Vec::new();
    // };
    let target = &input.targets[target_id];
    let global_tgt_offset = target.offset as u64 + target_range.start;

    let mut output = Vec::new();

    for (cid, cigar) in input.processed_lines.iter().enumerate() {
        if cigar.target_id != target_id {
            continue;
        }

        let start_ix = cigar
            .match_offsets
            .partition_point(|[tgt_ofs, _]| *tgt_ofs < target_range.start + global_tgt_offset);

        if start_ix == cigar.match_offsets.len() {
            continue;
        }

        let end_ix = cigar.match_offsets[start_ix..]
            .partition_point(|[tgt_ofs, _]| *tgt_ofs < target_range.end + global_tgt_offset);

        let match_range = start_ix..(start_ix + end_ix);

        let mut segments = Vec::new();

        for (match_i, &[start, end]) in cigar.match_edges[match_range].iter().enumerate() {
            // TODO only include the part of the line that's actually covered by the range?
            segments.push([start, end]);
        }

        if !segments.is_empty() {
            output.push((cid, segments));
        }
    }

    output
}

#[derive(Default, Clone)]
struct AnnotTestState {
    key: Option<(String, [u64; 2])>,
    match_output: Arc<egui::mutex::Mutex<Vec<(usize, Vec<[DVec2; 2]>)>>>,
}

pub fn draw_annotation_test_window(
    seq_names: &BiMap<String, usize>,
    input: &PafInput,
    ctx: &egui::Context,
    view: &crate::view::View,
) {
    let id_str = "annotation-test-window";
    let id = egui::Id::new(id_str);
    let (mut range_text, mut label_text, mut annot_state) = ctx.data(|d| {
        let (range_text, label_text) = d.get_temp::<(String, String)>(id).unwrap_or_default();
        let state = d.get_temp::<AnnotTestState>(id).unwrap_or_default();
        (range_text, label_text, state)
    });

    egui::Window::new("Annotation Test").show(&ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Range");
            ui.text_edit_singleline(&mut range_text);
        });
        ui.horizontal(|ui| {
            ui.label("Label");
            ui.text_edit_singleline(&mut label_text);
        });
    });

    let parse_range = |text: &str| {
        let mut splits = text.split(':');
        let name = splits.next()?;
        let mut range_comp = splits.next()?.split('-');
        let start = range_comp.next().and_then(|s| s.parse::<u64>().ok())?;
        let end = range_comp.next().and_then(|s| s.parse::<u64>().ok())?;
        Some((name.to_string(), [start, end]))
    };

    if let key @ Some((name, [start, end])) = parse_range(&range_text).as_ref() {
        if key != annot_state.key.as_ref() {
            let new_outputs = if let Some(seq_id) = seq_names.get_by_left(name) {
                find_matches_for_target_range(seq_names, input, *seq_id, *start..*end)
            } else {
                vec![]
            };
            annot_state.key = key.cloned();
            let mut outputs = annot_state.match_output.lock();
            *outputs = new_outputs;
        }
    } else {
        annot_state.key = None;
        let mut outputs = annot_state.match_output.lock();
        outputs.clear();
    }

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, id));

    let dims = ctx.screen_rect().size();

    let stroke = egui::Stroke {
        width: 4.0,
        color: egui::Color32::RED,
    };

    {
        let matches = annot_state.match_output.lock();
        let mut aabb_min = Vec2::broadcast(std::f32::MAX);
        let mut aabb_max = Vec2::broadcast(std::f32::MIN);
        for (cigar_i, matches) in matches.iter() {
            for (match_i, &[from, to]) in matches.iter().enumerate() {
                let s0 = view.map_world_to_screen(dims, from);
                let s1 = view.map_world_to_screen(dims, to);

                aabb_min = aabb_min.min_by_component(s0).min_by_component(s1);
                aabb_max = aabb_max.max_by_component(s0).max_by_component(s1);

                let p0: [f32; 2] = s0.into();
                let p1: [f32; 2] = s1.into();

                painter.line_segment([p0.into(), p1.into()], stroke);

                if match_i == matches.len() / 2 {
                    let d = s1 - s0;
                    let angle = d.y.atan2(d.x) as f32;

                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        &label_text,
                        0.0,
                        egui::text::TextFormat {
                            font_id: egui::FontId::monospace(12.0),
                            color: egui::Color32::PLACEHOLDER,
                            ..Default::default()
                        },
                    );

                    let galley = painter.layout_job(job);

                    let mut text_shape =
                        egui::epaint::TextShape::new(p0.into(), galley, egui::Color32::BLACK);
                    text_shape.angle = angle;

                    painter.add(text_shape);
                }
            }
        }

        // let rect = egui::Rect::from_x_y_ranges(aabb_min.x..=aabb_max.x, aabb_min.y..=aabb_max.y);
        // let color = egui::Rgba::from_rgba_unmultiplied(1.0, 0.0, 0.0, 0.5);
        // painter.rect_filled(rect, 0.0, color);
    }

    ctx.data_mut(|d| {
        d.insert_temp(id, (range_text, label_text));
        d.insert_temp(id, annot_state);
    });
}

#[derive(Default)]
pub struct AnnotationPainter {
    galley_cache: FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
    annotations: FxHashMap<usize, Box<dyn DrawAnnotation>>,
}

pub trait DrawAnnotation {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    );
}

pub struct AnnotationDrawCollection {
    draw: Vec<Box<dyn DrawAnnotation>>,
}

impl DrawAnnotation for AnnotationDrawCollection {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        for item in self.draw.iter() {
            item.draw(galley_cache, painter, view, screen_size);
        }
    }
}

pub struct AnnotationLabel {
    pub world_x_range: Option<std::ops::RangeInclusive<f64>>,
    pub world_y_range: Option<std::ops::RangeInclusive<f64>>,
    pub align: egui::Align,

    pub text: String,
    // can't use TextFormat as key bc not Eq; hash manually and key w/ u64, later
    // pub format: egui::TextFormat,
}

impl DrawAnnotation for AnnotationLabel {
    fn draw(
        &self,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        if !galley_cache.contains_key(&self.text) {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                &self.text,
                0.0,
                egui::text::TextFormat {
                    font_id: egui::FontId::monospace(12.0),
                    color: egui::Color32::PLACEHOLDER,
                    ..Default::default()
                },
            );
            let galley = painter.layout_job(job);
            galley_cache.insert(self.text.clone(), galley);
        }

        let Some(galley) = galley_cache.get(&self.text).cloned() else {
            unreachable!();
        };

        let view_min = DVec2::new(view.x_min, view.y_min);
        let view_max = DVec2::new(view.x_max, view.y_max);

        let [p0, p1] = match (&self.world_x_range, &self.world_y_range) {
            (Some(xs), Some(ys)) => {
                // draw in top left of screen rect, for now
                let p0 = DVec2::new(*xs.start(), *ys.start());
                let p1 = DVec2::new(*xs.end(), *ys.end());
                [p0, p1]
            }
            (Some(xs), None) => {
                // draw at top of screen of vertical region
                let p0 = DVec2::new(*xs.start(), view_min.y);
                let p1 = DVec2::new(*xs.end(), view_max.y);
                [p0, p1]
            }
            (None, Some(ys)) => {
                // draw at left of screen of horizontal region
                let p0 = DVec2::new(view_min.x, *ys.start());
                let p1 = DVec2::new(view_min.x, *ys.end());
                [p0, p1]
            }
            _ => {
                return;
            }
        };

        let min = p0.min_by_component(view_min);
        let max = p1.max_by_component(view_max);
        // let min = p0.min_by_component(DVec2::new(view.x_min, view.y_min));
        // let max = p1.max_by_component(DVec2::new(view.x_max, view.y_max));
        // let x_min = p0.x.min(view.x_min);
        // let y_min = p0.y.min(view.y_min);
        // let x_max = p1.x.max(view.x_max);
        // let y_max = p1.y.max(view.y_max);

        let q0: [f32; 2] = view.map_world_to_screen(screen_size, min).into();
        let q1: [f32; 2] = view.map_world_to_screen(screen_size, max).into();

        let rect = egui::Rect::from_two_pos(q0.into(), q1.into());
        painter.galley(rect.left_top(), galley, egui::Color32::BLACK);
    }
}

pub struct AnnotationWorldRegion {
    pub world_x_range: Option<std::ops::RangeInclusive<f64>>,
    pub world_y_range: Option<std::ops::RangeInclusive<f64>>,
    pub color: egui::Color32,
}

impl DrawAnnotation for AnnotationWorldRegion {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        _galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        let view_min = DVec2::new(view.x_min, view.y_min);
        let view_max = DVec2::new(view.x_max, view.y_max);

        let [p0, p1] = match (&self.world_x_range, &self.world_y_range) {
            (Some(xs), Some(ys)) => {
                let p0 = DVec2::new(*xs.start(), *ys.start());
                let p1 = DVec2::new(*xs.end(), *ys.end());
                [p0, p1]
            }
            (Some(xs), None) => {
                let p0 = DVec2::new(*xs.start(), view_min.y);
                let p1 = DVec2::new(*xs.end(), view_max.y);
                [p0, p1]
            }
            (None, Some(ys)) => {
                let p0 = DVec2::new(view_min.x, *ys.start());
                let p1 = DVec2::new(view_min.x, *ys.end());
                [p0, p1]
            }
            _ => {
                return;
            }
        };

        let q0: [f32; 2] = view.map_world_to_screen(screen_size, p0).into();
        let q1: [f32; 2] = view.map_world_to_screen(screen_size, p1).into();

        let rect = egui::Rect::from_two_pos(q0.into(), q1.into());
        painter.rect_filled(rect, 0.0, self.color);
        // painter.galley(rect.left_top(), galley, egui::Color32::BLACK);
    }
}
