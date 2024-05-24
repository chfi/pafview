use std::sync::Arc;

use anyhow::{anyhow, Result};
use bimap::BiMap;
use egui::Galley;
use rustc_hash::FxHashMap;
use ultraviolet::{DVec2, Vec2};

use crate::{
    grid::{AlignmentGrid, AxisRange},
    gui::AppWindowStates,
    sequences::SeqId,
    PafViewerApp,
};

use self::draw::AnnotShapeId;

pub mod draw;
pub mod label_layout;

pub mod physics;

// TODO newtype these
pub type RecordListId = usize;
pub type RecordEntryId = usize;
pub type AnnotationId = (RecordListId, RecordEntryId);

#[derive(Default)]
pub struct AnnotationStore {
    annotation_sources: BiMap<String, usize>,
    annotation_lists: Vec<RecordList>,
    // annotation_sources: FxHashMap<PathBuf, RecordList>,

    // same indices as annotation_list & RecordLists
    shapes: Vec<Vec<AnnotationShapes>>,
    // shapes: FxHashMap<(usize, usize), AnnotationShapes>,
}

#[derive(Debug, Clone, Copy)]
struct AnnotationShapes {
    pub target: AnnotShapeId,
    pub query: AnnotShapeId,
}

impl AnnotationStore {
    pub fn target_shape_for(&self, list_id: usize, record_id: usize) -> Option<AnnotShapeId> {
        self.shapes.get(list_id)?.get(record_id).map(|s| s.target)
    }
    pub fn query_shape_for(&self, list_id: usize, record_id: usize) -> Option<AnnotShapeId> {
        self.shapes.get(list_id)?.get(record_id).map(|s| s.query)
    }

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
        alignment_grid: &AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        painter: &mut draw::AnnotationPainter,
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
            let seq_id = *alignment_grid
                .sequence_names
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

            let mut color = if let Some(color) = color {
                color
            } else {
                let [r, g, b] = string_hash_color(&label);
                egui::Rgba::from_rgb(r, g, b).into()
            };

            color = color.linear_multiply(0.5);

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

        let record_list = RecordList {
            records: record_list,
        };
        let shapes = record_list.prepare_annotation_shapes(alignment_grid, painter);
        self.annotation_lists.push(record_list);
        self.shapes.push(shapes);

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
    pub seq_id: SeqId,
    pub seq_range: std::ops::Range<u64>,

    pub color: egui::Color32,
    pub label: String,
}

impl RecordList {
    fn prepare_annotation_shapes(
        &self,
        // app: &PafViewerApp,
        alignment_grid: &AlignmentGrid,
        painter: &mut draw::AnnotationPainter,
    ) -> Vec<AnnotationShapes> {
        let mut shapes = Vec::new();

        let x_axis = &alignment_grid.x_axis;
        let y_axis = &alignment_grid.y_axis;

        for (record_id, record) in self.records.iter().enumerate() {
            let axis_range = AxisRange::Seq {
                seq_id: record.seq_id,
                range: record.seq_range.clone(),
            };
            let world_x_range = x_axis.axis_range_into_global(&axis_range);
            let world_y_range = y_axis.axis_range_into_global(&axis_range);

            let color = record.color;

            let target_shape = draw::AnnotationWorldRegion {
                world_x_range: world_x_range.clone(),
                world_y_range: None,
                color,
            };
            let target_label = draw::AnnotationLabel {
                world_x_range: world_x_range.clone(),
                world_y_range: None,
                align: egui::Align2::CENTER_TOP,
                text: record.label.clone(),
            };
            let target =
                painter.add_collection([Box::new(target_shape) as _, Box::new(target_label) as _]);

            let query_shape = draw::AnnotationWorldRegion {
                world_x_range: None,
                world_y_range: world_y_range.clone(),
                color,
            };
            let query_label = draw::AnnotationLabel {
                world_x_range: None,
                world_y_range,
                align: egui::Align2::CENTER_TOP,
                text: record.label.clone(),
            };
            let query =
                painter.add_collection([Box::new(query_shape) as _, Box::new(query_label) as _]);

            shapes.push(AnnotationShapes { target, query });
        }

        shapes
    }
}

#[derive(Default)]
pub struct AnnotationGuiHandler {
    // target_enabled: Vec<bool>,
    // query_enabled: Vec<bool>,

    // draw_target_regions: FxHashMap<String,
    record_states: FxHashMap<(usize, usize), AnnotationState>,

    filter_text: String,
}

struct AnnotationState {
    draw_target_region: bool,
    draw_query_region: bool,

    galley: Option<Arc<Galley>>,
    seq_region: std::ops::RangeInclusive<f64>,
}

impl AnnotationGuiHandler {
    fn record_states_filtered_mut<'a: 'b, 'b>(
        &'a mut self,
        annotations: &'b AnnotationStore,
    ) -> impl Iterator<Item = &'a mut AnnotationState> + 'b {
        self.record_states
            .iter_mut()
            .filter_map(|(&(list_id, record_id), state)| {
                let list = annotations.annotation_lists.get(list_id)?;
                let record = list.records.get(record_id)?;
                if record.label.contains(&self.filter_text) {
                    Some(state)
                } else {
                    None
                }
            })
    }

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

                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Filter");
                        ui.text_edit_singleline(&mut self.filter_text);
                        if ui.button("Clear").clicked() {
                            self.filter_text.clear();
                        }
                    });

                    ui.separator();

                    for (_file_path, &list_id) in app.annotations.annotation_sources.iter() {
                        let list = &app.annotations.annotation_lists[list_id];

                        ui.vertical(|ui| {
                            ui.label("Toggle annotation display");
                            ui.horizontal(|ui| {
                                ui.label("All");
                                let target_btn = ui.button("Target");
                                let query_btn = ui.button("Query");

                                if target_btn.clicked() {
                                    self.record_states_filtered_mut(&app.annotations).for_each(
                                        |state| {
                                            state.draw_target_region = !state.draw_target_region
                                        },
                                    );
                                }

                                if query_btn.clicked() {
                                    self.record_states_filtered_mut(&app.annotations).for_each(
                                        |state| {
                                            state.draw_query_region = !state.draw_query_region;
                                        },
                                    );
                                }
                            });

                            ui.separator();

                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for (record_id, record) in list.records.iter().enumerate() {
                                    if !record.label.contains(&self.filter_text) {
                                        continue;
                                    }

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
                        });
                    }
                });
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

            let axis_len = app.alignment_grid.x_axis.total_len as f64;
            let extra = 0.03 * axis_len;

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
                let s1 = view.map_world_to_screen(screen_size, DVec2::new(x1, extra + axis_len));

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
                let s1 = view.map_world_to_screen(screen_size, DVec2::new(extra + axis_len, y1));

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

/*
pub fn find_matches_for_target_range(
    seq_names: &BiMap<String, SeqId>,
    input: &PafInput,
    // target_name: &str,
    target_id: SeqId,
    target_range: std::ops::Range<u64>,
) -> Vec<(usize, Vec<[DVec2; 2]>)> {
    // let Some(&target_id) = seq_names.get(target_name) else {
    //     return Vec::new();
    // };
    let target = &input.targets[target_id.0];
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
    seq_names: &BiMap<String, SeqId>,
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
*/
