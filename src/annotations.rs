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

use crate::{PafInput, PafViewerApp};

#[derive(Default)]
pub struct AnnotationStore {
    annotation_sources: FxHashMap<PathBuf, usize>,
    annotation_lists: Vec<RecordList>,
    // annotation_sources: FxHashMap<PathBuf, RecordList>,
}

impl AnnotationStore {
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

            let color = fields.get(8).map(|rgb| {
                let rgb = rgb.split(',').collect::<Vec<_>>();
                let chan = |i: usize| {
                    rgb.get(i)
                        .and_then(|s: &&str| s.parse().ok())
                        .unwrap_or(0usize) as u8
                };
                let r = chan(0);
                let g = chan(1);
                let b = chan(2);
                egui::Color32::from_rgb(r, g, b)
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

        self.annotation_sources
            .insert(source, self.annotation_lists.len());
        self.annotation_lists.push(RecordList {
            records: record_list,
        });

        Ok(())
    }
}

pub struct RecordList {
    records: Vec<Record>,
}

pub struct Record {
    seq_id: usize,
    seq_range: std::ops::Range<u64>,

    color: egui::Color32,
    label: String,
}

#[derive(Default)]
pub struct AnnotationGuiHandler {
    prepared_records: BTreeMap<(usize, usize), (String, Vec<Vec<[DVec2; 2]>>)>,

    prepared_galleys: FxHashMap<String, Arc<Galley>>,
}

struct SeqPairAnnotations {
    target_id: usize,
    query_id: usize,

    anchor_min: DVec2,
    anchor_max: DVec2,
    // in order from start to end on the line between the anchors
    labels: Vec<(String, (f64, f64))>,
}

struct LayoutConfig {
    max_rows: usize,
}

impl std::default::Default for LayoutConfig {
    fn default() -> Self {
        Self { max_rows: 3 }
    }
}

fn draw_annotation_labels(
    ctx: &egui::Context,
    annots: &SeqPairAnnotations,
    galleys: &mut FxHashMap<String, Arc<Galley>>,
    cfg: LayoutConfig,
) {
    let id_str = "annotation-painter";
    let id = egui::Id::new(id_str);

    let screen_size = ctx.screen_rect().size();

    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, id));

    // let mut remaining = VecDeque::new();
    let mut remaining = Vec::new();

    for (ix, (text, (from, to))) in annots.labels.iter().enumerate() {
        if !galleys.contains_key(text) {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                text,
                0.0,
                egui::text::TextFormat {
                    font_id: egui::FontId::monospace(12.0),
                    color: egui::Color32::PLACEHOLDER,
                    ..Default::default()
                },
            );
            galleys.insert(text.to_string(), painter.layout_job(job));
        }

        let Some(galley) = galleys.get(text) else {
            unreachable!();
        };

        let range = *from..=*to;

        remaining.push((ix, range, galley.clone()));
    }

    // remaining.sort_by_key(|(_, range, _)| *range.start());
    remaining.sort_by(|(_, r0, _), (_, r1, _)| r0.start().partial_cmp(r1.start()).unwrap());

    let mut remaining = remaining.into_iter().collect::<VecDeque<_>>();

    // labels to draw; index into `annots.labels` plus offset along
    // the line `annots.anchor_min` to `annots.anchor_max`, in screenspace
    let mut current_row: Vec<(usize, f32)> = Vec::new();

    let mut overflow: Vec<(usize, f32)> = Vec::new();

    let mut last_added: Option<std::ops::RangeInclusive<f32>> = None;

    loop {
        let Some((ix, anchor_range, galley)) = remaining.pop_front() else {
            break;
        };

        // let next_anchor =

        // match last_added.take() {
        //     Some(last_range) => {
        //     }
        //     None => {
        //         //
        //     }
        // };

        // if let Some(last_range) = last_added {
        //     //
        // } else {
        //     //
        // }
    }
}

impl SeqPairAnnotations {
    fn blocks_from_processed_line(
        // fn from_processed_line(
        seq_names: &BiMap<String, usize>,
        input: &PafInput,
        records: &[Record],
    ) -> Option<Vec<Self>> {
        // go through the records, partitioning them by sequence (target) name

        // (target_id, list of relevant records)
        let mut columns: FxHashMap<usize, Vec<&Record>> = FxHashMap::default();

        for (record_id, record) in records.iter().enumerate() {
            // let Some(seq_name) = seq_names.get_by_right(&record.seq_id) else {
            //     continue;
            // };

            let column_records = columns.entry(record.seq_id).or_default();
            column_records.push(record);
        }

        for (_tgt_id, records) in columns.iter_mut() {
            records.sort_by_key(|&r| (r.seq_range.start, r.seq_range.end));
        }

        // for

        // let target_id = line.target_id;
        // let query_id = line.query_id;

        // let [start, _] = *line.match_edges.get(0)?;
        // let [_, end] = *line.match_edges.get(line.match_edges.len() - 1)?;

        // this might all be the wrong way around...

        // for record in records {
        //     //
        // }

        todo!();
    }
}

impl AnnotationGuiHandler {
    pub fn show_annotation_list(&mut self, ctx: &egui::Context, app: &PafViewerApp) {
        egui::Window::new("Annotations").show(&ctx, |ui| {
            // TODO scrollable & filterable list of annotation records

            if ui.button("Show annotations").clicked() {
                if self.prepared_records.is_empty() {
                    for (file_path, &list_id) in app.annotations.annotation_sources.iter() {
                        let list = &app.annotations.annotation_lists[list_id];
                        log::info!("Processing {file_path:?}");

                        for (record_id, record) in list.records.iter().enumerate() {
                            // let seq_id = app.seq_names.get_by_right(&record.seq_id);

                            let matches = find_matches_for_target_range(
                                &app.seq_names,
                                &app.paf_input,
                                record.seq_id,
                                record.seq_range.clone(),
                            )
                            .into_iter()
                            .map(|(_, v)| v)
                            .collect::<Vec<_>>();

                            let key = (list_id, record_id);
                            self.prepared_records
                                .insert(key, (record.label.clone(), matches));
                        }
                    }
                }
            }
        });
    }

    pub fn draw_annotations(
        &mut self,
        ctx: &egui::Context,
        app: &PafViewerApp,
        view: &crate::view::View,
    ) {
        let id_str = "annotation-painter";
        let id = egui::Id::new(id_str);

        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, id));

        let dims = ctx.screen_rect().size();

        for ((list_id, record_id), (label, match_sets)) in self.prepared_records.iter() {
            // TODO

            if !self.prepared_galleys.contains_key(label) {
                // TODO this technically has to be redone if points-per-pixel changes
                let mut job = egui::text::LayoutJob::default();
                job.append(
                    label,
                    0.0,
                    egui::text::TextFormat {
                        font_id: egui::FontId::monospace(12.0),
                        color: egui::Color32::PLACEHOLDER,
                        ..Default::default()
                    },
                );

                let galley = painter.layout_job(job);
                self.prepared_galleys.insert(label.to_string(), galley);
            }

            let Some(galley) = self.prepared_galleys.get(label) else {
                unreachable!();
            };

            let color = app.annotations.annotation_lists[*list_id].records[*record_id].color;

            let stroke = egui::Stroke { width: 4.0, color };

            for matches in match_sets {
                if matches.is_empty() {
                    continue;
                }

                for &[from, to] in matches {
                    let s0 = view.map_world_to_screen(dims, from);
                    let s1 = view.map_world_to_screen(dims, to);

                    let d = s1 - s0;
                    let angle = d.y.atan2(d.x) as f32;

                    let p0: [f32; 2] = s0.into();
                    let p1: [f32; 2] = s1.into();
                    painter.line_segment([p0.into(), p1.into()], stroke);
                }

                let [start, _] = matches[0];
                let [_, end] = matches[matches.len() - 1];

                let s0 = view.map_world_to_screen(dims, start);
                let s1 = view.map_world_to_screen(dims, end);

                let d = s1 - s0;
                let angle = d.y.atan2(d.x) as f32;

                let p: [f32; 2] = (s0 + (s1 - s0) * 0.5).into();

                let mut text_shape =
                    egui::epaint::TextShape::new(p.into(), galley.clone(), egui::Color32::BLACK);
                text_shape.angle = angle;

                painter.add(text_shape);
            }
        }
    }
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
