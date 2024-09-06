use std::sync::Arc;

use anyhow::Result;
use bimap::BiMap;
use egui::Galley;

use crate::{
    grid::{AlignmentGrid, AxisRange},
    sequences::SeqId,
};

use self::draw::AnnotShapeId;

pub mod draw;
// pub mod label_layout;

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

    pub fn is_empty(&self) -> bool {
        self.annotation_lists.is_empty()
    }
}

impl AnnotationStore {
    pub fn load_bed_file(
        &mut self,
        sequence_names: &bimap::BiMap<String, SeqId>,
        bed_path: impl AsRef<std::path::Path>,
    ) -> Result<RecordListId> {
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

            let Some(seq_id) = fields.get(0).and_then(|&seq_name| {
                let seq_id = *sequence_names.get_by_left(seq_name)?;
                Some(seq_id)
            }) else {
                continue;
            };

            let parse_range = |from: usize, to: usize| -> Option<std::ops::Range<u64>> {
                let start = fields.get(from);
                let end = fields.get(to);
                start.zip(end).and_then(|(s, e)| {
                    let s = s.trim().parse().ok()?;
                    let e = e.trim().parse().ok()?;
                    Some(s..e)
                })
            };

            let Some(seq_range) = parse_range(1, 2) else {
                continue;
            };

            let label = fields.get(3).unwrap().to_string();

            let _rev_strand = fields.get(5).map(|&s| s == "-").unwrap_or(false);

            let _thick_range = parse_range(6, 7);

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
                qry_id: seq_id,
                qry_range: seq_range.clone(),
                tgt_id: seq_id,
                tgt_range: seq_range.clone(),
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
        // let shapes = record_list.prepare_annotation_shapes(alignment_grid, painter);
        let list_id = self.annotation_lists.len();
        self.annotation_lists.push(record_list);
        // self.shapes.push(shapes);

        Ok(list_id)
    }

    pub fn load_bedpe_file(
        &mut self,
        sequence_names: &bimap::BiMap<String, SeqId>,
        bedpe_path: impl AsRef<std::path::Path>,
    ) -> Result<RecordListId> {
        use std::io::prelude::*;
        use std::io::BufReader;

        let bed_reader = std::fs::File::open(bedpe_path.as_ref()).map(BufReader::new)?;

        let mut record_list: Vec<Record> = Vec::new();

        for line in bed_reader.lines() {
            let line = line?;
            let fields = line.trim().split('\t').collect::<Vec<_>>();

            if fields.len() < 6 {
                continue;
            }

            let get_id = |ix: usize| {
                fields
                    .get(ix)
                    .and_then(|&seq_name| sequence_names.get_by_left(seq_name).copied())
            };

            let Some((qry_id, tgt_id)) = get_id(0).zip(get_id(3)) else {
                continue;
            };

            let parse_range = |from: usize, to: usize| -> Option<std::ops::Range<u64>> {
                let start = fields.get(from);
                let end = fields.get(to);
                start.zip(end).and_then(|(s, e)| {
                    let s = s.trim().parse().ok()?;
                    let e = e.trim().parse().ok()?;
                    Some(s..e)
                })
            };

            let qry_range = parse_range(1, 2);
            let tgt_range = parse_range(4, 5);

            let Some((qry_range, tgt_range)) = qry_range.zip(tgt_range) else {
                continue;
            };

            let label = fields.get(6).map(|&s| s).unwrap_or_default().to_string();

            let mut color: egui::Color32 = {
                let [r, g, b] = string_hash_color(&label);
                egui::Rgba::from_rgb(r, g, b).into()
            };

            color = color.linear_multiply(0.5);

            record_list.push(Record {
                qry_id,
                qry_range,
                tgt_id,
                tgt_range,
                color,
                label,
            });
        }
        let source = bedpe_path.as_ref().to_owned();

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
        // let shapes = record_list.prepare_annotation_shapes(alignment_grid, painter);
        let list_id = self.annotation_lists.len();
        self.annotation_lists.push(record_list);
        // self.shapes.push(shapes);

        Ok(list_id)
    }
}

pub struct RecordList {
    pub records: Vec<Record>,
}

pub struct Record {
    pub qry_id: SeqId,
    pub qry_range: std::ops::Range<u64>,

    pub tgt_id: SeqId,
    pub tgt_range: std::ops::Range<u64>,

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
            let tgt_range = AxisRange::Seq {
                seq_id: record.tgt_id,
                range: record.tgt_range.clone(),
            };
            let qry_range = AxisRange::Seq {
                seq_id: record.qry_id,
                range: record.qry_range.clone(),
            };
            let world_x_range = x_axis.axis_range_into_global(&tgt_range);
            let world_y_range = y_axis.axis_range_into_global(&qry_range);

            let color = record.color;

            let target_shape = draw::AnnotationWorldRegion {
                world_x_range: world_x_range.clone(),
                world_y_range: None,
                color,
            };
            let target_label = draw::AnnotationLabel {
                // world_x_range: world_x_range.clone(),
                // world_y_range: None,
                // align: egui::Align2::CENTER_TOP,
                screen_pos: None,
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
                // world_x_range: None,
                // world_y_range,
                // align: egui::Align2::CENTER_TOP,
                screen_pos: None,
                text: record.label.clone(),
            };
            let query =
                painter.add_collection([Box::new(query_shape) as _, Box::new(query_label) as _]);

            shapes.push(AnnotationShapes { target, query });
        }

        shapes
    }
}

struct AnnotationState {
    draw_target_region: bool,
    draw_query_region: bool,

    galley: Option<Arc<Galley>>,
    seq_region: std::ops::RangeInclusive<f64>,
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
