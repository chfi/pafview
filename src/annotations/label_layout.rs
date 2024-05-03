use std::sync::Arc;

use egui::Galley;
use na::Isometry2;
use rapier2d::{
    dynamics::RigidBodySet,
    geometry::{Collider, ColliderBuilder, ColliderSet, HalfSpace, RayCast, SharedShape},
    na::{self, vector},
    parry::{query::Ray, shape::Cuboid},
    pipeline::QueryPipeline,
};
use rustc_hash::FxHashMap;
use ultraviolet::DVec2;

use crate::{sequences::SeqId, PafViewerApp};

// TODO generalize for query annotations that "fall right" too
pub struct LabelDef<'a> {
    pub text: &'a str,
    pub galley: Arc<Galley>,
    pub world_x_region: std::ops::RangeInclusive<f64>,
}

/*

this is assuming annotation labels along the target (i.e. X axis),
with horizontal (i.e. normal text) labels, and that we're zoomed
in on a single alignment tile, but it should generalize

each label (String) is associated with a range on the target (Range<u64>)

given a collection of such labels, the goal is to place each piece of text
on the screen as close to the projection of its corresponding range onto
the alignment in the tile, while maximizing (and not giving up any) legibility

depending on the view scale, and the number and nature of the labels, it
may be impossible to render all the labels without overlap, in which case
it's acceptable to collapse a neighborhood of labels into a single object
(e.g. an icon, showing the entire list of labels on hover)


use a coarse linear approximation of the "height" of the alignment, using
uniform bins for speed, to compute collisions/positions of labels relative
to the visualized alignment



one approach, to part of the solution...

 * track "anchor" in world space...
 * labels live in screen space, but are influenced by the screen space
     projection of their world space anchor
 * when panning or zooming, the label will initially not be moved,
     only being affected by the anchor-related dynamics due to the
     anchor being moved to a different screen position
 * (maybe) anchors exert repulsive forces on one another when close
 * labels will try to sit above their anchor, moving vertically to avoid overlap
 * (maybe) if the new position is farther away than some constant threshold,
     collapse "into" the overlapping label

*/

pub fn debug_window(ctx: &egui::Context, app: &PafViewerApp, view: &crate::view::View) {
    let labels = vec![
        ("LABEL ONE".to_string(), 3000f64..20_000f64),
        ("second LABEL".to_string(), 18_000f64..28_000f64),
        ("Label no. 03".to_string(), 30_000f64..33_000f64),
        ("another label".to_string(), 50_000f64..57_000f64),
    ];

    let painter = ctx.debug_painter();

    let screen_size = ctx.screen_rect().size();

    let id = egui::Id::new("label-debugger");
    let placed_labels = ctx
        .data(|data| data.get_temp::<Arc<Vec<(egui::Pos2, Arc<Galley>)>>>(id))
        .unwrap_or_default();

    if !placed_labels.is_empty() {
        for (pos, galley) in placed_labels.iter() {
            painter.galley(*pos, galley.clone(), egui::Color32::BLACK);
        }
    }

    let binned_index = ctx
        .data(|data| data.get_temp::<Option<Arc<BinnedCigarIndex>>>(id))
        .unwrap_or_default();

    if binned_index.is_none() {
        let alignment = app.alignments.pairs.values().next().unwrap();
        let binned_index = bin_cigar_index(&alignment.cigar, 4096);
        ctx.data_mut(|data| data.insert_temp(id, Some(Arc::new(binned_index))));
    }

    let mat = view.to_mat4();

    if let Some(bins) = binned_index.as_ref() {
        let alignment = app.alignments.pairs.values().next().unwrap();
        let world_y0 = alignment.location.query_range.start as f32;
        let world_x0 = alignment.location.target_range.start as f32;

        let world_origin =
            ultraviolet::Vec4::new(world_x0, bins.bins[0] as f32 + world_y0, 0.0, 0.0);
        let screen_origin = mat * world_origin;

        println!("screen_origin: {screen_origin:?}");
        let rect = egui::Rect::from_center_size(
            [screen_origin.x, screen_origin.y].into(),
            [50.0, 50.0].into(),
        );
        painter.rect(
            rect,
            0.0,
            egui::Color32::BLUE,
            egui::Stroke::new(2.0, egui::Color32::GREEN),
        );
    }

    egui::Window::new("Label Placement Test").show(&ctx, |ui| {
        if ui.button("compute layout").clicked() {
            let alignment = app.alignments.pairs.values().next().unwrap();
            let new_labels = place_labels(
                &app.alignment_grid,
                alignment,
                &labels,
                screen_size,
                &painter,
                view,
            );

            ctx.data_mut(|data| data.insert_temp(id, Arc::new(new_labels)));
        }
        //
    });
}

// target only for now; coordinates are world, but focusing on a single tile for now
// (in practice, using only the top visible tile in the relevant column(s) might work)
pub fn place_labels(
    grid: &crate::grid::AlignmentGrid,
    alignment: &crate::paf::Alignment,
    // match_lines: &[[DVec2; 2]],
    labels: &[(String, std::ops::Range<f64>)],
    canvas_size: impl Into<[f32; 2]>,
    painter: &egui::Painter,
    view: &crate::view::View,
) -> Vec<(egui::Pos2, Arc<Galley>)> {
    // ) {
    // let canvas_size = painter.s
    let canvas_size = canvas_size.into();

    let mut out = Vec::new();

    for (text, world_range) in labels {
        let Some((_, seq_start)) = grid
            .x_axis
            .global_to_axis_exact(world_range.start as u64)
            .filter(|(seq_id, _)| *seq_id == alignment.target_id)
        else {
            continue;
        };

        let Some((_, seq_end)) = grid
            .x_axis
            .global_to_axis_exact(world_range.end as u64)
            .filter(|(seq_id, _)| *seq_id == alignment.target_id)
        else {
            continue;
        };

        // first try to place the label on the center of the
        // intersection of the world range with the view (horizontally)
        let x_min = view
            .map_world_to_screen(canvas_size, [world_range.start, 0.0])
            .x;

        let x_max = view
            .map_world_to_screen(canvas_size, [world_range.end, 0.0])
            .x;

        let x_mid = 0.5 * (x_min + x_max);

        // compute the Y coordinate (i.e. the query offset at the target
        // corresponding to x_mid)

        // first need to convert x_mid to sequence loca
        let Some((_, seq_x_mid)) = grid
            .x_axis
            .global_to_axis_exact((0.5 * (world_range.end + world_range.start)) as u64)
        // .global_to_axis_local(0.5 * (world_range.end + world_range.start))
        else {
            continue;
        };

        // println!("{seq_x_mid}");
        // let world_y_mid = alignment.query_offset_at_target(seq_x_mid);
        // println!("{world_y_mid}");

        // let screen_y_mid = view
        //     .map_world_to_screen(canvas_size, [0.0, world_y_mid as f64])
        //     .y;

        let Some((_, seq_x)) = grid.x_axis.global_to_axis_exact(world_range.end as u64)
        // .global_to_axis_local(0.5 * (world_range.end + world_range.start))
        else {
            continue;
        };
        let world_y = alignment.query_offset_at_target(seq_x);
        // let world_y = alignment.query_offset_at_target(world_range_end);

        let screen_y = view
            .map_world_to_screen(canvas_size, [0.0, world_y as f64])
            .y;

        let galley = painter.layout(
            text.into(),
            egui::FontId::monospace(16.0),
            egui::Color32::BLACK,
            128.0,
        );

        out.push(([x_min, screen_y].into(), galley));
    }

    //

    out
}

pub fn compute_layout_for_labels<'a, Labels>(
    // grid: &crate::AlignmentGrid,
    // galley_cache: &FxHashMap<String, Arc<Galley>>,
    canvas_size: impl Into<[f32; 2]>,
    view: &crate::view::View,
    labels: Labels,
) -> Vec<(egui::Pos2, Arc<Galley>)>
where
    Labels: IntoIterator<Item = LabelDef<'a>> + 'a,
{
    let canvas_size = canvas_size.into();
    // let dims = [width as u32, height as u32];
    // let [width, height] = canvas_size.into();
    // let dims = [width as u32, height as u32];

    // for the menubar; TODO: actually take the menubar size into
    // account (here and elsewhere)
    let top_padding = 32f32;
    let pad_world = (top_padding as f64 / canvas_size[1] as f64) * view.height();

    // let y_top = view.y_min + pad_world;
    let y_top = view.y_max - pad_world;
    // let x_min = view_x_min;

    let mut placed_shapes: Vec<(Cuboid, Isometry2<f32>)> = Vec::new();
    let mut placed_labels = Vec::new();

    for label in labels {
        let galley = &label.galley;

        let size = galley.size();

        let size_arr: [f32; 2] = size.into();

        let shape = Cuboid::new(vector![size.x, size.y] * 0.5);
        let x = (*label.world_x_region.start() + *label.world_x_region.end()) * 0.5;

        let origin_w = [x, y_top];
        let origin = view.map_world_to_screen(canvas_size, origin_w);

        let ray = Ray::new(na::Point2::new(origin.x, origin.y), [0.0, 1.0].into());
        log::warn!("casting ray: {ray:?}");

        let mut intersection = None;

        for (ix, (other_shape, iso)) in placed_shapes.iter().enumerate() {
            let Some(toi) = other_shape.cast_ray(&iso, &ray, 10_000.0, false) else {
                continue;
            };

            if let Some((_, prev_toi)) = intersection {
                if prev_toi > toi {
                    intersection = Some((ix, toi));
                }
            } else {
                intersection = Some((ix, toi));
            }
        }

        if let Some((other_ix, toi)) = intersection {
            let impact = ray.point_at(toi);
            log::warn!("impact! with {other_ix} at {impact:?}");
            let top = impact - vector![0.0, size.y];
            let iso = Isometry2::new([top.x, top.y].into(), 0.0);
            placed_shapes.push((shape, iso));

            let label_pos = egui::pos2(top.x - size.x * 0.5, top.y);
            placed_labels.push((label_pos, galley.clone()));
        } else {
            let top = vector![ray.origin.coords.x, canvas_size[1] - 32.0];
            log::warn!("no collision, placing at {top:?}");

            let iso = Isometry2::new([top.x, top.y].into(), 0.0);
            placed_shapes.push((shape, iso));

            let label_pos = egui::pos2(top.x - size.x * 0.5, top.y);
            placed_labels.push((label_pos, galley.clone()));
        }
    }

    placed_labels
}

// Binned linear approximation of the "height" of the alignment
// above the major axis
pub struct BinnedCigarIndex {
    pub bins: Vec<f64>,
    pub x_min: f64,
    pub bin_size: f64,
}

impl BinnedCigarIndex {
    pub fn new(bins: Vec<f64>, x_min: f64, bin_size: f64) -> Self {
        BinnedCigarIndex {
            bins,
            x_min,
            bin_size,
        }
    }

    pub fn lookup(&self, x: f64) -> f64 {
        if self.bins.is_empty() {
            return 0.0; // No bins, return default value
        }

        // Calculate bin index
        let index = ((x - self.x_min) / self.bin_size).floor() as usize;

        // If the index is out of bounds, clamp to the bounds of the bins
        if index >= self.bins.len() - 1 {
            return *self.bins.last().unwrap();
        }

        // Linear interpolation
        let x0 = self.x_min + index as f64 * self.bin_size;
        let y0 = self.bins[index];
        let x1 = x0 + self.bin_size;
        let y1 = self.bins[index + 1];

        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

pub fn bin_cigar_index(cigar: &crate::CigarIndex, bin_count: usize) -> BinnedCigarIndex {
    // Ensure there are points and bins to process
    if cigar.op_target_offsets.is_empty() || bin_count == 0 {
        return BinnedCigarIndex::new(vec![], 0.0, 0.0);
    }

    // Find the range of X values
    let x_min = *cigar.op_target_offsets.first().unwrap();
    let x_max = *cigar.op_target_offsets.last().unwrap();
    let x_range = x_max - x_min;
    let bin_size = x_range as f64 / bin_count as f64;

    // Initialize bins
    let mut bins = vec![0.0; bin_count];
    let mut bin_counts = vec![0; bin_count];

    // Populate bins with average Y values
    for (x, y) in cigar.op_target_offsets.iter().zip(&cigar.op_query_offsets) {
        let bin_index = (((*x - x_min) as f64 / bin_size) as usize).min(bin_count - 1);
        bins[bin_index] += *y as f64;
        bin_counts[bin_index] += 1;
    }

    // Compute the average for each bin
    for i in 0..bin_count {
        if bin_counts[i] > 0 {
            bins[i] /= bin_counts[i] as f64;
        }
    }

    BinnedCigarIndex::new(bins, x_min as f64, bin_size)
}

/*
pub fn bin_cigar_index(cigar: &crate::CigarIndex, bin_count: usize) -> BinnedCigarIndex {
    // Ensure there are points and bins to process
    if cigar.op_target_offsets.is_empty() || bin_count == 0 {
        return BinnedCigarIndex { bins: vec![] };
    }

    // Find the range of X values
    let x_min = *cigar.op_target_offsets.first().unwrap();
    let x_max = *cigar.op_target_offsets.last().unwrap();
    let x_range = x_max - x_min;
    let bin_size = x_range as f64 / bin_count as f64;

    // Initialize bins
    let mut bins = vec![0.0; bin_count];
    let mut bin_counts = vec![0; bin_count];

    // Populate bins with average Y values
    for (x, y) in cigar.op_target_offsets.iter().zip(&cigar.op_query_offsets) {
        let bin_index = (((*x - x_min) as f64 / bin_size) as usize).min(bin_count - 1);
        bins[bin_index] += *y as f64;
        bin_counts[bin_index] += 1;
    }

    // Compute the average for each bin
    for i in 0..bin_count {
        if bin_counts[i] > 0 {
            bins[i] /= bin_counts[i] as f64;
        }
    }

    BinnedCigarIndex { bins }
}
*/

/*
impl BinnedCigarIndex {
    // Method to get interpolated value for a given x
    pub fn interpolate(&self, x: u64, x_min: u64, x_max: u64, bin_count: usize) -> f64 {
        if self.bins.is_empty() {
            return 0.0; // Return default if there are no bins
        }

        // Calculate bin size
        let x_range = x_max - x_min;
        let bin_size = x_range as f64 / bin_count as f64;

        // Handle edge cases
        if x <= x_min {
            return self.bins[0];
        }
        if x >= x_max {
            return self.bins.last().unwrap().clone();
        }

        // Calculate bin indices
        let bin_index = ((x - x_min) as f64 / bin_size) as usize;
        let bin_index_next = bin_index + 1;

        // Handle the last bin case
        if bin_index_next >= self.bins.len() {
            return self.bins[bin_index];
        }

        // Linear interpolation
        let x1 = x_min as f64 + bin_index as f64 * bin_size;
        let x2 = x_min as f64 + bin_index_next as f64 * bin_size;
        let y1 = self.bins[bin_index];
        let y2 = self.bins[bin_index_next];

        // Interpolation formula
        let interpolated_value = y1 + (y2 - y1) * ((x as f64 - x1) / (x2 - x1));
        interpolated_value
    }
}
*/

struct LabelState {
    galley: Arc<Galley>,

    anchor_world: Option<ultraviolet::DVec2>,
    pos_screen: Option<ultraviolet::Vec2>,
}

pub struct LabelLayoutState {
    // world space
    labels: Vec<(String, std::ops::Range<f64>)>,

    alignment_heights: FxHashMap<(SeqId, SeqId), BinnedCigarIndex>,

    label_state: Vec<LabelState>,
}

impl LabelLayoutState {
    pub fn from_alignments(
        alignments: &FxHashMap<(SeqId, SeqId), crate::paf::Alignment>,
        labels: impl IntoIterator<Item = (String, std::ops::Range<f64>)>,
    ) -> Self {
        let mut alignment_heights = FxHashMap::default();

        for (key, alignment) in alignments.iter() {
            let binned = bin_cigar_index(&alignment.cigar, 4096);
            alignment_heights.insert(key.clone(), binned);
        }

        LabelLayoutState {
            labels: labels.into_iter().collect(),
            alignment_heights,
            label_state: Vec::new(),
        }
    }

    pub fn update(&mut self, ctx: &egui::Context, view: &crate::view::View) {
        // initialize galleys so the size of each label is known
        if self.labels.len() > self.label_state.len() {
            ctx.fonts(|fonts| {
                let to_add = self.label_state.len()..self.labels.len();

                for i in to_add {
                    let (text, _) = &self.labels[i];
                    let galley = fonts.layout(
                        text.clone(),
                        egui::FontId::monospace(12.0),
                        egui::Color32::BLACK,
                        256.0,
                    );

                    self.label_state.push(LabelState {
                        galley,
                        anchor_world: None,
                        pos_screen: None,
                    });
                }
            })
        }

        // need to know which alignments to check at this point

        let screen_dims = ctx.screen_rect().size();

        for ((_, world_range), state) in std::iter::zip(&self.labels, &self.label_state) {
            //
        }
    }
}

pub struct LabelLayoutTest {
    // width in pixels, range in world space
    labels: Vec<(f32, std::ops::Range<f64>)>,
    heights: BinnedCigarIndex,
}

impl LabelLayoutTest {
    pub fn show(&mut self, ctx: &egui::Context) {
        egui::Window::new("LabelLayoutTest2").show(ctx, |ui| {
            if ui.button("Initialize").clicked() {
                // ...
            }
        });
    }
}
