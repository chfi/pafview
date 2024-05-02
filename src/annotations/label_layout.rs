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

use crate::PafViewerApp;

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




*/

// Binned linear approximation of the "height" of the alignment
// above the major axis
struct AlignmentHeightMap {
    len: u64,
    bin_size: u64,
    bins: Vec<u64>,
}

impl AlignmentHeightMap {
    fn alignment_query_height(alignment: &crate::paf::Alignment) -> Self {
        let bin_count = 4096;
        let len = alignment.location.aligned_target_len();
        let bin_size = len / bin_count;

        let mut bins = Vec::with_capacity(bin_count as usize);

        let mut last_bin_end = None;

        for i in 0..bin_count {
            todo!();
        }

        AlignmentHeightMap {
            len,
            bin_size,
            bins,
        }
    }
}

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
