use rapier2d::prelude::*;
use rustc_hash::FxHashMap;
use ultraviolet::Vec2;

use crate::grid::{AlignmentGrid, AxisRange, GridAxis};
use crate::paf::{AlignmentLocation, Alignments};
use crate::sequences::SeqId;
use crate::view::Viewport;
use crate::{paf::Alignment, view::View};

use crate::math_conv::*;

use super::draw::AnnotationPainter;
use super::AnnotationStore;

pub mod debug;

/*

TODO:

- finish heightfield projection (doneish)
- anchor initialization & update (done)
- label creation (doneish)
- enable/disable physics & rendering via existing UI
- hooking up to rendering
-*- DrawAnnotation for AnnotationLabel chooses positions entirely by itself;
-*-*-  we need to feed the positions to the AnnotationPainter somehow
-*-*-  makes sense to have some sort of position associated with pretty much any displayed annotation, anyway


- labels for query annotations

*/

#[derive(Default)]
pub struct LabelPhysics {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub query_pipeline: QueryPipeline,

    physics: Physics,

    pub heightfields: AlignmentHeightFields,

    // annotation_map: AnnotationDefs,
    annotations: FxHashMap<super::AnnotationId, AnnotationData>,
    // annotations_label_map: FxHashMap<super::AnnotationId, AnnotationLabelIxs>,
    target_labels: LabelHandles,
    query_labels: LabelHandles,
}

struct AnnotationData {
    size: Vec2,
    world_range: AnnotationRange,

    // indices into the vectors stored in the `target_labels` and `query_labels`
    // `LabelHandles` in `LabelPhysics`
    target_label_ix: usize,
    query_label_ix: usize,
}

#[derive(Default)]
struct LabelHandles {
    annotation_id: Vec<super::AnnotationId>,
    anchor_screen_pos: Vec<Option<Vec2>>,
    label_rigid_body: Vec<Option<RigidBodyHandle>>,
}

impl LabelHandles {
    fn push(&mut self, annot_id: super::AnnotationId) -> usize {
        let id = self.annotation_id.len();
        self.annotation_id.push(annot_id);
        self.anchor_screen_pos.push(None);
        self.label_rigid_body.push(None);
        id
    }
}

impl LabelPhysics {
    pub fn prepare_annotations(
        &mut self,
        grid: &AlignmentGrid,
        annotation_store: &AnnotationStore,
        annotations: impl IntoIterator<Item = super::AnnotationId>,
        fonts: &egui::text::Fonts, // needed to derive text size
        annotation_painter: &mut super::draw::AnnotationPainter, // cache laid out text
    ) {
        let len = self.annotations.len();
        for annot_id @ (list_id, record_id) in annotations {
            if self.annotations.contains_key(&annot_id) {
                continue;
            }

            let Some(record) = annotation_store
                .list_by_id(list_id)
                .and_then(|list| list.records.get(record_id))
            else {
                continue;
            };

            let galley = annotation_painter.cache_label_fonts(fonts, &record.label);
            let size = galley.size().as_uv();

            let axis_range = AxisRange::Seq {
                seq_id: record.seq_id,
                range: record.seq_range.clone(),
            };
            let world_x_range = grid.x_axis.axis_range_into_global(&axis_range);
            let world_y_range = grid.y_axis.axis_range_into_global(&axis_range);

            let Some(world_range) = AnnotationRange::new(world_x_range, world_y_range) else {
                continue;
            };

            let target_label_ix = self.target_labels.push(annot_id);
            let query_label_ix = self.query_labels.push(annot_id);

            let data = AnnotationData {
                size,
                world_range,

                target_label_ix,
                query_label_ix,
            };

            self.annotations.insert(annot_id, data);
        }
        println!("prepared {} annotations", self.annotations.len() - len);
    }
}

impl LabelPhysics {
    pub fn update_anchors(&mut self, grid: &AlignmentGrid, viewport: &Viewport) {
        use ultraviolet::mat::*;
        use ultraviolet::vec::*;
        let world_screen = viewport.world_screen_dmat3();
        let [xs, ys, zs] = world_screen.cols;
        let ws_x2 = DMat3x2::new(DVec3x2::splat(xs), DVec3x2::splat(ys), DVec3x2::splat(zs));

        let (vx_min, vx_max) = viewport.x_range().into_inner();

        let mut count = 0;
        for (annot_id, annot_data) in self.annotations.iter() {
            // target for now

            // find anchor target (X) pos (middle of screen intersection w/ range)
            let Some(world_range) = annot_data.world_range.target_range() else {
                continue;
            };

            let (wx_min, wx_max) = world_range.clone().into_inner();

            let screen_xrange = ws_x2
                .transform_point2(DVec2x2::new([wx_min, wx_max].into(), [0.0, 0.0].into()))
                .x;

            let &[sx_min, sx_max] = screen_xrange.as_array_ref();

            let cl_min = sx_min.clamp(vx_min, vx_max);
            let cl_max = sx_max.clamp(vx_min, vx_max);

            let cl_mid = (cl_min + cl_max) * 0.5;

            // project target onto heightfield
            let Some(mid_y) =
                self.heightfields
                    .project_screen_from_top(grid, viewport, cl_mid as f32)
            else {
                continue;
            };

            // println!("viewport: [{viewport:?}]");
            // println!("anchor pos: [{cl_mid}, {mid_y}]");
            // let left_y = self
            //     .heightfields
            //     .project_screen_from_top(grid, viewport, sx_min as f32);
            // let right_y = self
            //     .heightfields
            //     .project_screen_from_top(grid, viewport, sx_max as f32);

            // update annotation anchor position (remove if offscreen)
            let new_anchor = if sx_max < vx_min
                || sx_min > vx_max
                || mid_y < 0.0
                || mid_y > viewport.canvas_size.y
            {
                None
            } else {
                Some([cl_mid as f32, mid_y].as_uv())
            };

            self.target_labels.anchor_screen_pos[annot_data.target_label_ix] = new_anchor;
            count += 1;
        }
        // println!("updated {count} anchors");
    }

    pub fn update_labels(
        &mut self,
        grid: &AlignmentGrid,
        annotations: &AnnotationStore,
        painter: &mut AnnotationPainter,
        viewport: &Viewport,
    ) {
        let mut position_count = 0;
        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let Some(shape_id) = annotations.target_shape_for(annot_id.0, annot_id.1) else {
                continue;
            };

            if let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] {
                position_count += 1;
                let mut not_enough_space = false;

                if self.target_labels.label_rigid_body[handle_ix].is_none() {
                    // initialize label rigid body
                    let size = annot_data.size;

                    if let Some(label_pos) =
                        self.try_to_place_label(grid, viewport, anchor_pos, size)
                    {
                        let collider = ColliderBuilder::cuboid(size.x * 0.5, size.y * 0.5)
                            .mass(1.0)
                            // .friction(0.1)
                            .build();

                        // println!("try_to_place_label results: {label_pos:?}");
                        let rigid_body = RigidBodyBuilder::dynamic()
                            .translation(label_pos.as_na())
                            .lock_rotations()
                            .linear_damping(3.0)
                            // .linear_damping(5.0)
                            .build();

                        let rb_handle = self.rigid_body_set.insert(rigid_body);
                        let collider_handle = self.collider_set.insert_with_parent(
                            collider,
                            rb_handle,
                            &mut self.rigid_body_set,
                        );
                        self.query_pipeline.update_incremental(
                            &self.collider_set,
                            &[collider_handle],
                            &[],
                            true,
                        );
                        self.target_labels.label_rigid_body[handle_ix] = Some(rb_handle);
                    } else {
                        not_enough_space = true;
                    };
                }

                // if label not visible,
                // show label, enable physics object

                let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                    .and_then(|rb_handle| {
                        let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                        Some((rb_handle, rigid_body))
                    })
                else {
                    continue;
                };

                if !rigid_body.is_enabled() {
                    rigid_body.set_enabled(true);
                }
                if let Some(label) = painter.get_shape_mut(shape_id) {
                    let pos = rigid_body.position().translation;
                    // println!("setting label position to {pos:?}");
                    label.set_position(Some(pos.as_epos2()));
                    // if !pos.x.is_nan() && !pos.y.is_nan() {
                    //     println!("setting label position to {pos:?}");
                    // }
                }
            } else {
                // hide label, disable physics object

                let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                    .and_then(|rb_handle| {
                        let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                        Some((rb_handle, rigid_body))
                    })
                else {
                    continue;
                };

                if rigid_body.is_enabled() {
                    rigid_body.set_enabled(false);
                    if let Some(label) = painter.get_shape_mut(shape_id) {
                        println!("unsetting label");
                        label.set_position(None);
                    }
                }
            }
        }

        // println!("labels with anchor positions: {position_count}");
    }

    pub fn step(&mut self, grid: &AlignmentGrid, dt: f32, viewport: &Viewport) {
        const SPRING_K: f32 = 10.0;
        const LABEL_ANCHOR_DIST_THRESHOLD: f32 = 300.0;
        // const CLUSTER_TIME_MIN_SEC: f32 = 1.0;
        const CLUSTER_TIME_MIN_SEC: f32 = 0.2;

        const STACK_MIN_FORCE_THRESHOLD: f32 = 10.0;

        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                .and_then(|rb_handle| {
                    let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                    Some((rb_handle, rigid_body))
                })
            else {
                continue;
            };

            let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] else {
                continue;
            };

            if !rigid_body.is_enabled() {
                continue;
            }

            let label_pos = rigid_body.position().translation.as_uv();
            let label_anchor_diff = label_pos - anchor_pos;
            // TODO check if stacking/clustering is needed (prediction etc.)

            if rigid_body.user_force().norm() > 0.0 {
                rigid_body.reset_forces(false);
            }

            let force_x = -label_anchor_diff.x * SPRING_K;

            let y_diff = label_anchor_diff.y;
            println!("anchor: {anchor_pos:?}\tlabel: {label_pos:?}");
            // println!("y_diff: {y_diff}");
            // let hfield_y_diff = self.heightfields.project_screen_from_top(grid, viewport, screen_x)

            // TODO apply vertical force (to clear heightfield/match lines)
            // let force_y = 0f32;
            let y_tgt = (anchor_pos.y - 100.0).max(10.0);
            let force_y = -(label_pos.y - y_tgt) * SPRING_K;
            // let force_y = -label_anchor_diff.y * SPRING_K;
            // let force_y = if label_anchor_diff.y.abs() > 200.0 {
            //     -label_anchor_diff.y * SPRING_K
            // } else {
            //     0.0
            // };
            let force = [force_x, force_y].as_uv();
            let force_min = 1.0; // arbitrary
            if force.mag() > force_min {
                rigid_body.add_force(force.as_na(), true);
            }

            // TODO handle contacts, stack
        }

        self.physics.step(
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.query_pipeline,
        );
    }
}

impl LabelPhysics {
    fn try_to_place_label(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        // anchor_screen_x_range: std::
        anchor_pos: Vec2,
        rect_size: impl Into<[f32; 2]>,
    ) -> Option<ultraviolet::Vec2> {
        // println!("try to place anchor pos: {anchor_pos:?}");
        let proposed_center = anchor_pos + [0.0, -40.0].as_uv();
        self.find_position_for_screen_rectangle(grid, viewport, proposed_center, rect_size)
    }

    fn find_position_for_screen_rectangle(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        proposed_center: impl Into<[f32; 2]>,
        rect_size: impl Into<[f32; 2]>,
    ) -> Option<ultraviolet::Vec2> {
        let screen_world = viewport.screen_world_mat3();

        let center = proposed_center.as_uv();
        let size = rect_size.as_uv();

        // idk if this is right
        let screen_norm_size = size / viewport.canvas_size;
        let world_size =
            screen_norm_size * Vec2::new(viewport.view_size.x as f32, viewport.view_size.y as f32);

        let world_center = screen_world.transform_point2(center);

        // find height using heightmap
        let ground_screen_y = self
            .heightfields
            .project_screen_from_top(grid, viewport, center.x)?;
        // let ground_y = todo!();
        // let ground_y = self.heightfield_project_world_x(world_center.x + world_size.x * 0.5)?;

        let screen_center = viewport
            .world_screen_mat3()
            .transform_point2([world_center.x, 0.0].as_uv());

        let this_shape = Cuboid::new((size * 0.5).as_na());

        let mut intersecting = Vec::new();
        let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, ground_screen_y);
        // let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, screen_center.y);

        let mut can_add = false;

        let mut iter_count = 0;
        const ITER_LIMIT: usize = 10;

        loop {
            let this_aabb = this_shape.aabb(&cur_position);

            // self.query_pipeline.intersections_with_shape(bodies, colliders, shape_pos, shape, filter, callback)
            intersecting.clear();
            self.query_pipeline
                .colliders_with_aabb_intersecting_aabb(&this_aabb, |other_handle| {
                    //
                    intersecting.push(*other_handle);
                    false
                });

            // {
            // let pos = (cur_position.translation.as_uv())
            //         painter.rect_stroke(
            //             egui::Rect::from_center_size(pos.as_epos2(), size.as_evec2()),
            //             0.0,
            //             egui::Stroke::new(1.0, egui::Color32::BLACK),
            //         );
            // }

            if intersecting.is_empty() {
                can_add = true;
                break;
            }

            for other_handle in &intersecting {
                let Some(other_aabb) = self
                    .collider_set
                    .get(*other_handle)
                    .map(|c| c.compute_aabb())
                else {
                    continue;
                };
                let overlap = aabb_overlap(&this_aabb, &other_aabb);
                if overlap.x.abs() > 0.0 && overlap.y.abs() > 0.0 {
                    cur_position.translation.y = other_aabb.center().y
                        - other_aabb.half_extents().y
                        - this_shape.half_extents.y
                        - 0.1;
                    //
                }
                //
            }

            if iter_count >= ITER_LIMIT {
                break;
            }
            iter_count += 1;
        }

        if can_add {
            Some(cur_position.translation.as_uv())
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct AlignmentHeightFields {
    heightfields: FxHashMap<(SeqId, SeqId), LabelHeightField>,
}

impl AlignmentHeightFields {
    pub fn from_alignments(alignments: &Alignments) -> Self {
        let bin_count = 4096;

        let mut heightfields = FxHashMap::default();

        for (&tile, alignment) in &alignments.pairs {
            let hfield = LabelHeightField::from_alignment(alignment, bin_count);
            heightfields.insert(tile, hfield);
        }

        Self { heightfields }
    }
}

impl AlignmentHeightFields {
    fn project_screen_from_top(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        screen_x: f32,
    ) -> Option<f32> {
        let mat = viewport.screen_world_dmat3();
        let world_x = mat.transform_point2([screen_x as f64, 0.0].as_duv()).x;

        let (qry_id, hfield) = self.top_heightfield_in_visible_column(grid, viewport, world_x)?;
        let (tgt_id, norm_x) = grid.x_axis.global_to_axis_local(world_x)?;

        // need to shift `world_x` to account for the (intended) offset of the heightfield
        let offset = grid.x_axis.sequence_offset(tgt_id)? as f64;

        let y_offset = grid.y_axis.sequence_offset(qry_id)? as f64;

        let hfield_y = hfield.heightfield_project_x((world_x - offset) as f32)?;
        let world_y = y_offset + hfield_y as f64;
        let world = [world_x, world_y];

        let screen = viewport
            .world_screen_dmat3()
            .transform_point2(world.as_duv());

        Some(viewport.canvas_size.y as f32 - screen.y as f32)

        // Some(screen.y as f32)
    }

    // fn project_screen_from_left(&self,
    //                             viewport: &Viewport,
    //                             screen_y: f32,
    // ) -> Option<f32> {
    //     todo!();
    // }

    // TODO this isn't trivial -- this solution will probably end up with some cases of overlap
    // as labels are drawn "too high", but a proper solution will likely need to take several
    // heightfields into account, and the view in relation to them (even actually projecting
    // the view X-range onto them)
    fn top_heightfield_in_visible_column(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        world_target: f64,
    ) -> Option<(SeqId, &LabelHeightField)> {
        let (tgt_id, seq_pos) = grid.x_axis.global_to_axis_local(world_target)?;

        // find map works since the axes are already sorted & the pairs are provided in that order
        let top_visible_qry = grid
            .pairs_with_target(tgt_id)
            .iter()
            // .rev()
            .find_map(|&qry_id| {
                let world_y_range = grid.y_axis.sequence_axis_range(qry_id)?;
                let min = world_y_range.start as f64;
                let max = world_y_range.end as f64;

                // if viewport.view_center.y -
                // let pair_location =
                //
                Some(qry_id)
            })?;

        let qry_id = top_visible_qry;

        let hfield = self.heightfields.get(&(tgt_id, qry_id))?;
        let qry_name = grid.sequence_names.get_by_right(&qry_id);
        println!("using heightfield for query: {qry_id:?} ({qry_name:?})");

        Some((qry_id, hfield))
    }

    fn left_heightfield_in_visible_row(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        world_query: f64,
    ) -> Option<&LabelHeightField> {
        todo!();
        None
    }
}

struct LabelHeightField {
    target_id: SeqId,
    query_id: SeqId,
    heightfield: HeightField,
    location: AlignmentLocation,
}

impl LabelHeightField {
    fn from_alignment(alignment: &Alignment, bin_count: usize) -> Self {
        let location = alignment.location.clone();
        let target_id = alignment.target_id;
        let query_id = alignment.query_id;

        let bin_size = location.aligned_target_len() as f32 / bin_count as f32;

        let mut bins = vec![0.0; bin_count];

        let mut y_max = f32::NEG_INFINITY;

        // TODO this can be done faster

        let mut bin_ix = 0;
        let mut current_bin_end = bin_size.floor() as usize;

        let cigar = &alignment.cigar;
        for (&x, &y) in std::iter::zip(&cigar.op_target_offsets, &cigar.op_query_offsets) {
            // shouldn't happen, but
            if bin_ix >= bins.len() {
                break;
            }

            let yf = y as f32;
            if bins[bin_ix] > yf {
                bins[bin_ix] = yf;
                y_max = y_max.max(yf);
            }

            if x as usize >= current_bin_end {
                bin_ix += 1;
                current_bin_end = ((bin_ix + 1) as f32 * bin_size).floor() as usize;
            }
        }

        let scale_x = location.aligned_target_len() as f32;
        let heights = nalgebra::DVector::from_vec(bins);
        let heightfield = HeightField::new(heights, [scale_x, 1.0].into());

        Self {
            heightfield,
            target_id,
            query_id,
            location,
        }
    }
}

impl LabelHeightField {
    fn heightfield_aabb(&self) -> Aabb {
        let pos = nalgebra::Isometry2::translation(self.heightfield.scale().x * 0.5, 0.0);
        self.heightfield.compute_aabb(&pos)
    }

    // input and output are in "heightfield-local" (but unnormalized) coordinates
    fn heightfield_project_x(&self, x: f32) -> Option<f32> {
        if x < 0.0 || x > self.heightfield.scale().x {
            println!(
                "out of scale: {x} not in [0.0, {}]",
                self.heightfield.scale().x
            );
            return None;
        }
        let x_ = x - self.heightfield.scale().x * 0.5;

        let point = nalgebra::Point2::new(x_, 0.0);

        // NB: parry's height_at_point() implementation is bugged (offsets by the segment's y)
        let inter_y = self
            .heightfield
            .cell_at_point(&point)
            .and_then(|cell| self.heightfield.segment_at(cell))
            .map(|seg| {
                rapier2d::parry::query::details::closest_points_line_line_parameters(
                    &seg.a,
                    &seg.scaled_direction(),
                    &point,
                    &Vector::y(),
                )
                .1
            });

        inter_y
    }

    /*
    fn heightfield_project_screen(
        &self,
        viewport: &Viewport,
        // screen_point: impl Into<[f32; 2]>,
        screen_x: f32,
    ) -> Option<Vec2> {
        // let pt = screen_point.as_uv();
        let world = viewport
            .screen_world_mat3()
            .transform_point2([screen_x, 0.0].as_uv());
        let y = self.heightfield_project_x(world.x)?;
        Some([world.x, y].as_uv())
    }
    */

    fn heightfield_screen_segments(
        &self,
        viewport: &Viewport,
    ) -> impl Iterator<Item = [egui::Pos2; 2]> + '_ {
        let aabb = self.heightfield_aabb();
        let offset = [aabb.half_extents().x, 0.0].as_uv();
        let mat = viewport.world_screen_mat3();

        self.heightfield.segments().map(move |Segment { a, b }| {
            let pa = mat.transform_point2(a.as_uv() + offset);
            let pb = mat.transform_point2(b.as_uv() + offset);
            [pa.as_epos2(), pb.as_epos2()]
        })
    }

    // fn height_at_target(&self, align_tgt_pos: f64) -> Option<f64> {
    //     todo!();
    // }

    // fn height_at_query(&self, align_qry_pos: f64) -> Option<f64> {
    //     todo!();
    // }
}

#[derive(Debug, Clone, PartialEq)]
enum AnnotationRange {
    // Symmetric {
    //     world_range: std::ops::RangeInclusive<f64>,
    // },
    Both {
        target_range: std::ops::RangeInclusive<f64>,
        query_range: std::ops::RangeInclusive<f64>,
    },
    TargetOnly(std::ops::RangeInclusive<f64>),
    QueryOnly(std::ops::RangeInclusive<f64>),
}

impl AnnotationRange {
    fn target_range(&self) -> Option<&std::ops::RangeInclusive<f64>> {
        match self {
            // AnnotationRange::Symmetric { world_range } => Some(world_range),
            AnnotationRange::Both { target_range, .. } => Some(target_range),
            AnnotationRange::TargetOnly(range) => Some(range),
            AnnotationRange::QueryOnly(_) => None,
        }
    }

    fn query_range(&self) -> Option<&std::ops::RangeInclusive<f64>> {
        match self {
            // AnnotationRange::Symmetric { world_range } => Some(world_range),
            AnnotationRange::Both { query_range, .. } => Some(query_range),
            AnnotationRange::TargetOnly(_) => None,
            AnnotationRange::QueryOnly(range) => Some(range),
        }
    }

    fn new(
        target_range: Option<std::ops::RangeInclusive<f64>>,
        query_range: Option<std::ops::RangeInclusive<f64>>,
    ) -> Option<Self> {
        match (target_range, query_range) {
            (None, None) => None,
            (None, Some(range)) => Some(Self::QueryOnly(range)),
            (Some(range), None) => Some(Self::TargetOnly(range)),
            (Some(target_range), Some(query_range)) => Some(Self::Both {
                target_range,
                query_range,
            }),
        }
    }
}

fn aabb_overlap(
    aabb1: &rapier2d::geometry::Aabb,
    aabb2: &rapier2d::geometry::Aabb,
) -> nalgebra::Vector2<f32> {
    let center1 = aabb1.center();
    let center2 = aabb2.center();

    let half_extents1 = aabb1.half_extents();
    let half_extents2 = aabb2.half_extents();

    let overlap_x = (half_extents1.x + half_extents2.x) - (center2.x - center1.x).abs();
    let overlap_y = (half_extents1.y + half_extents2.y) - (center2.y - center1.y).abs();

    let sign_x = (center2.x - center1.x).signum();
    let sign_y = (center2.y - center1.y).signum();

    nalgebra::Vector2::new(overlap_x * sign_x, overlap_y * sign_y)
}

struct Physics {
    gravity: nalgebra::Vector2<f32>,
    physics_pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
}

impl std::default::Default for Physics {
    fn default() -> Self {
        Self {
            gravity: vector![0.0, 0.0],
            // gravity: vector![0.0, 9.81],
            // gravity: vector![0.0, 20.0],
            // gravity: vector![0.0, 8.0],
            physics_pipeline: PhysicsPipeline::default(),
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::default(),
            broad_phase: DefaultBroadPhase::default(),
            narrow_phase: NarrowPhase::default(),
            impulse_joint_set: ImpulseJointSet::default(),
            multibody_joint_set: MultibodyJointSet::default(),
            ccd_solver: CCDSolver::default(),
        }
    }
}

impl Physics {
    fn step(
        &mut self,
        rigid_bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        query_pipeline: &mut QueryPipeline,
        // physics_hooks: Option<&dyn PhysicsHooks>,
        // event_handler: Option<&dyn EventHandler>,
    ) {
        let physics_hooks = ();
        let event_handler = ();

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            rigid_bodies,
            colliders,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(query_pipeline),
            &physics_hooks,
            &event_handler,
        );
    }
}

/*
pub mod draw {

    use egui::Galley;

    use crate::annotations::{draw::DrawAnnotation, AnnotationId};

    use super::*;
    use std::sync::{Arc, Mutex};

    // pub struct PhysicsAnnotationLabel {
    //     pos:
    // }

    pub struct DrawPhysicsLabels {
        positions: Arc<Mutex<Vec<Option<Vec2>>>>,
        galleys: Vec<Arc<Galley>>,
    }

    impl DrawPhysicsLabels {
        pub fn from_record_list(
            annotations: &AnnotationStore,
            annotation_painter: &mut AnnotationPainter,
            records: impl IntoIterator<Item = AnnotationId>,
        ) -> Self {
            let mut positions = Vec::new();
            let mut galleys = Vec::new();
        }

        pub fn swap_positions(&mut self, new_pos: &mut Vec<Option<Vec2>>) {
            let mut pos = self.positions.lock().unwrap();
            debug_assert_eq!(pos.len(), new_pos.len());
            std::mem::swap(pos.as_mut(), new_pos);
        }
    }

    // impl DrawPhysicsLabels {
    //     pub fn
    // }

    impl DrawAnnotation for DrawPhysicsLabels {
        fn draw(
            &self,
            _galley_cache: &mut FxHashMap<String, Arc<Galley>>,
            painter: &egui::Painter,
            _view: &crate::view::View,
            _screen_size: egui::Vec2,
        ) {
            let pos = self.positions.lock().unwrap();

            for (ix, &label_pos) in pos.iter().enumerate() {
                let Some(label_pos) = label_pos else {
                    continue;
                };

                let galley = &self.galleys[ix];
                let pos = label_pos.as_epos2() - galley.size() * 0.5;

                painter.galley(pos, galley.clone(), egui::Color32::BLACK);
            }
        }

        fn set_color(&mut self, _color: egui::Color32) {
            //
        }
    }
}

*/
