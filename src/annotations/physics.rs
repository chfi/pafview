use rapier2d::prelude::*;
use rustc_hash::FxHashMap;
use ultraviolet::Vec2;

use crate::grid::{AlignmentGrid, AxisRange, GridAxis};
use crate::paf::{AlignmentLocation, Alignments};
use crate::sequences::SeqId;
use crate::view::Viewport;
use crate::{paf::Alignment, view::View};

use crate::math_conv::*;

use super::AnnotationStore;

/*

TODO:

implement (individual) heightfield logic
- height_at_query is the only one that requires anything new
-*- just need to map query to target position & use height_at_target
- heightfield_rectangle_collision function
-*- same one should work for target & query, but may require a transform/extra step for the query case

implement AlignmentHeightFields
- from_alignments()
- choose "best" heightfield in a defined column (for target; row for query) for a view
-*- e.g. above if there's space, or below otherwise... there should always be space


implement LabelPhysics::update_anchors()
- initialize positions for anchors, or update them (if out of screen bounds, remove)

methods (on LabelPhysics) for checking positions &c for labels
- find_position_for_rect_in_aabb and/or find_position_for_screen_rectangle


implement LabelPhysics::update_labels()
- try to place labels by respective anchor (if anchor on screen & label not already placed)

implement LabelPhysics::step()
- apply spring force on label toward anchor, and collision with heightfield

integrate with annotation store, GUI, and rendering

*/
#[derive(Default)]
pub struct LabelPhysics {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub query_pipeline: QueryPipeline,

    physics: Physics,

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
    }
}

impl LabelPhysics {
    pub fn update_anchors(&mut self, viewport: &Viewport) {
        for (annot_id, annot_data) in self.annotations.iter() {
            // target for now

            // find anchor target (X) pos (middle of screen intersection w/ range)

            // project target onto heightfield

            // update annotation anchor position (remove if offscreen)
        }

        //
        todo!();
    }

    pub fn update_labels(&mut self, viewport: &Viewport) {
        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            if let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] {
                if self.target_labels.label_rigid_body[handle_ix].is_none() {
                    // initialize label rigid body
                    // create collider
                    // position
                    todo!();
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
                }
            }
        }

        todo!();
    }

    pub fn step(&mut self, dt: f32, viewport: &Viewport) {
        self.physics.step(
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.query_pipeline,
        );
    }
}

impl LabelPhysics {
    fn find_position_for_screen_rectangle(
        &self,
        viewport: &Viewport,
        proposed_center: impl Into<[f32; 2]>,
        rect_size: impl Into<[f32; 2]>,
    ) -> Option<ultraviolet::Vec2> {
        let screen_world = viewport.screen_world_mat3();

        let center = proposed_center.as_uv();
        let size = rect_size.as_uv();

        // idk if this is right
        let screen_norm_size = size / viewport.canvas_size;
        let world_size = screen_norm_size * viewport.view_size;

        let world_center = screen_world.transform_point2(center);

        // find height using heightmap
        let ground_y = todo!();
        // let ground_y = self.heightfield_project_world_x(world_center.x + world_size.x * 0.5)?;

        let screen_center = viewport
            .world_screen_mat3()
            .transform_point2([world_center.x, ground_y].as_uv());

        let this_shape = Cuboid::new((size * 0.5).as_na());

        let mut intersecting = Vec::new();
        let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, screen_center.y);

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

struct AlignmentHeightFields {
    heightfields: FxHashMap<(SeqId, SeqId), LabelHeightField>,
}

impl AlignmentHeightFields {
    fn from_alignments(grid: &AlignmentGrid, alignments: &Alignments) -> Self {
        let bin_count = 4096;

        let mut heightfields = FxHashMap::default();

        for (&tile, alignment) in &alignments.pairs {
            let hfield = LabelHeightField::from_alignment(alignment, bin_count);
            heightfields.insert(tile, hfield);
        }

        Self { heightfields }
    }

    fn top_heightfield_in_visible_column(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        world_target: f64,
    ) -> Option<&LabelHeightField> {
        let (tgt_id, seq_pos) = grid.x_axis.global_to_axis_local(world_target)?;

        todo!();
        None
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
        let location = alignment.location;
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
        let heightfield = HeightField::new(heights, [scale_x, 1.0]);

        Self { heightfield }
    }
}

impl LabelHeightField {
    fn heightfield_aabb(&self) -> Aabb {
        let pos = nalgebra::Isometry2::translation(self.heightfield.scale().x * 0.5, 0.0);
        self.heightfield.compute_aabb(&pos)
    }

    fn heightfield_project_world_x(&self, x: f32) -> Option<f32> {
        if x < 0.0 || x > self.heightfield.scale().x {
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

    fn heightfield_project_screen(
        &self,
        viewport: &Viewport,
        screen_point: impl Into<[f32; 2]>,
    ) -> Option<Vec2> {
        let pt = screen_point.as_uv();
        let world = viewport.screen_world_mat3().transform_point2(pt);
        let y = self.heightfield_project_world_x(world.x)?;
        Some([world.x, y].as_uv())
    }

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

    fn height_at_target(&self, align_tgt_pos: f64) -> Option<f64> {
        todo!();
    }

    fn height_at_query(&self, align_qry_pos: f64) -> Option<f64> {
        todo!();
    }
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
            // gravity: vector![0.0, 0.0],
            gravity: vector![0.0, 9.81],
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
