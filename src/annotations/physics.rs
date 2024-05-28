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
    anchor_rigid_body: Vec<Option<RigidBodyHandle>>,
}

impl LabelHandles {
    fn push(&mut self, annot_id: super::AnnotationId) -> usize {
        let id = self.annotation_id.len();
        self.annotation_id.push(annot_id);
        self.anchor_screen_pos.push(None);
        self.anchor_rigid_body.push(None);
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

            // project target onto heightfield*
            //  -- need to know which heightfield to use here
        }

        //
        todo!();
    }

    pub fn update_labels(&mut self, viewport: &Viewport) {
        //
        todo!();
    }

    pub fn step(&mut self, dt: f32, viewport: &Viewport) {
        todo!();
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
        todo!();
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
