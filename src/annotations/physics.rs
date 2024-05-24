use rapier2d::prelude::*;
use rustc_hash::FxHashMap;
use ultraviolet::Vec2;

use crate::view::Viewport;
use crate::{paf::Alignment, view::View};

use crate::math_conv::*;

/*

need label sizes, ranges, anchors

alignment -> heightfield map

plus logic to handle query annotations, without duplicating too much code

*/

#[derive(Default)]
pub struct LabelPhysics {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub query_pipeline: QueryPipeline,

    // annotations: AnnotationDefs,
    annotations: FxHashMap<super::AnnotationId, AnnotationDef>,

    target_labels: LabelHandles,
    query_labels: LabelHandles,
}

// the data needed for the physics objects & spring constraint
struct AnnotationDef {
    size: Vec2,
    world_range: std::ops::RangeInclusive<f64>,
}

#[derive(Default)]
struct LabelHandles {
    annotation_id: Vec<super::AnnotationId>,
    anchor_screen_pos: Vec<Option<Vec2>>,
    anchor_rigid_body: Vec<Option<RigidBodyHandle>>,
}

pub struct LabelHeightField {
    heightfield: HeightField,
    // location: AlignmentLocation,
}

impl LabelPhysics {
    pub fn step(&mut self, dt: f32, viewport: &View) {
        todo!();
    }
}

impl LabelHeightField {
    pub fn from_alignment(alignment: &Alignment, bin_count: usize) -> Self {
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

    pub fn heightfield_screen_segments(
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

    pub fn height_at_target(&self, align_tgt_pos: f64) -> Option<f64> {
        todo!();
    }

    pub fn height_at_query(&self, align_qry_pos: f64) -> Option<f64> {
        todo!();
    }
}
