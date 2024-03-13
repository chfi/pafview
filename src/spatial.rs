use rstar::{
    primitives::{GeomWithData, Line},
    Point, RStarInsertionStrategy, RTree, RTreeNode, RTreeParams, AABB,
};
use ultraviolet::{DVec2, Vec2};

pub type MatchTree = RTree<TreePoint>;
pub type TreePoint = GeomWithData<Line<[f64; 2]>, usize>;

pub struct RStarMatches {
    pub tree: MatchTree,
}

impl RStarMatches {
    pub fn from_paf(input: &super::PafInput) -> Self {
        let elems = input
            .match_edges
            .iter()
            .enumerate()
            .map(|(ix, &[p0, p1])| TreePoint::new(Line::new(p0.into(), p1.into()), ix))
            .collect();
        let tree = MatchTree::bulk_load(elems);
        Self { tree }
    }

    pub fn lookup_screen_pos(
        &self,
        screen_dims: impl Into<[f32; 2]>,
        view: &crate::view::View,
        screen_pos: Vec2,
    ) -> Option<usize> {
        let dims = screen_dims.into();
        let wp: [f64; 2] = view.map_screen_to_world(dims, screen_pos).into();

        let val = self.tree.locate_at_point(&wp);
        val.map(|v| v.data)
    }
}
