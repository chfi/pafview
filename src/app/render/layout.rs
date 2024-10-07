use bevy::{math::DVec2, prelude::*, utils::HashMap};

// use rapier2d::parry;
use avian2d::parry::{self, bounding_volume::Aabb};

use crate::{app::SequencePairTile, sequences::SeqId};

#[derive(Component, Clone)]
pub struct SeqPairLayout {
    pub aabbs: HashMap<SequencePairTile, Aabb>,

    pub layout_qbvh: LayoutQbvh,
}

#[derive(Resource, Clone)]
pub struct DefaultLayout(pub SeqPairLayout);

pub struct LayoutBuilder {
    /// stack tiles using this uniform (cumulative) offset, allowing overlap,
    /// instead of lining them up side-by-side
    /// ignored if builder was created from tile positions
    pub vertical_offset: Option<f64>,

    /// see vertical_offset
    pub horizontal_offset: Option<f64>,

    data: LayoutInput,
}

impl LayoutBuilder {
    pub fn build(self, sequences: &crate::Sequences) -> SeqPairLayout {
        let aabbs = match self.data {
            LayoutInput::Axes { targets, queries } => {
                let mut aabbs = HashMap::default();

                let mut x_offset = 0.0;

                for target in targets {
                    let Some(tgt_seq) = sequences.get(target) else {
                        continue;
                    };

                    let tgt_len = tgt_seq.len() as f64;

                    let x0 = x_offset;
                    x_offset += self.horizontal_offset.unwrap_or(tgt_len);

                    let mut y_offset = 0.0;

                    for &query in &queries {
                        let Some(qry_seq) = sequences.get(query) else {
                            continue;
                        };
                        let qry_len = qry_seq.len() as f64;

                        let y0 = y_offset;
                        y_offset += self.vertical_offset.unwrap_or(qry_len);

                        let center = [x0, y0];

                        let half_extents = [tgt_len * 0.5, qry_len * 0.5];

                        let aabb = Aabb::from_half_extents([x0, y0].into(), half_extents.into());

                        aabbs.insert(SequencePairTile { target, query }, aabb);
                    }
                }

                aabbs
            }
            LayoutInput::TilePositions { offsets } => offsets
                .iter()
                .filter_map(|(&seq_pair, &offset)| {
                    let tgt_len = sequences.get(seq_pair.target)?.len() as f64;
                    let qry_len = sequences.get(seq_pair.query)?.len() as f64;
                    let half_extents = [tgt_len * 0.5, qry_len * 0.5];
                    let aabb =
                        Aabb::from_half_extents([offset.x, offset.y].into(), half_extents.into());
                    Some((seq_pair, aabb))
                })
                .collect(),
        };

        let layout_qbvh = LayoutQbvh::from_tiles(aabbs.iter().map(|(&sp, &aabb)| (sp, aabb)));

        SeqPairLayout { aabbs, layout_qbvh }
    }

    pub fn from_axes<T, Q>(targets: T, queries: Q) -> Self
    where
        T: IntoIterator<Item = SeqId>,
        Q: IntoIterator<Item = SeqId>,
    {
        let targets = targets.into_iter().collect::<Vec<_>>();
        let queries = queries.into_iter().collect::<Vec<_>>();

        let data = LayoutInput::Axes { targets, queries };
        Self {
            vertical_offset: None,
            horizontal_offset: None,
            data,
        }
    }

    pub fn from_positions<P: Into<DVec2>>(
        tile_positions: impl IntoIterator<Item = (SequencePairTile, P)>,
    ) -> Self {
        let offsets = tile_positions
            .into_iter()
            .map(|(t, p)| (t, p.into()))
            .collect();

        Self {
            vertical_offset: None,
            horizontal_offset: None,
            data: LayoutInput::TilePositions { offsets },
        }
    }

    pub fn with_vertical_offset(mut self, offset: Option<f64>) -> Self {
        self.vertical_offset = offset;
        self
    }

    pub fn with_horizontal_offset(mut self, offset: Option<f64>) -> Self {
        self.horizontal_offset = offset;
        self
    }
}

enum LayoutInput {
    Axes {
        targets: Vec<SeqId>,
        queries: Vec<SeqId>,
    },
    TilePositions {
        offsets: HashMap<SequencePairTile, DVec2>,
    },
}

#[derive(Clone)]
pub struct LayoutQbvh {
    qbvh: parry::partitioning::Qbvh<usize>,
    tile_index_map: Vec<SequencePairTile>,
    // tile_index_map: HashMap<usize,
}

impl LayoutQbvh {
    pub fn from_tiles<T>(tiles: T) -> Self
    where
        T: ExactSizeIterator<Item = (SequencePairTile, Aabb)>,
    {
        use parry::partitioning::Qbvh;

        let (tile_index_map, leaf_data): (
            Vec<SequencePairTile>,
            Vec<(usize, parry::bounding_volume::Aabb)>,
        ) = tiles
            .enumerate()
            .map(|(ix, (seq_pair, aabb))| (seq_pair, (ix, aabb)))
            .unzip();

        let mut qbvh = Qbvh::new();
        qbvh.clear_and_rebuild(leaf_data.into_iter(), 1.0);

        Self {
            qbvh,
            tile_index_map,
        }
    }

    pub fn tiles_in_rect_callback(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
        mut callback: impl FnMut(SequencePairTile) -> bool,
    ) {
        let leaf_cb = &mut |index: &usize| {
            let seq_pair = self.tile_index_map[*index];
            callback(seq_pair)
        };

        let center = center.into();
        let half_extents = half_extents.into();

        let aabb =
            parry::bounding_volume::Aabb::from_half_extents(center.into(), half_extents.into());
        let mut visitor =
            parry::query::visitors::BoundingVolumeIntersectionsVisitor::new(&aabb, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn tiles_in_rect(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
    ) -> Vec<SequencePairTile> {
        let mut results = Vec::new();

        self.tiles_in_rect_callback(center, half_extents, |tile| {
            results.push(tile);
            true
        });

        results
    }

    pub fn tiles_at_point_callback(
        &self,
        point: impl Into<[f64; 2]>,
        mut callback: impl FnMut(SequencePairTile) -> bool,
    ) {
        let leaf_cb = &mut |index: &usize| {
            let seq_pair = self.tile_index_map[*index];
            callback(seq_pair)
        };

        let query_pt = point.into().into();
        let mut visitor =
            parry::query::visitors::PointIntersectionsVisitor::new(&query_pt, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn tiles_at_point(&self, point: impl Into<[f64; 2]>) -> Vec<SequencePairTile> {
        let mut results = Vec::new();

        self.tiles_at_point_callback(point, |tile| {
            results.push(tile);
            true
        });

        results
    }
}
