use bevy::{math::DVec2, prelude::*, utils::HashMap};

use crate::{app::SequencePairTile, sequences::SeqId};

#[derive(Component, Clone)]
pub struct SeqPairLayout {
    pub offsets: HashMap<SequencePairTile, DVec2>,
}

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
        let offsets = match self.data {
            LayoutInput::Axes { targets, queries } => {
                let mut offsets = HashMap::default();

                let mut x_offset = 0.0;

                for target in targets {
                    let Some(tgt_seq) = sequences.get(target) else {
                        continue;
                    };

                    let x0 = x_offset;
                    x_offset += self.horizontal_offset.unwrap_or(tgt_seq.len() as f64);

                    let mut y_offset = 0.0;

                    for &query in &queries {
                        let Some(qry_seq) = sequences.get(query) else {
                            continue;
                        };

                        let y0 = y_offset;
                        y_offset += self.vertical_offset.unwrap_or(qry_seq.len() as f64);

                        offsets.insert(SequencePairTile { target, query }, [x0, y0].into());
                    }
                }

                offsets
            }
            LayoutInput::TilePositions { offsets } => offsets,
        };

        SeqPairLayout { offsets }
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

    pub fn with_vertical_offset(&mut self, offset: Option<f64>) -> &mut Self {
        self.vertical_offset = offset;
        self
    }

    pub fn with_horizontal_offset(&mut self, offset: Option<f64>) -> &mut Self {
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
