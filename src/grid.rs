use std::sync::Arc;

use bimap::BiMap;
use rustc_hash::FxHashMap;

use crate::sequences::SeqId;

/// An `AlignmentGrid` defines the global position of the aligned sequence pairs
#[derive(Debug)]
pub struct AlignmentGrid {
    pub x_axis: GridAxis,
    pub y_axis: GridAxis,

    pub sequence_names: Arc<BiMap<String, SeqId>>,

    pairs_by_target: FxHashMap<SeqId, Vec<SeqId>>,
    pairs_by_query: FxHashMap<SeqId, Vec<SeqId>>,
}

impl AlignmentGrid {
    pub fn pairs_with_target(&self, target: SeqId) -> &[SeqId] {
        self.pairs_by_target
            .get(&target)
            .map(|q| q.as_slice())
            .unwrap_or_default()
    }

    pub fn pairs_with_query(&self, query: SeqId) -> &[SeqId] {
        self.pairs_by_query
            .get(&query)
            .map(|t| t.as_slice())
            .unwrap_or_default()
    }

    pub fn from_alignments(
        alignments: &crate::paf::Alignments,
        sequence_names: Arc<BiMap<String, SeqId>>,
        // x_axis: GridAxis,
        // y_axis: GridAxis,
    ) -> Self {
        let mut targets = alignments
            .pairs
            .values()
            .map(|al| (al.target_id, al.location.target_total_len))
            .collect::<Vec<_>>();
        targets.sort_by_key(|(_, l)| *l);
        targets.dedup_by_key(|(id, _)| *id);

        let x_axis = crate::grid::GridAxis::from_index_and_lengths(targets);
        let mut queries = alignments
            .pairs
            .values()
            .map(|al| (al.query_id, al.location.query_total_len))
            .collect::<Vec<_>>();
        queries.sort_by_key(|(_, l)| *l);
        queries.dedup_by_key(|(id, _)| *id);
        let y_axis = crate::grid::GridAxis::from_index_and_lengths(queries);

        let mut pairs_by_target: FxHashMap<_, Vec<_>> = FxHashMap::default();
        let mut pairs_by_query: FxHashMap<_, Vec<_>> = FxHashMap::default();

        for &(tgt_id, qry_id) in alignments.pairs.keys() {
            pairs_by_target.entry(tgt_id).or_default().push(qry_id);
            pairs_by_query.entry(qry_id).or_default().push(tgt_id);
        }

        for (_tgt_id, queries) in pairs_by_target.iter_mut() {
            queries.sort_by_cached_key(|qry| {
                y_axis.seq_index_map.get(qry).copied().unwrap_or_default()
            });
        }

        for (_qry_id, targets) in pairs_by_query.iter_mut() {
            targets.sort_by_cached_key(|tgt| {
                x_axis.seq_index_map.get(tgt).copied().unwrap_or_default()
            });
        }

        Self {
            x_axis,
            y_axis,
            sequence_names,
            pairs_by_target,
            pairs_by_query,
        }
    }
}

impl AlignmentGrid {
    // pub fn closest_points_screen(&self, viewport: &Viewport, point: Vec2) -> [Option<Vec2>; 2] {
    // pub fn closest_points_screen(&self, viewport: &crate::view::Viewport, point: Vec2) -> impl Iterator<Item = Vec2> {
    /*
    pub fn closest_points_screen(
        &self,
        canvas_size: [f32; 2],
        view: &crate::view::View,
        point: ultraviolet::Vec2,
    ) -> impl Iterator<Item = ultraviolet::Vec2> {
        // let screen_world = viewport.screen_world_mat3();
        // let world_screen = viewport.world_screen_mat3();

        let world_p = view.map_screen_to_world(canvas_size, point);
        let tgt = self.x_axis.global_to_axis_local(world_p.x);
        let qry = self.y_axis.global_to_axis_local(world_p.y);

        // let tgt = tgt.and_then(|(tgt_id, _local)| {
        // });

        // let (tgt_id, _tgt_local) =
        // let mat = viewport.screen
        todo!();
    }
    */
}

pub fn parse_axis_range_into_global(
    seq_names: &BiMap<String, SeqId>,
    axis: &GridAxis,
    text: &str,
) -> Option<AxisRange> {
    let mut split = text.split(':');
    let name = split.next()?;
    let id = *seq_names.get_by_left(name)?;

    let offset = axis.sequence_offset(id)?;
    // let offset = seqs[id].offset;

    let mut range = split
        .next()?
        .split('-')
        .filter_map(|s| s.parse::<u64>().ok());
    let start = (range.next()? + offset) as f64;
    let end = (range.next()? + offset) as f64;

    Some(AxisRange::Global(start..=end))
}

#[derive(Debug, Clone, PartialEq)]
pub enum AxisRange {
    Global(std::ops::RangeInclusive<f64>),
    Seq {
        seq_id: SeqId,
        range: std::ops::Range<u64>,
    },
}

impl AxisRange {
    pub fn to_global(self) -> Option<std::ops::RangeInclusive<f64>> {
        match self {
            AxisRange::Global(r) => Some(r),
            AxisRange::Seq { .. } => None,
        }
    }

    pub fn seq(seq_id: SeqId, range: std::ops::Range<u64>) -> Self {
        AxisRange::Seq { seq_id, range }
    }

    pub fn from_string_with_names(seq_names: &BiMap<String, SeqId>, text: &str) -> Option<Self> {
        if let Some((seq_name, range)) = text.rsplit_once(':') {
            let (from, to) = range.split_once('-')?;
            let from = from.parse::<u64>().ok()?;
            let to = to.parse::<u64>().ok()?;
            let seq_id = *seq_names.get_by_left(seq_name)?;
            Some(AxisRange::Seq {
                seq_id,
                range: from..to,
            })
        } else {
            let (from, to) = text.split_once('-')?;
            let from = from.parse::<f64>().ok()?;
            let to = to.parse::<f64>().ok()?;
            Some(AxisRange::Global(from..=to))
        }
    }

    pub fn to_string_with_names(&self, seq_names: &BiMap<String, SeqId>) -> String {
        match self {
            AxisRange::Global(range) => format!("{:.2}-{:.2}", range.start(), range.end()),
            AxisRange::Seq { seq_id, range } => {
                let seq_name = seq_names
                    .get_by_right(seq_id)
                    .map(|s| s.as_str())
                    .unwrap_or("<ERROR>");
                format!("{seq_name}:{}-{}", range.start, range.end)
            }
        }
    }
}

impl From<std::ops::RangeInclusive<f64>> for AxisRange {
    fn from(value: std::ops::RangeInclusive<f64>) -> Self {
        AxisRange::Global(value)
    }
}

#[derive(Debug, Clone)]
pub struct GridAxis {
    /// Maps global sequence indices to the indices in `seq_order`,
    seq_index_map: FxHashMap<SeqId, usize>,

    /// The IDs of the sequences in the axis
    seq_order: Vec<SeqId>,
    seq_offsets: Vec<u64>,
    seq_lens: Vec<u64>,
    pub total_len: u64,
}

impl GridAxis {
    pub fn tile_count(&self) -> usize {
        self.seq_order.len()
    }

    pub fn axis_range_into_global(
        &self,
        axis_range: &AxisRange,
    ) -> Option<std::ops::RangeInclusive<f64>> {
        match axis_range {
            AxisRange::Global(range) => Some(range.clone()),
            AxisRange::Seq { seq_id, range } => {
                let offset = self.sequence_offset(*seq_id)?;
                let start = (offset + range.start) as f64;
                let end = (offset + range.end) as f64;
                Some(start..=end)
            }
        }
    }

    pub fn tiles_covered_by_range(
        &self,
        range: std::ops::RangeInclusive<f64>,
    ) -> Option<impl Iterator<Item = SeqId> + '_> {
        if *range.start() > self.total_len as f64 || *range.end() < 0.0 {
            return None;
        }

        let start = range.start().floor() as u64;
        let end = range.end().ceil() as u64;

        let mut start_i = self.seq_offsets.partition_point(|&p| p < start);
        let end_i = self.seq_offsets.partition_point(|&p| p < end);

        if start_i == self.seq_offsets.len() {
            return None;
        }

        if self.seq_offsets[start_i] > start {
            start_i -= 1;
        }

        Some(self.seq_order[start_i..end_i].iter().copied())
    }

    pub fn from_sequences<'a>(
        sequence_names: &Arc<BiMap<String, SeqId>>,
        sequences: impl IntoIterator<Item = (&'a str, u64)>,
    ) -> Self {
        let iter = sequences.into_iter().filter_map(|(name, len)| {
            let seq_id = *sequence_names.get_by_left(name)?;
            Some((seq_id, len))
        });

        Self::from_index_and_lengths(iter)
    }

    pub fn from_index_and_lengths(items: impl IntoIterator<Item = (SeqId, u64)>) -> Self {
        let mut seq_indices = FxHashMap::default();

        let mut seq_order = Vec::new();
        let mut seq_offsets = Vec::new();
        let mut seq_lens = Vec::new();

        let mut offset = 0u64;

        for (seq_id, seq_len) in items {
            seq_indices.insert(seq_id, seq_order.len());

            seq_order.push(seq_id);
            seq_offsets.push(offset);
            seq_lens.push(seq_len);

            offset += seq_len;
        }
        // push the last "marker" offset for later convenience
        seq_offsets.push(offset);

        Self {
            seq_index_map: seq_indices,
            seq_order,
            seq_offsets,
            seq_lens,
            total_len: offset,
        }
    }

    pub fn offsets(&self) -> impl Iterator<Item = u64> + '_ {
        self.seq_offsets.iter().copied().chain([self.total_len])
    }

    pub fn sequence_offset(&self, seq_id: SeqId) -> Option<u64> {
        let ix = self.seq_index_map.get(&seq_id)?;
        self.seq_offsets.get(*ix).copied()
    }

    pub fn sequence_axis_range(&self, seq_id: SeqId) -> Option<std::ops::Range<u64>> {
        let ix = *self.seq_index_map.get(&seq_id)?;
        let start = *self.seq_offsets.get(ix)?;

        let end = if ix == self.seq_offsets.len() {
            *self.seq_offsets.get(ix + 1)?
        } else {
            self.total_len
        };

        Some(start..end)
    }

    /// Maps a point in `0 <= t <= self.total_len` to a sequence ID and
    /// point in the sequence, normalized to [0, 1)
    pub fn global_to_axis_local(&self, t: f64) -> Option<(SeqId, f64)> {
        if t < 0.0 || t > self.total_len as f64 {
            return None;
        }

        // let (seq_id, pos) = self.global_to_axis_exact(t as u64)?;

        let i = self
            .seq_offsets
            .partition_point(|&v| (v as f64) <= t)
            .checked_sub(1) // 2024-05-29 NB: this is going below 0 w/ the physics calls
            .unwrap_or_default();
        let offset = self.seq_offsets[i] as f64;

        let v = (t - offset) / self.seq_lens[i] as f64;

        let seq_id = self.seq_order[i];

        Some((seq_id, v))
    }

    pub fn global_to_axis_exact(&self, t: u64) -> Option<(SeqId, u64)> {
        if t > self.total_len {
            return None;
        }

        let i = self
            .seq_offsets
            .partition_point(|&v| v <= t)
            .checked_sub(1)
            .unwrap();
        let offset = self.seq_offsets[i];
        let len = self.seq_lens[i];

        let seq_id = self.seq_order[i];

        let v = t.checked_sub(offset).unwrap();

        Some((seq_id, v))
    }

    pub fn axis_local_to_global_exact(&self, seq_id: SeqId, t: u64) -> Option<u64> {
        let ix = *self.seq_index_map.get(&seq_id)?;
        let offset = self.seq_offsets.get(ix)?;
        let len = self.seq_lens.get(ix)?;
        if t > offset + len {
            // maybe allow this? not sure if it matters
            return None;
        }
        Some(offset + t)
    }

    /// Maps a point in [0, 1] inside a grid "row" to a point in the global grid offset
    pub fn axis_local_to_global(&self, seq_id: SeqId, t: f64) -> Option<f64> {
        if t < 0.0 || t > 1.0 {
            return None;
        }
        let ix = *self.seq_index_map.get(&seq_id)?;

        let offset = self.seq_offsets[ix] as f64;
        let v = self.seq_lens[ix] as f64 * t;
        Some(offset + v)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::approx_eq;
    use proptest::prelude::*;

    fn test_axis_short() -> GridAxis {
        GridAxis::from_index_and_lengths((0..4).map(|i| (SeqId(i), 1000)))
    }

    fn test_axis() -> GridAxis {
        GridAxis::from_index_and_lengths((0usize..10).map(|i| (SeqId(i), (1 + i as u64) * 1000)))
    }

    #[test]
    fn grid_axis_edgecases() {
        let axis = test_axis();

        assert_eq!(Some(0.0), axis.axis_local_to_global(SeqId(0), 0.0));

        assert_eq!(
            Some(axis.seq_offsets[1] as f64),
            axis.axis_local_to_global(SeqId(1), 0.0)
        );

        assert_eq!(
            Some(axis.total_len as f64),
            axis.axis_local_to_global(SeqId(9), 1.0)
        );

        assert_eq!(
            Some((axis.total_len - axis.seq_lens[9]) as f64),
            axis.axis_local_to_global(SeqId(9), 0.0)
        );
    }

    #[test]
    fn grid_axis_map_isomorphic() {
        let axis = test_axis();

        proptest!(|(seq_id in 0usize..10, t in 0f64..=1.0)| {
            let seq_id = SeqId(seq_id);
            let global = axis.axis_local_to_global(seq_id, t).unwrap();
            let (seq_id_, t_) = axis.global_to_axis_local(global).unwrap();
            prop_assert_eq!(seq_id, seq_id_);

            let eps = std::f32::EPSILON as f64;
            prop_assert!(approx_eq!(f64, t, t_, epsilon = eps));
        });
    }

    #[test]
    fn grid_axis_get_tiles_covered_by_range() {
        let axis = test_axis_short();

        let get_tiles = |range| {
            axis.tiles_covered_by_range(range)
                .map(|t| t.collect::<Vec<_>>())
        };

        let test_vec = |input: Vec<usize>| -> Option<Vec<SeqId>> {
            Some(input.into_iter().map(SeqId).collect())
        };

        let cov_all = get_tiles(0f64..=axis.total_len as f64);
        assert_eq!(cov_all, test_vec(vec![0, 1, 2, 3]));

        let cov_0 = get_tiles(0f64..=999.0);
        let cov_1 = get_tiles(1000f64..=1999.0);
        assert_eq!(cov_0, test_vec(vec![0]));
        assert_eq!(cov_1, test_vec(vec![1]));

        let cov_01 = get_tiles(0f64..=1999.0);
        let cov_01_half = get_tiles(500f64..=1499.0);
        assert_eq!(cov_01, test_vec(vec![0, 1]));
        assert_eq!(cov_01_half, test_vec(vec![0, 1]));

        let cov_part_0 = get_tiles(200f64..=500.0);
        assert_eq!(cov_part_0, test_vec(vec![0]));

        let cov_part_last = get_tiles(3300f64..=3700f64);
        assert_eq!(cov_part_last, test_vec(vec![3]));

        let cov_23_half = get_tiles(2500f64..=3499.0);
        assert_eq!(cov_23_half, test_vec(vec![2, 3]));

        let cov_123_half = get_tiles(1500f64..=3499.0);
        assert_eq!(cov_123_half, test_vec(vec![1, 2, 3]));
    }
}
