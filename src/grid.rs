use std::sync::Arc;

use bimap::BiMap;
use rustc_hash::FxHashMap;

use crate::PafInput;

/// An `AlignmentGrid` defines the global position of the aligned sequence pairs
#[derive(Debug)]
pub struct AlignmentGrid {
    pub x_axis: GridAxis,
    pub y_axis: GridAxis,

    pub sequence_names: Arc<BiMap<String, usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AxisRange {
    Global(std::ops::RangeInclusive<f64>),
    Seq {
        seq_id: usize,
        range: std::ops::Range<u64>,
    },
}

impl AxisRange {
    pub fn seq(seq_id: usize, range: std::ops::Range<u64>) -> Self {
        AxisRange::Seq { seq_id, range }
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
    seq_index_map: FxHashMap<usize, usize>,

    /// The IDs of the sequences in the axis
    seq_order: Vec<usize>,
    seq_offsets: Vec<u64>,
    seq_lens: Vec<u64>,
    pub total_len: u64,
}

impl GridAxis {
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
    ) -> Option<impl Iterator<Item = usize> + '_> {
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
        sequence_names: &Arc<BiMap<String, usize>>,
        sequences: impl IntoIterator<Item = &'a crate::AlignedSeq>,
    ) -> Self {
        let iter = sequences.into_iter().filter_map(|seq| {
            let seq_id = *sequence_names.get_by_left(&seq.name)?;
            Some((seq_id, seq.len))
        });

        Self::from_index_and_lengths(iter)
    }

    pub fn from_index_and_lengths(items: impl IntoIterator<Item = (usize, u64)>) -> Self {
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

    pub fn sequence_offset(&self, seq_id: usize) -> Option<u64> {
        let ix = self.seq_index_map.get(&seq_id)?;
        self.seq_offsets.get(*ix).copied()
    }

    pub fn sequence_axis_range(&self, seq_id: usize) -> Option<std::ops::Range<u64>> {
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
    pub fn global_to_axis_local(&self, t: f64) -> Option<(usize, f64)> {
        if t < 0.0 || t > self.total_len as f64 {
            return None;
        }

        // let (seq_id, pos) = self.global_to_axis_exact(t as u64)?;

        let i = self
            .seq_offsets
            .partition_point(|&v| (v as f64) <= t)
            .checked_sub(1)
            .unwrap();
        let offset = self.seq_offsets[i] as f64;

        let v = (t - offset) / self.seq_lens[i] as f64;

        let seq_id = self.seq_order[i];

        Some((seq_id, v))
    }

    pub fn global_to_axis_exact(&self, t: u64) -> Option<(usize, u64)> {
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

    /// Maps a point in [0, 1] inside a grid "row" to a point in the global grid offset
    pub fn axis_local_to_global(&self, seq_id: usize, t: f64) -> Option<f64> {
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
        GridAxis::from_index_and_lengths((0..4).map(|i| (i, 1000)))
    }

    fn test_axis() -> GridAxis {
        GridAxis::from_index_and_lengths((0usize..10).map(|i| (i, (1 + i as u64) * 1000)))
    }

    #[test]
    fn grid_axis_edgecases() {
        let axis = test_axis();

        assert_eq!(Some(0.0), axis.axis_local_to_global(0, 0.0));

        assert_eq!(
            Some(axis.seq_offsets[1] as f64),
            axis.axis_local_to_global(1, 0.0)
        );

        assert_eq!(
            Some(axis.total_len as f64),
            axis.axis_local_to_global(9, 1.0)
        );

        assert_eq!(
            Some((axis.total_len - axis.seq_lens[9]) as f64),
            axis.axis_local_to_global(9, 0.0)
        );
    }

    #[test]
    fn grid_axis_map_isomorphic() {
        let axis = test_axis();

        proptest!(|(seq_id in 0usize..10, t in 0f64..=1.0)| {
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

        let cov_all = get_tiles(0f64..=axis.total_len as f64);
        assert_eq!(cov_all, Some(vec![0, 1, 2, 3]));

        let cov_0 = get_tiles(0f64..=999.0);
        let cov_1 = get_tiles(1000f64..=1999.0);
        assert_eq!(cov_0, Some(vec![0]));
        assert_eq!(cov_1, Some(vec![1]));

        let cov_01 = get_tiles(0f64..=1999.0);
        let cov_01_half = get_tiles(500f64..=1499.0);
        assert_eq!(cov_01, Some(vec![0, 1]));
        assert_eq!(cov_01_half, Some(vec![0, 1]));

        let cov_part_0 = get_tiles(200f64..=500.0);
        assert_eq!(cov_part_0, Some(vec![0]));

        let cov_part_last = get_tiles(3300f64..=3700f64);
        assert_eq!(cov_part_last, Some(vec![3]));

        let cov_23_half = get_tiles(2500f64..=3499.0);
        assert_eq!(cov_23_half, Some(vec![2, 3]));

        let cov_123_half = get_tiles(1500f64..=3499.0);
        assert_eq!(cov_123_half, Some(vec![1, 2, 3]));
    }
}
