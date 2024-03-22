use bimap::BiMap;
use rustc_hash::FxHashMap;

use crate::PafInput;

/// An `AlignmentGrid` defines the global position of the aligned sequence pairs
#[derive(Debug)]
pub struct AlignmentGrid {
    pub x_axis: GridAxis,
    pub y_axis: GridAxis,
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
    pub fn from_sequences<'a>(
        sequence_names: &BiMap<String, usize>,
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
}
