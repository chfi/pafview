use bimap::BiMap;

use crate::PafInput;

/// An `AlignmentGrid` defines the global position of the aligned sequence pairs
pub struct AlignmentGrid {
    x_axis: GridAxis,
    y_axis: GridAxis,
}

#[derive(Clone)]
struct GridAxis {
    /// The IDs of the sequences in the axis
    seq_order: Vec<usize>,
    seq_offsets: Vec<u64>,
    seq_lens: Vec<u64>,
    total_len: u64,
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
        let mut seq_order = Vec::new();
        let mut seq_offsets = Vec::new();
        let mut seq_lens = Vec::new();

        let mut offset = 0u64;

        for (seq_id, seq_len) in items {
            seq_order.push(seq_id);
            seq_offsets.push(offset);
            seq_lens.push(seq_len);

            offset += seq_len;
        }

        Self {
            seq_order,
            seq_offsets,
            seq_lens,
            total_len: offset,
        }
    }

    /// Maps a point in `0 <= t <= self.total_len` to a sequence ID and
    /// point in the sequence, normalized to [0, 1]
    pub fn global_to_grid(&self, t: f64) -> Option<(usize, f64)> {
        let i = self.seq_offsets.partition_point(|&v| (v as f64) < t);

        todo!();
    }
}

#[cfg(test)]
mod tests {

    use super::*;
}
