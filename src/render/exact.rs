use ultraviolet::UVec2;

use crate::cigar::{CigarOp, ProcessedCigar};

/// pixel/bp-perfect CPU rasterization

pub fn draw_subsection(
    match_data: &crate::ProcessedCigar,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas_size: UVec2,
    canvas_data: &mut Vec<egui::Color32>,
) {
    let size = (canvas_size.x * canvas_size.y) as usize;
    canvas_data.clear();
    canvas_data.resize(size, egui::Color32::TRANSPARENT);

    // indices of matches covered by the target range
    let match_subrange = {
        let t_start = match_data
            .match_offsets
            .partition_point(|[t, _]| *t <= target_range.start);
        let t_end =
            match_data.match_offsets[t_start..].partition_point(|[t, _]| *t <= target_range.end);
        // let q_start = match_data
        //     .match_offsets
        //     .partition_point(|[_, q]| *q <= query_range.start);

        // let q_end =
        //     match_data.match_offsets[q_start..].partition_point(|[t, _]| *t <= query_range.end);

        t_start..t_end
    };

    for x in 0..canvas_size.x {
        // for x == 0, we want to look at target_range.start;
        // when x == canvas_size.x - 1, at target_range.end;

        // -- really what we want is to find the correct matches (via
        // binary search on the offsets buffer) & then step through or
        // sample from that

        // let

        // for y in 0..canvas_size.y {
        // }
    }

    todo!();
}

struct MatchOpIter<'a> {
    match_offsets: &'a [[u64; 2]],
    match_cg_ix: &'a [usize],
    cigar: &'a [(CigarOp, u64)],

    target_range: std::ops::Range<u64>,

    index: usize,
    // current_match: Option<(usize, [u64; 2], bool)>,
    current_match: Option<(std::ops::Range<u64>, [u64; 2], bool, bool)>,
}

impl<'a> MatchOpIter<'a> {
    fn from_range(
        match_offsets: &'a [[u64; 2]],
        match_cg_ix: &'a [usize],
        cigar: &'a [(CigarOp, u64)],
        target_range: std::ops::Range<u64>,
    ) -> Self {
        let t_start = match_offsets.partition_point(|[t, _]| *t <= target_range.start);

        Self {
            // data: match_data,
            match_offsets,
            match_cg_ix,
            cigar,

            target_range,

            index: t_start,
            current_match: None,
        }
    }
}

impl<'a> Iterator for MatchOpIter<'a> {
    // outputs each individual match/mismatch op's position along the
    // target and query
    type Item = ([u64; 2], bool);

    fn next(&mut self) -> Option<Self::Item> {
        todo!();
    }
}

/*
impl<'a> Iterator for MatchOpIter<'a> {
    // outputs each individual match/mismatch op's position along the
    // target and query
    type Item = ([u64; 2], bool);

    // doesn't take strand into account yet
    fn next(&mut self) -> Option<Self::Item> {
        if self.target_range.is_empty() {
            return None;
        }

        if self.current_match.is_none() {
            let ix = self.index;
            self.index += 1;

            let origin = self.match_offsets[ix];
            let len = self.match_lens[ix];

            let bp_range = 0..len;
            let is_match = self.match_is_match[ix];
            let is_rev = self.match_is_rev[ix];

            self.current_match = Some((bp_range, origin, is_match, is_rev));
        }

        let mut current_done = false;

        // if let Some((bp_range, origin, is_match, is_rev)) = self.current_match.as_mut() {
        if let Some((mut bp_range, origin @ [tgt, qry], is_match, is_rev)) =
            self.current_match.take()
        {
            let next_offset = bp_range.next()?;

            // let tgt = tgt as isize;

            // let [tgt, qry] = origin;
            // let output = [tgt + next_offset, qry + next_offset]

            if !bp_range.is_empty() {
                self.current_match = Some((bp_range, origin, is_match, is_rev));
            }
        }

        /*
        if self.current_match.is_none() {
            self.index += 1;
            let is_match = self.match_is_match[self.index];
            let offsets = self.match_offsets[self.index];
            self.current_match = Some((self.index, offsets, is_match));
        }

        if let Some((ix, [t_off, q_off], is_match)) = self.current_match.as_mut() {
            let out_offsets = [*t_off, *q_off];
            *t_off += 1;
            *q_off += 1;
            let _ = self.target_range.next();
            return Some((out_offsets, *is_match));
        } else {
            return None;
        }
        */
    }
}
*/

#[cfg(test)]
mod tests {

    use ultraviolet::DVec2;

    use super::*;
    use crate::ProcessedCigar;

    #[test]
    fn test_match_op_iter() {
        todo!();
    }
}
