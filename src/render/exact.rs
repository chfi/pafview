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

    // TODO doesn't take strand into account yet
    let match_iter = MatchOpIter::from_range(
        &match_data.match_offsets,
        &match_data.match_cigar_index,
        &match_data.cigar,
        target_range.clone(),
    );

    let tgt_len = target_range.end - target_range.start;
    let bp_width = canvas_size.x as f64 / tgt_len as f64;

    let qry_len = query_range.end - query_range.start;
    let bp_height = canvas_size.y as f64 / qry_len as f64;

    for ([target_pos, query_pos], is_match) in match_iter {
        // want to map target_pos to an x_range, query_pos to a y_range

        let color = if is_match {
            egui::Color32::BLACK
        } else {
            egui::Color32::RED
        };

        let x0 = target_pos as f64 * bp_width;
        let x1 = (1 + target_pos) as f64 * bp_width;

        let y0 = query_pos as f64 * bp_height;
        let y1 = (1 + query_pos) as f64 * bp_height;

        for x in (x0.floor() as usize)..(x1.floor() as usize) {
            for y in (y0.floor() as usize)..(y1.floor() as usize) {
                let ix = x + y * canvas_size.x as usize;
                canvas_data[ix] = color;
            }
        }
    }
}

struct MatchOpIter<'a> {
    match_offsets: &'a [[u64; 2]],
    match_cg_ix: &'a [usize],
    cigar: &'a [(CigarOp, u64)],

    target_range: std::ops::Range<u64>,

    index: usize,
    // current_match: Option<(usize, [u64; 2], bool)>,
    current_match: Option<(CigarOp, std::ops::Range<u64>, [u64; 2])>,
    // current_match: Option<(std::ops::Range<u64>, [u64; 2], bool, bool)>,
}

impl<'a> MatchOpIter<'a> {
    fn from_range(
        match_offsets: &'a [[u64; 2]],
        match_cg_ix: &'a [usize],
        cigar: &'a [(CigarOp, u64)],
        target_range: std::ops::Range<u64>,
    ) -> Self {
        let t_start = match_offsets.partition_point(|[t, _]| *t <= target_range.start);

        let t_start = t_start.checked_sub(1).unwrap_or_default();

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
        if self.target_range.is_empty() {
            return None;
        }

        if self.current_match.is_none() {
            if self.index >= self.match_offsets.len() {
                return None;
            }
            let ix = self.index;
            self.index += 1;
            let cg_ix = self.match_cg_ix[ix];
            let (op, count) = self.cigar[cg_ix];
            let range = 0..count;
            let origin = self.match_offsets[ix];
            self.current_match = Some((op, range, origin));
        }

        if let Some((op, mut range, origin @ [tgt, qry])) = self.current_match.take() {
            let next_offset = range.next()?;
            let _ = self.target_range.next();
            if !range.is_empty() {
                self.current_match = Some((op, range, origin));
            }

            let pos = [tgt + next_offset, qry + next_offset];
            let out = (pos, op.is_match());

            return Some(out);
        }

        None
    }
}

#[cfg(test)]
mod tests {

    use ultraviolet::DVec2;

    use super::*;
    use crate::ProcessedCigar;

    fn test_cigar() -> Vec<(CigarOp, u64)> {
        use crate::CigarOp as C;

        vec![
            (C::M, 10),
            (C::X, 1),
            (C::M, 13),
            (C::D, 7),
            (C::M, 13),
            (C::X, 1),
            (C::M, 21),
            (C::I, 13),
            (C::M, 18),
            (C::X, 5),
            (C::M, 39),
            (C::X, 1),
            (C::M, 3),
        ]
    }

    fn cigar_offsets(cg: &[(CigarOp, u64)]) -> (Vec<[u64; 2]>, Vec<usize>) {
        let mut offsets = Vec::new();
        let mut indices = Vec::new();

        let mut origin = [0u64, 0];

        for (cg_ix, &(op, count)) in cg.iter().enumerate() {
            if op.is_match_or_mismatch() {
                offsets.push(origin);
                indices.push(cg_ix);
            }
            origin = op.apply_to_offsets(count, origin);
        }

        (offsets, indices)
    }

    #[test]
    fn test_match_op_iter() {
        let cigar = test_cigar();
        let (offsets, cg_ix) = cigar_offsets(&cigar);

        let len = cigar.iter().map(|(_, c)| *c).sum::<u64>();

        let iter = MatchOpIter::from_range(&offsets, &cg_ix, &cigar, 0..len);

        for ([tgt, qry], is_match) in iter {
            println!("[{tgt:3}, {qry:3}] - {is_match}");
        }
    }
}
