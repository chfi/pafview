use ultraviolet::UVec2;

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
    data: &'a crate::ProcessedCigar,
    index: usize,

    target_range: std::ops::Range<u64>,
}

impl<'a> MatchOpIter<'a> {
    fn from_range(
        match_data: &'a crate::ProcessedCigar,
        target_range: std::ops::Range<u64>,
        query_range: std::ops::Range<u64>,
    ) -> Self {
        //
        todo!();
    }
}

impl<'a> Iterator for MatchOpIter<'a> {
    // outputs each individual match/mismatch op's position along the
    // target and query
    type Item = [u64; 2];

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_match_op_iter() {
        todo!();
    }
}
