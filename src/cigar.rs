use bimap::BiMap;
use ultraviolet::DVec2;

use anyhow::anyhow;

// TODO needs a thorough cleaning
pub struct ProcessedCigar {
    pub target_id: usize,
    #[deprecated]
    pub target_offset: u64,
    pub target_len: u64,

    pub query_id: usize,
    #[deprecated]
    pub query_offset: u64,
    pub query_len: u64,

    pub strand_rev: bool,

    pub match_edges: Vec<[DVec2; 2]>,
    pub match_offsets: Vec<[u64; 2]>,
    pub match_lens: Vec<u64>,
    // TODO these flags should be in a struct & single vec
    pub match_is_match: Vec<bool>,
    // pub match_is_rev: Vec<bool>,
    pub match_cigar_index: Vec<usize>,

    pub aabb_min: DVec2,
    pub aabb_max: DVec2,

    pub cigar: Vec<(CigarOp, u64)>,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Strand {
    #[default]
    Forward,
    Reverse,
}

impl std::str::FromStr for Strand {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(Self::Forward),
            "-" => Ok(Self::Reverse),
            _ => Err("Strand must be + or -"),
        }
    }
}

pub struct CigarIterItem {
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,

    // target_offset: u64,
    // query_offset: u64,
    query_strand: Strand,
    op: CigarOp,
    op_count: u64,
}

impl CigarIterItem {
    //
}

pub struct CigarIter<'a, 'seq> {
    cigar: &'a CigarIndex,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,

    target_seq: Option<&'seq [u8]>,
    query_seq: Option<&'seq [u8]>,
}

impl<'a, 'seq> CigarIter<'a, 'seq> {
    fn new(cigar: &'a CigarIndex, target_range: std::ops::Range<u64>) -> Self {
        let start_i = cigar
            .op_target_offsets
            .partition_point(|&t| t < target_range.start);
        let start_i = start_i.checked_sub(1).unwrap_or_default();

        let end_i = cigar
            .op_target_offsets
            .partition_point(|&t| t < target_range.end);

        let t_start = cigar.op_target_offsets[start_i];
        let t_end = cigar.op_target_offsets[end_i];

        let q_start = cigar.op_query_offsets[start_i];
        let q_end = cigar.op_query_offsets[end_i];
        let query_range = q_start..q_end; //... hmmmm

        //

        // let [t_start, q_start] = cigar.ops.match_offsets[i_start];
        // let [t_end, q_end] = cigar.ops.match_offsets[i_end];

        let op_index_range = start_i..end_i;

        Self {
            cigar,

            op_index_range,
            target_range,
            query_range,

            target_seq: None,
            query_seq: None,
        }
    }

    // fn new_with_seqs(
    //     target_seq: &'seq [u8],
    //     query_seq: &'seq [u8],
    //     cigar: &'a ProcessedCigar,
    //     target_range: std::ops::Range<u64>,
    // ) -> Self {
    //     let mut result = Self::new(cigar, target_range);
    //     result.target_seq = Some(target_seq);
    //     result.query_seq = Some(query_seq);
    //     result
    // }
}

impl<'a, 'seq> Iterator for CigarIter<'a, 'seq> {
    type Item = CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.op_index_range.next()?;

        // get the target range for the current op
        let target_offset = self.cigar.op_target_offsets[index];

        // the query range -- which must be inverted if strand is reverse

        // let query_offset_fwd = self.

        let query_strand = self.cigar.query_strand;

        let item = CigarIterItem {
            target_range: todo!(),
            query_range: todo!(),
            query_strand,
            op: todo!(),
            op_count: todo!(),
        };

        todo!()
    }
}

impl<'a, 'seq> DoubleEndedIterator for CigarIter<'a, 'seq> {
    fn next_back(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CigarOp {
    M,
    X,
    Eq,
    D,
    I,
    S,
    H,
    N,
}

impl CigarOp {
    pub fn parse_str_into_vec(cigar: &str) -> Vec<(CigarOp, u64)> {
        cigar
            .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
            .filter_map(|opstr| {
                let count = opstr[..opstr.len() - 1].parse::<u64>().ok()?;
                let op = opstr.as_bytes()[opstr.len() - 1] as char;
                let op = CigarOp::try_from(op).ok()?;
                Some((op, count))
            })
            .collect::<Vec<_>>()
    }

    pub fn apply_to_offsets(&self, count: u64, offsets: [u64; 2]) -> [u64; 2] {
        use CigarOp as Cg;
        let [tgt, qry] = offsets;
        match self {
            Cg::M | Cg::X | Cg::Eq => [tgt + count, qry + count],
            Cg::D | Cg::N => [tgt + count, qry],
            Cg::I => [tgt, qry + count],
            Cg::S | Cg::H => offsets,
        }
    }

    pub fn is_match(&self) -> bool {
        match self {
            CigarOp::M | CigarOp::Eq => true,
            _ => false,
        }
    }

    pub fn is_mismatch(&self) -> bool {
        *self == CigarOp::X
    }

    pub fn is_match_or_mismatch(&self) -> bool {
        match self {
            CigarOp::M | CigarOp::Eq | CigarOp::X => true,
            _ => false,
        }
    }
}

impl From<CigarOp> for char {
    fn from(value: CigarOp) -> Self {
        match value {
            CigarOp::M => 'M',
            CigarOp::X => 'X',
            CigarOp::Eq => '=',
            CigarOp::D => 'D',
            CigarOp::I => 'I',
            CigarOp::S => 'S',
            CigarOp::H => 'H',
            CigarOp::N => 'N',
        }
    }
}

impl TryFrom<char> for CigarOp {
    type Error = &'static str;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'M' => Ok(Self::M),
            'X' => Ok(Self::X),
            '=' => Ok(Self::Eq),
            'D' => Ok(Self::D),
            'I' => Ok(Self::I),
            'S' => Ok(Self::S),
            'H' => Ok(Self::H),
            'N' => Ok(Self::N),
            _ => Err("Unknown op"),
        }
    }
}

pub struct CigarIndex {
    pub target_seq_id: usize,
    pub query_seq_id: usize,

    pub target_len: u64,
    pub query_len: u64,

    pub query_strand: Strand,
    pub ops: Vec<(CigarOp, u64)>,

    pub op_line_vertices: Vec<[DVec2; 2]>,
    pub op_target_offsets: Vec<u64>,
    pub op_query_offsets: Vec<u64>,
}

impl CigarIndex {
    pub fn op_target_range(&self, op_ix: usize) -> Option<std::ops::Range<u64>> {
        if op_ix + 1 >= self.op_target_offsets.len() {
            return None;
        }
        let start = *self.op_target_offsets.get(op_ix)?;
        let end = *self.op_target_offsets.get(op_ix + 1)?;
        Some(start..end)
    }

    fn op_query_range_fwd(&self, op_ix: usize) -> Option<std::ops::Range<u64>> {
        if op_ix + 1 >= self.op_query_offsets.len() {
            return None;
        }
        let start = *self.op_query_offsets.get(op_ix)?;
        let end = *self.op_query_offsets.get(op_ix + 1)?;
        Some(start..end)
    }

    pub fn op_query_range(&self, op_ix: usize) -> Option<std::ops::Range<u64>> {
        let fwd_range = self.op_query_range_fwd(op_ix)?;

        let start = self.query_len - fwd_range.start;
        let end = self.query_len - fwd_range.end;
        Some(start..end)
    }

    pub fn from_paf_line(
        paf_line: &crate::PafLine<&str>,
        target_seq_id: usize,
        query_seq_id: usize,
    ) -> Self {
        use CigarOp as Cg;

        let cigar = CigarOp::parse_str_into_vec(&paf_line.cigar);

        let query_strand = if paf_line.strand_rev {
            Strand::Reverse
        } else {
            Strand::Forward
        };

        let mut op_line_vertices = Vec::new();
        let mut op_target_offsets = Vec::new();
        let mut op_query_offsets = Vec::new();

        let mut target_offset = 0u64;

        // query offsets are always stored as 0-based & increasing,
        // even when reverse
        let mut query_offset = 0u64;

        // let mut query_offset = match query_strand {
        //     Strand::Forward => 0u64,
        //     Strand::Reverse =>
        // };

        for (op, count) in cigar.iter() {
            op_target_offsets.push(target_offset);
            op_query_offsets.push(query_offset);

            match op {
                Cg::M | Cg::X | Cg::Eq => {
                    // output match line for high-scale view
                    todo!();

                    // increment target & query
                    target_offset += count;
                    query_offset += count;
                }
                Cg::D => {
                    // increment target
                    target_offset += count;
                }
                Cg::I => {
                    // increment query
                    query_offset += count;
                }
                _ => (),
            }
        }

        //
        op_target_offsets.push(target_offset);
        op_query_offsets.push(query_offset);

        Self {
            target_seq_id,
            query_seq_id,
            query_strand,
            ops: cigar,
            op_line_vertices,
            op_target_offsets,
            op_query_offsets,
            target_len: target_offset,
            query_len: query_offset,
        }
    }
    //
}

impl ProcessedCigar {
    pub fn from_line_local(
        seq_names: &BiMap<String, usize>,
        paf_line: &crate::PafLine<&str>,
    ) -> anyhow::Result<Self> {
        Self::from_line(seq_names, paf_line, [0, 0])
    }

    pub fn from_line(
        seq_names: &BiMap<String, usize>,
        paf_line: &crate::PafLine<&str>,
        origin: [u64; 2],
    ) -> anyhow::Result<Self> {
        let ops = CigarOp::parse_str_into_vec(&paf_line.cigar);

        let strand_rev = paf_line.strand_rev;

        let [mut target_pos, mut query_pos] = origin;

        if strand_rev {
            target_pos = paf_line.tgt_seq_len - 1;
        }

        let target_id = *seq_names
            .get_by_left(paf_line.tgt_name)
            .ok_or_else(|| anyhow!("Target sequence `{}` not found", paf_line.tgt_name))?;
        let query_id = *seq_names
            .get_by_left(paf_line.query_name)
            .ok_or_else(|| anyhow!("Query sequence `{}` not found", paf_line.query_name))?;

        let mut match_edges = Vec::new();
        let mut match_offsets = Vec::new();
        let mut match_lens = Vec::new();
        let mut match_is_match = Vec::new();
        // let mut match_is_rev = Vec::new();

        let mut match_cigar_index = Vec::new();

        let mut aabb_min = DVec2::broadcast(std::f64::MAX);
        let mut aabb_max = DVec2::broadcast(std::f64::MIN);

        for (cg_ix, &(op, count)) in ops.iter().enumerate() {
            match char::from(op) {
                'M' | '=' | 'X' => {
                    let x = target_pos;
                    let y = query_pos;

                    {
                        let x0 = x as f64;
                        let y0 = y as f64;

                        let x_end = if paf_line.strand_rev {
                            x.checked_sub(count).unwrap_or_default()
                        } else {
                            x + count
                        };
                        let x1 = x_end as f64;
                        let y1 = (y + count) as f64;

                        let p0 = DVec2::new(x0, y0);
                        let p1 = DVec2::new(x1, y1);

                        aabb_min = aabb_min.min_by_component(p0).min_by_component(p1);
                        aabb_max = aabb_max.max_by_component(p0).max_by_component(p1);

                        match_edges.push([p0, p1]);
                        match_offsets.push([target_pos, query_pos]);
                        match_lens.push(count);

                        match_is_match.push(op.is_match());
                        // match_is_rev.push(paf_line.strand_rev);
                        match_cigar_index.push(cg_ix);
                    }

                    target_pos += count;
                    if paf_line.strand_rev {
                        query_pos = query_pos.checked_sub(count).unwrap_or_default()
                    } else {
                        query_pos += count;
                    }
                }
                'D' => {
                    target_pos += count;
                }
                'I' => {
                    if paf_line.strand_rev {
                        query_pos = query_pos.checked_sub(count).unwrap_or_default()
                    } else {
                        query_pos += count;
                    }
                }
                _ => (),
            }
        }

        let target_len = target_pos - origin[0];
        let query_len = query_pos - origin[1];

        Ok(Self {
            target_id,
            target_offset: origin[0],
            target_len,

            query_id,
            query_offset: origin[1],
            query_len,

            strand_rev,

            match_edges,
            match_offsets,
            match_lens,
            match_is_match,

            match_cigar_index,
            cigar: ops,

            aabb_min,
            aabb_max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cigar_raster_iter() {
        todo!();
    }
}
