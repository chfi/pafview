use bimap::BiMap;
use ultraviolet::DVec2;

use anyhow::anyhow;

use crate::sequences::SeqId;

// TODO needs a thorough cleaning
pub struct ProcessedCigar {
    pub target_id: SeqId,
    #[deprecated]
    pub target_offset: u64,
    pub target_len: u64,

    pub query_id: SeqId,
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

pub struct CigarIter<'a> {
    cigar: &'a CigarIndex,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    // target_seq: Option<&'seq [u8]>,
    // query_seq: Option<&'seq [u8]>,
}

impl<'a> CigarIter<'a> {
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
            // target_seq: None,
            // query_seq: None,
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

impl<'a> Iterator for CigarIter<'a> {
    type Item = CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.op_index_range.next()?;

        let (op, op_count) = *self.cigar.ops.get(index)?;

        // get the target range for the current op
        let target_range = self.cigar.op_target_range(index)?;
        let query_range = self.cigar.op_query_range(index)?;

        let query_strand = self.cigar.query_strand;

        let item = CigarIterItem {
            target_range,
            query_range,
            query_strand,
            op,
            op_count,
        };

        Some(item)
    }
}

// impl<'a, 'seq> DoubleEndedIterator for CigarIter<'a, 'seq> {
//     fn next_back(&mut self) -> Option<Self::Item> {
//         todo!()
//     }
// }

// Packed representation of an entire cigar
pub struct Cigar(Vec<u32>);

impl Cigar {
    pub fn parse_str(cigar: &str) -> Self {
        let ops = cigar
            .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
            .filter_map(|opstr| {
                let count = opstr[..opstr.len() - 1].parse::<u32>().ok()?;
                let op = opstr.as_bytes()[opstr.len() - 1] as char;
                let op = CigarOp::try_from(op).ok()?;
                Some(op.pack(count))
            })
            .collect::<Vec<_>>();

        Cigar(ops)
    }

    pub fn iter(&self) -> impl Iterator<Item = (CigarOp, u32)> + '_ {
        self.0.iter().copied().map(CigarOp::unpack)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CigarOp {
    Eq = 0,
    X = 1,
    I = 2,
    D = 3,
    M = 4,
    // S = 5,
    // H = 6,
    // N = 7,
}

impl CigarOp {
    pub fn pack(&self, count: u32) -> u32 {
        let op = (*self as u32) << 29;
        op | count
    }

    pub fn unpack(val: u32) -> (CigarOp, u32) {
        use CigarOp as Cg;
        let op = (val >> 29).min(4); // just to be safe (default to M)
        let count = val & !(0x11 << 29);

        let op = match op {
            0 => Cg::Eq,
            1 => Cg::X,
            2 => Cg::I,
            3 => Cg::D,
            _ => Cg::M,
        };
        (op, count)
    }

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
            Cg::D /* | Cg::N */ => [tgt + count, qry],
            Cg::I => [tgt, qry + count],
            // Cg::S | Cg::H => offsets,
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
            // CigarOp::S => 'S',
            // CigarOp::H => 'H',
            // CigarOp::N => 'N',
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
            // 'S' => Ok(Self::S),
            // 'H' => Ok(Self::H),
            // 'N' => Ok(Self::N),
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

    pub fn from_cigar(
        cigar: &[(CigarOp, u64)],
        target_seq_id: usize,
        query_seq_id: usize,
        target_len: u64,
        query_len: u64,
        query_strand: Strand,
    ) -> Self {
        use CigarOp as Cg;

        let cigar = cigar.to_vec();

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

        for &(op, count) in cigar.iter() {
            op_target_offsets.push(target_offset);
            op_query_offsets.push(query_offset);

            match op {
                Cg::M | Cg::X | Cg::Eq => {
                    // output match line for high-scale view
                    let x0 = target_offset as f64;
                    let x1 = x0 + count as f64;

                    let [y0, y1] = match query_strand {
                        Strand::Forward => {
                            let y0 = query_offset as f64;
                            let y1 = y0 + count as f64;
                            [y0, y1]
                        }
                        Strand::Reverse => {
                            let y0 = (query_len - query_offset) as f64;
                            let y1 = y0 + count as f64;
                            [y0, y1]
                        }
                    };

                    op_line_vertices.push([[x0, y0].into(), [x1, y1].into()]);

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

        debug_assert_eq!(target_offset, target_len);
        debug_assert_eq!(query_offset, query_len);
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

    pub fn from_cigar_string(
        cigar: &str,
        target_seq_id: usize,
        query_seq_id: usize,
        target_len: u64,
        query_len: u64,
        query_strand: Strand,
    ) -> Self {
        let cigar = CigarOp::parse_str_into_vec(cigar);

        Self::from_cigar(
            &cigar,
            target_seq_id,
            query_seq_id,
            target_len,
            query_len,
            query_strand,
        )
    }

    pub fn from_paf_line(
        paf_line: &crate::PafLine<&str>,
        target_seq_id: usize,
        query_seq_id: usize,
    ) -> Self {
        let query_strand = if paf_line.strand_rev {
            Strand::Reverse
        } else {
            Strand::Forward
        };

        Self::from_cigar_string(
            &paf_line.cigar,
            target_seq_id,
            query_seq_id,
            paf_line.tgt_seq_len,
            paf_line.query_seq_len,
            query_strand,
        )

        /*
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

        for &(op, count) in cigar.iter() {
            op_target_offsets.push(target_offset);
            op_query_offsets.push(query_offset);

            match op {
                Cg::M | Cg::X | Cg::Eq => {
                    // output match line for high-scale view
                    let x0 = target_offset as f64;
                    let x1 = x0 + count as f64;

                    let [y0, y1] = match query_strand {
                        Strand::Forward => {
                            let y0 = query_offset as f64;
                            let y1 = y0 + count as f64;
                            [y0, y1]
                        }
                        Strand::Reverse => {
                            let query_len = paf_line.query_seq_len;
                            let y0 = (query_len - query_offset) as f64;
                            let y1 = y0 + count as f64;
                            [y0, y1]
                        }
                    };

                    op_line_vertices.push([[x0, y0].into(), [x1, y1].into()]);

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
        */
    }
    //
}

impl ProcessedCigar {
    pub fn from_line_local(
        seq_names: &BiMap<String, SeqId>,
        paf_line: &crate::PafLine<&str>,
    ) -> anyhow::Result<Self> {
        Self::from_line(seq_names, paf_line, [0, 0])
    }

    pub fn from_line(
        seq_names: &BiMap<String, SeqId>,
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

    const TEST_PAF_RECORD: &'static str = "Y12#1#chrXIV	800685	30204	50196	+	S288C#1#chrXIV	800685	30000	50000	19922	20001	255	cg:Z:578=1X922=1X1135=1X334=1X194=1X653=1X90=1X32=1X715=1X41=1X29=1X92=1X368=1X504=1X140=1X14=1X18=1X233=1X703=4D6=1X603=1X844=1X86=1X562=1X1081=1X64=1X151=1X32=1X73=1X58=1X59=1X861=1X1216=1X1432=1X291=1X313=1X825=4D189=1X173=1X53=1X26=1X334=1X190=1X74=1X226=2X59=1X251=1X90=1X118=1X80=1X405=1X265=1X80=1X184=1X27=1X212=1X59=1X41=1X197=1X132=1X414=1X38=1X15=1X60=1X69=1X51=2X21=1D236=1X60=1I17=1X85=1X39=";

    const TEST_CIGAR: &'static str = "578=1X922=1X1135=1X334=1X194=1X653=1X90=1X32=1X715=1X41=1X29=1X92=1X368=1X504=1X140=1X14=1X18=1X233=1X703=4D6=1X603=1X844=1X86=1X562=1X1081=1X64=1X151=1X32=1X73=1X58=1X59=1X861=1X1216=1X1432=1X291=1X313=1X825=4D189=1X173=1X53=1X26=1X334=1X190=1X74=1X226=2X59=1X251=1X90=1X118=1X80=1X405=1X265=1X80=1X184=1X27=1X212=1X59=1X41=1X197=1X132=1X414=1X38=1X15=1X60=1X69=1X51=2X21=1D236=1X60=1I17=1X85=1X39=";

    #[test]
    fn cigar_raster_iter() {
        let paf_line = crate::parse_paf_line(TEST_PAF_RECORD.split("\t")).unwrap();
        // let cigar_ops = CigarOp::parse_str_into_vec(TEST_CIGAR);

        // let cg_index = CigarIndex::from_cigar(&cigar_ops, 0, 0,

        todo!();
    }
}
