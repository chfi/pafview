use bimap::BiMap;
use ultraviolet::DVec2;

use anyhow::anyhow;

use crate::sequences::SeqId;

pub mod implicit;

pub type BoxedCigarIter<'a> = Box<dyn Iterator<Item = CigarIterItem> + 'a>;

pub trait IndexedCigar {
    // fn iter(&self) -> Box<dyn Iterator<Item = CigarIterItem> + '_>;
    fn whole_cigar(&self) -> Box<dyn Iterator<Item = (CigarOp, u32)> + '_>;

    fn iter_target_range(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Box<dyn Iterator<Item = CigarIterItem> + '_>;
}

impl IndexedCigar for CigarIndex {
    // fn iter(&self) -> Box<dyn Iterator<Item = CigarIterItem> + '_> {
    fn whole_cigar(&self) -> Box<dyn Iterator<Item = (CigarOp, u32)> + '_> {
        let iter = self.cigar.iter();
        Box::new(iter)
    }

    fn iter_target_range(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Box<dyn Iterator<Item = CigarIterItem> + '_> {
        let iter = self.iter_target_range_impl(target_range);
        Box::new(iter)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Strand {
    #[default]
    Forward = 0,
    Reverse = 1,
}

impl Strand {
    pub fn is_rev(&self) -> bool {
        *self == Strand::Reverse
    }
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

#[derive(Debug)]
pub struct CigarIterItem {
    pub target_range: std::ops::Range<u64>,
    pub query_range: std::ops::Range<u64>,

    // target_offset: u64,
    // query_offset: u64,
    // pub query_strand: Strand,
    pub op: CigarOp,
    pub op_count: u32,
}

pub struct CigarIter<'a> {
    cigar: &'a CigarIndex,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
    // query_range: std::ops::Range<u64>,
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

        // let t_start = cigar.op_target_offsets[start_i];
        // let t_end = cigar.op_target_offsets[end_i];

        // let q_start = cigar.op_query_offsets[start_i];
        // let q_end = cigar.op_query_offsets[end_i];
        // let query_range = q_start..q_end; //... hmmmm

        //

        // let [t_start, q_start] = cigar.ops.match_offsets[i_start];
        // let [t_end, q_end] = cigar.ops.match_offsets[i_end];

        let op_index_range = start_i..end_i;

        Self {
            cigar,

            op_index_range,
            target_range,
            // query_range,
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

        let (op, mut op_count) = self.cigar.cigar.get(index)?;

        // get the target range for the current op
        let op_target = self.cigar.op_target_range(index)?;
        let op_query = self.cigar.op_query_range(index)?;

        // compute the clipped query range from the target
        let mut target_range = op_target.clone();
        let mut query_range = op_query.clone();

        if op_target.end > self.target_range.end {
            let clipped = op_target.end - self.target_range.end;
            target_range.end -= clipped;
            op_count -= clipped as u32;

            if op.consumes_query() {
                query_range.end -= clipped;
            }
        }

        if op_target.start < self.target_range.start {
            let clipped = self.target_range.start - op_target.start;
            target_range.start += clipped;
            op_count -= clipped as u32;

            if op.consumes_query() {
                query_range.start += clipped;
            }
        }

        let item = CigarIterItem {
            // target_range: target_clamped,
            target_range,
            query_range,
            // query_strand,
            op,
            op_count,
        };

        Some(item)
    }
}

// Packed representation of an entire cigar
pub struct Cigar(Vec<u32>);

// impl std::fmt::Debug for Cigar

impl Cigar {
    pub fn get(&self, index: usize) -> Option<(CigarOp, u32)> {
        let packed = self.0.get(index)?;
        Some(CigarOp::unpack(*packed))
    }

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

    pub fn target_and_query_len(&self) -> [u64; 2] {
        self.iter()
            .map(|(op, count)| {
                let c = count as u64;
                match op {
                    CigarOp::Eq | CigarOp::X | CigarOp::M => [c, c],
                    CigarOp::I => [0, c],
                    CigarOp::D => [c, 0],
                }
            })
            .fold([0, 0], |[a_t, a_q], [t, q]| [a_t + t, a_q + q])
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
        op | (count & !0xE0000000)
    }

    pub fn unpack(val: u32) -> (CigarOp, u32) {
        use CigarOp as Cg;
        let op = (val >> 29).min(4); // just to be safe (default to M)
        let count = val & !0xE0000000;

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

    pub fn consumes_target(&self) -> bool {
        use CigarOp::*;
        match self {
            Eq | X | M | D => true,
            _ => false,
        }
    }

    pub fn consumes_query(&self) -> bool {
        use CigarOp::*;
        match self {
            Eq | X | M | I => true,
            _ => false,
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

// #[derive(Debug)]
pub struct CigarIndex {
    // pub target_seq_id: usize,
    // pub query_seq_id: usize,

    // pub range_in_target_seq: std::ops::Range<u64>,
    // pub range_in_query_seq: std::ops::Range<u64>,
    pub target_len: u64,
    pub query_len: u64,
    pub query_strand: Strand,
    pub cigar: Cigar,
    // pub ops: Vec<(CigarOp, u64)>,
    pub op_line_vertices: Vec<[DVec2; 2]>,
    pub op_target_offsets: Vec<u64>,
    pub op_query_offsets: Vec<u64>,
}

impl CigarIndex {
    pub(crate) fn iter_target_range_impl(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> CigarIter<'_> {
        CigarIter::new(self, target_range)
    }

    // these provide the offsets for the operation *within the context of the cigar*,
    // i.e. they do not account for target/query start & end fields in PAF, or strand
    fn op_target_range(&self, op_ix: usize) -> Option<std::ops::Range<u64>> {
        if op_ix + 1 >= self.op_target_offsets.len() {
            return None;
        }
        let start = *self.op_target_offsets.get(op_ix)?;
        let end = *self.op_target_offsets.get(op_ix + 1)?;
        Some(start..end)
    }

    fn op_query_range(&self, op_ix: usize) -> Option<std::ops::Range<u64>> {
        if op_ix + 1 >= self.op_query_offsets.len() {
            return None;
        }
        let start = *self.op_query_offsets.get(op_ix)?;
        let end = *self.op_query_offsets.get(op_ix + 1)?;
        Some(start..end)
    }

    pub fn target_and_query_len(&self) -> [u64; 2] {
        self.cigar.target_and_query_len()
    }

    pub fn from_cigar(
        cigar: Cigar,
        target_len: u64,
        query_len: u64,
        // NB & TODO: Strand here vestigial; op_line_vertices
        // should be in Alignment or elsewhere
        query_strand: Strand,
    ) -> Self {
        use CigarOp as Cg;

        let mut op_line_vertices = Vec::new();
        let mut op_target_offsets = Vec::new();
        let mut op_query_offsets = Vec::new();

        let mut target_offset = 0u64;

        // query offsets are always stored as 0-based & increasing,
        // even when reverse
        let mut query_offset = 0u64;

        for (op, count) in cigar.iter() {
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
                    target_offset += count as u64;
                    query_offset += count as u64;
                }
                Cg::D => {
                    // increment target
                    target_offset += count as u64;
                }
                Cg::I => {
                    // increment query
                    query_offset += count as u64;
                }
                _ => (),
            }
        }
        // debug_assert_eq!(target_offset, target_len);
        // debug_assert_eq!(query_offset, query_len);
        op_target_offsets.push(target_offset);
        op_query_offsets.push(query_offset);

        Self {
            query_strand,
            cigar,
            op_line_vertices,
            op_target_offsets,
            op_query_offsets,
            target_len: target_offset,
            query_len: query_offset,
        }
    }

    pub fn from_cigar_string(
        cigar: &str,
        target_len: u64,
        query_len: u64,
        query_strand: Strand,
    ) -> Self {
        let cigar = Cigar::parse_str(cigar);
        // let cigar = CigarOp::parse_str_into_vec(cigar);

        Self::from_cigar(cigar, target_len, query_len, query_strand)
    }

    pub fn from_paf_line(
        paf_line: &crate::PafLine<&str>,
        // target_seq_id: usize,
        // query_seq_id: usize,
    ) -> Self {
        let query_strand = if paf_line.strand_rev {
            Strand::Reverse
        } else {
            Strand::Forward
        };

        let target_len = paf_line.tgt_seq_end - paf_line.tgt_seq_start;
        let query_len = paf_line.query_seq_end - paf_line.query_seq_start;

        Self::from_cigar_string(&paf_line.cigar, target_len, query_len, query_strand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PAF_RECORD: &'static str = "Y12#1#chrXIV	800685	30204	50196	+	S288C#1#chrXIV	800685	30000	50000	19922	20001	255	cg:Z:578=1X922=1X1135=1X334=1X194=1X653=1X90=1X32=1X715=1X41=1X29=1X92=1X368=1X504=1X140=1X14=1X18=1X233=1X703=4D6=1X603=1X844=1X86=1X562=1X1081=1X64=1X151=1X32=1X73=1X58=1X59=1X861=1X1216=1X1432=1X291=1X313=1X825=4D189=1X173=1X53=1X26=1X334=1X190=1X74=1X226=2X59=1X251=1X90=1X118=1X80=1X405=1X265=1X80=1X184=1X27=1X212=1X59=1X41=1X197=1X132=1X414=1X38=1X15=1X60=1X69=1X51=2X21=1D236=1X60=1I17=1X85=1X39=";

    const TEST_CIGAR: &'static str = "578=1X922=1X1135=1X334=1X194=1X653=1X90=1X32=1X715=1X41=1X29=1X92=1X368=1X504=1X140=1X14=1X18=1X233=1X703=4D6=1X603=1X844=1X86=1X562=1X1081=1X64=1X151=1X32=1X73=1X58=1X59=1X861=1X1216=1X1432=1X291=1X313=1X825=4D189=1X173=1X53=1X26=1X334=1X190=1X74=1X226=2X59=1X251=1X90=1X118=1X80=1X405=1X265=1X80=1X184=1X27=1X212=1X59=1X41=1X197=1X132=1X414=1X38=1X15=1X60=1X69=1X51=2X21=1D236=1X60=1I17=1X85=1X39=";

    fn print_u32(val: u32) {
        print!("{val:0<32b}");
    }

    #[test]
    fn test_cigar_index() {
        let cg_str = "50=10I5X7D20M";
        // let cg_str = "5=5I5X5D5M";

        let cg_ops = Cigar::parse_str(cg_str);
        let [target_len, query_len] = cg_ops.target_and_query_len();

        // for (i, (val, (op, count))) in std::iter::zip(&cg_ops.0, cg_ops.iter()).enumerate() {}
        // println!("target: {target_len}, query: {query_len}");

        assert_eq!(target_len, 82);
        assert_eq!(query_len, 85);

        // create cigarindex
        let cg_index = CigarIndex::from_cigar(cg_ops, target_len, query_len, Strand::Forward);
        // TODO test range mapping

        // TODO test cigar index iteration
        let expected_ranges = [
            (0..50, 0..50u64),
            (50..50, 50..60),
            (50..55, 60..65),
            (55..62, 65..65),
            (62..82, 65..85),
        ];
        let actual = cg_index
            .iter_target_range(0..target_len)
            .map(|c| (c.target_range.clone(), c.query_range.clone()))
            .collect::<Vec<_>>();
        assert_eq!(expected_ranges.as_slice(), actual.as_slice());

        // test bp-level cutting of cigar index iter
        let expected_ranges = [(30..50, 30..50u64), (50..50, 50..60), (50..55, 60..65)];
        let actual_ranges = cg_index
            .iter_target_range(30..55)
            .map(|c| (c.target_range.clone(), c.query_range.clone()))
            .collect::<Vec<_>>();

        assert_eq!(expected_ranges.as_slice(), actual_ranges.as_slice());

        // for item in cg_index.iter_target_range(0..target_len) {
        // for item in cg_index.iter_target_range(0..30) {
        // for item in cg_index.iter_target_range(30..55) {
        //     println!("{item:?}");
        // }
    }

    /*
    #[test]
    fn cigar_raster_iter() {
        let paf_line = crate::parse_paf_line(TEST_PAF_RECORD.split("\t")).unwrap();
        // let cigar_ops = CigarOp::parse_str_into_vec(TEST_CIGAR);

        let cigar = Cigar::parse_str(test_cigar);

        let cg_index = CigarIndex::from_paf_line(&paf_line, 0, 0);
        println!("{:?}", cg_index);

        todo!();
    }
    */
}
