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

    pub match_edges: Vec<[DVec2; 2]>,
    pub match_offsets: Vec<[u64; 2]>,
    pub match_lens: Vec<u64>,
    // TODO these flags should be in a struct & single vec
    pub match_is_match: Vec<bool>,
    pub match_is_rev: Vec<bool>,

    pub match_cigar_index: Vec<usize>,

    pub aabb_min: DVec2,
    pub aabb_max: DVec2,

    pub cigar: Vec<(CigarOp, u64)>,
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
        let ops = paf_line
            .cigar
            .split_inclusive(['M', 'X', '=', 'D', 'I', 'S', 'H', 'N'])
            .filter_map(|opstr| {
                let count = opstr[..opstr.len() - 1].parse::<u64>().ok()?;
                let op = opstr.as_bytes()[opstr.len() - 1] as char;
                let op = CigarOp::try_from(op).ok()?;
                Some((op, count))
            })
            .collect::<Vec<_>>();

        let [mut target_pos, mut query_pos] = origin;

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
        let mut match_is_rev = Vec::new();

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
                        match_is_rev.push(paf_line.strand_rev);
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

            match_edges,
            match_offsets,
            match_lens,
            match_is_match,
            match_is_rev,

            match_cigar_index,
            cigar: ops,

            aabb_min,
            aabb_max,
        })
    }
}
