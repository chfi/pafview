use rustc_hash::{FxHashMap, FxHashSet};

use ultraviolet::DVec2;

use crate::{
    sequences::{SeqId, Sequences},
    Cigar, CigarIndex, CigarIter, CigarOp, ProcessedCigar, Strand,
};

/// Location and orientation of an alignment of two sequences of
/// lengths `target_total_len` and `query_total_len`
#[derive(Clone)]
pub struct AlignmentLocation {
    pub target_total_len: u64,
    pub target_range: std::ops::Range<u64>,

    pub query_total_len: u64,
    pub query_range: std::ops::Range<u64>,
    pub query_strand: Strand,
}

impl AlignmentLocation {
    pub fn aligned_target_len(&self) -> u64 {
        self.target_range.end - self.target_range.start
    }

    pub fn aligned_query_len(&self) -> u64 {
        self.query_range.end - self.query_range.start
    }

    /// Maps `local_range` so that it is offset according to `self.target_range`
    /// `local_range.end` must be smaller than or equal to `self.aligned_target_len()`
    pub fn map_from_local_target_range(
        &self,
        local_range: std::ops::Range<u64>,
    ) -> std::ops::Range<u64> {
        let start = local_range.start + self.target_range.start;
        let end = local_range.end + self.target_range.start;
        start..end
    }

    /// Maps `local_range` so that it is offset according to `self.query_range`.
    /// Takes strand into account, e.g. if `query_strand` is `Reverse` and `local_range = 0..10`,
    /// the output will point to the last 10 bytes of the aligned part of the query sequence
    ///
    /// `local_range.end` must be smaller than or equal to `self.aligned_query_len()`
    pub fn map_from_local_query_range(
        &self,
        local_range: std::ops::Range<u64>,
    ) -> std::ops::Range<u64> {
        match self.query_strand {
            Strand::Forward => {
                let start = local_range.start + self.query_range.start;
                let end = local_range.end + self.query_range.start;
                start..end
            }
            Strand::Reverse => {
                let end = self.query_range.end - local_range.start;
                let start = end - (local_range.end - local_range.start);
                start..end
            }
        }
    }
}

pub struct Alignment {
    pub target_id: SeqId,
    pub query_id: SeqId,

    pub location: AlignmentLocation,
    pub cigar: CigarIndex,
    // pub cigar_op_vertices: Vec<[DVec2; 2]>,
}

pub struct AlignmentIter<'cg> {
    // cigar: &'cg CigarIndex,
    cigar_iter: CigarIter<'cg>,
    location: AlignmentLocation,
    // op_index_range: std::ops::Range<usize>,
    // target_range: std::ops::Range<u64>,
    // query_range: std::ops::Range<u64>,
}

impl<'cg> AlignmentIter<'cg> {
    fn new(
        alignment: &'cg Alignment,
        // cigar: &'cg CigarIndex,
        target_range: std::ops::Range<u64>,
    ) -> Self {
        // let c
        let cigar_iter = alignment.cigar.iter_target_range(target_range);

        Self {
            cigar_iter,
            location: alignment.location.clone(),
            // op_index_range: todo!(),
            // target_range,
            // query_range: todo!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignmentIterItem {
    // op_ix: usize,
    pub op: CigarOp,
    pub op_count: u32,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,

    query_rev: bool,
}

impl AlignmentIterItem {
    pub fn target_seq_range(&self) -> &std::ops::Range<u64> {
        &self.target_range
    }

    pub fn query_seq_range(&self) -> &std::ops::Range<u64> {
        &self.query_range
    }

    // pub fn iter_seqs<'seq>(
    //     &self,
    //     target_seq: Option<&'seq [u8]>,
    //     query_seq: Option<&'seq [u8]>,
    // ) -> impl Iterator<Item = [(u64, Option<char>); 2]> + 'seq + '_ {
    //     match self.op {
    //         CigarOp::Eq => todo!(),
    //         CigarOp::X => todo!(),
    //         CigarOp::I => todo!(),
    //         CigarOp::D => todo!(),
    //         CigarOp::M => todo!(),
    //     }
    // }
    // ) -> impl Iterator<Item = ([u64; 2], [Option<char>; 2]
}

/// Steps through the alignment operation one bp at a time, outputting
/// the target and query sequence offsets at each point
impl Iterator for AlignmentIterItem {
    type Item = [usize; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if self.op_count == 0 {
            return None;
        }
        let next_tgt = self.target_range.next()?;

        let next_qry = if self.query_rev {
            self.query_range.next_back()?
        } else {
            self.query_range.next()?
        };
        self.op_count -= 1;
        Some([next_tgt as usize, next_qry as usize])
    }
}

impl<'cg> Iterator for AlignmentIter<'cg> {
    type Item = AlignmentIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let cg_item = self.cigar_iter.next()?;
        let target_range = self
            .location
            .map_from_local_target_range(cg_item.target_range);
        let query_range = self
            .location
            .map_from_local_query_range(cg_item.query_range);

        Some(AlignmentIterItem {
            op: cg_item.op,
            op_count: cg_item.op_count,
            target_range,
            query_range,
            query_rev: self.location.query_strand.is_rev(),
        })
    }
}

impl Alignment {
    pub fn new(seq_names: &bimap::BiMap<String, SeqId>, paf_line: PafLine<&'_ str>) -> Self {
        let cigar = CigarIndex::from_paf_line(&paf_line);

        let target_id = *seq_names.get_by_left(paf_line.tgt_name).unwrap();
        let query_id = *seq_names.get_by_left(paf_line.query_name).unwrap();

        let target_range = paf_line.tgt_seq_start..paf_line.tgt_seq_end;
        let query_range = paf_line.query_seq_start..paf_line.query_seq_end;

        let location = AlignmentLocation {
            target_range,
            query_range,
            query_strand: cigar.query_strand,
            target_total_len: paf_line.tgt_seq_len,
            query_total_len: paf_line.query_seq_len,
        };

        Self {
            target_id,
            query_id,
            location,
            cigar,
        }
    }

    pub fn iter_target_range<'cg>(
        &'cg self,
        target_range: std::ops::Range<u64>,
    ) -> AlignmentIter<'cg> {
        AlignmentIter::new(&self, target_range)
    }
}

pub struct Alignments {
    pub pairs: FxHashMap<(SeqId, SeqId), Alignment>,
}

pub fn load_input_files(
    paf_path: impl AsRef<std::path::Path>,
    fasta_path: Option<impl AsRef<std::path::Path>>,
) -> anyhow::Result<(Alignments, Sequences)> {
    use std::io::prelude::*;

    let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;
    let mut lines = Vec::new();

    for line in reader.lines() {
        lines.push(line?);
    }
    println!("parsing {} lines", lines.len());
    let paf_lines = lines
        .iter()
        .filter_map(|s| parse_paf_line(s.split('\t')))
        .collect::<Vec<_>>();

    let sequences = if let Some(fasta_path) = fasta_path {
        Sequences::from_fasta(fasta_path)?
    } else {
        Sequences::from_paf(&paf_lines).unwrap()
    };

    println!("using {} sequences", sequences.len());

    let alignments = Alignments::from_paf_lines(&sequences, paf_lines);

    Ok((alignments, sequences))
}

impl Alignments {
    pub fn from_paf_lines<'l>(
        // NB: construct Sequences from iterator over PafLines (or FASTA) before
        sequences: &Sequences,
        lines: impl IntoIterator<Item = PafLine<&'l str>>,
    ) -> Self {
        let mut pairs = FxHashMap::default();

        for paf_line in lines {
            let target_id = sequences.names().get_by_left(paf_line.tgt_name).unwrap();
            let query_id = sequences.names().get_by_left(paf_line.query_name).unwrap();
            let alignment = Alignment::new(&sequences.names(), paf_line);

            pairs.insert((*target_id, *query_id), alignment);
        }

        Self { pairs }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PafLine<S> {
    pub query_name: S,
    pub query_seq_len: u64,
    pub query_seq_start: u64,
    pub query_seq_end: u64,

    pub tgt_name: S,
    pub tgt_seq_len: u64,
    pub tgt_seq_start: u64,
    pub tgt_seq_end: u64,

    pub strand_rev: bool,
    pub cigar: S,
}

pub(crate) fn parse_paf_line<'a>(
    mut fields: impl Iterator<Item = &'a str>,
) -> Option<PafLine<&'a str>> {
    let (query_name, query_seq_len, query_seq_start, query_seq_end) =
        parse_name_range(&mut fields)?;
    let strand = fields.next()?;
    let (tgt_name, tgt_seq_len, tgt_seq_start, tgt_seq_end) = parse_name_range(&mut fields)?;

    let cigar = fields.skip(3).find_map(|s| s.strip_prefix("cg:Z:"))?;

    Some(PafLine {
        query_name,
        query_seq_len,
        query_seq_start,
        query_seq_end,

        tgt_name,
        tgt_seq_len,
        tgt_seq_start,
        tgt_seq_end,

        strand_rev: strand == "-",
        cigar,
    })
}

fn parse_name_range<'a>(
    mut fields: impl Iterator<Item = &'a str>,
) -> Option<(&'a str, u64, u64, u64)> {
    let name = fields.next()?;
    let len = fields.next()?.parse().ok()?;
    let start = fields.next()?.parse().ok()?;
    let end = fields.next()?.parse().ok()?;
    Some((name, len, start, end))
}

#[deprecated]
pub struct PafInput {
    // pub alignments: FxHashMap<(SeqId, SeqId), Alignment>,
    // pub alignments: FxHashMap<(SeqId, SeqId), CigarIndex>,
    pub queries: Vec<crate::AlignedSeq>,
    pub targets: Vec<crate::AlignedSeq>,

    // alignment_pairs: Vec<(
    pub pair_line_ix: FxHashMap<(SeqId, SeqId), usize>,

    // match_edges: Vec<[DVec2; 2]>,
    pub processed_lines: Vec<ProcessedCigar>,
}

impl PafInput {
    pub fn total_matches(&self) -> usize {
        self.processed_lines
            .iter()
            .map(|l| l.match_edges.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_iter() {
        // AlignmentIter should emit (or, AlignmentIterItem should have methods that return)
        //  - a sequence of [target, query] positions, each with cigar op & possibly 1 or 2 sequence indices
        // -- order doesn't really matter there

        //  - the forward sequence slices of the corresponding target and query sequences, if available
        //  - (maybe) char-level iterators over the sequences in orientation order

        // the returned sequences/cigar ops (and ranges) should be cut
        // at the ends, but that can be done by the caller

        let cg_str = "50=10I5X7D20M";
        // let cg_str = "5=5I5X5D5M";

        let cg_ops = Cigar::parse_str(cg_str);
        let [target_len, query_len] = cg_ops.target_and_query_len();

        let target_total = target_len + 30;
        let query_total = query_len + 20;

        let alignment = Alignment {
            target_id: SeqId(0),
            query_id: SeqId(0),

            location: AlignmentLocation {
                target_total_len: target_total,
                target_range: 15..(15 + target_len),
                query_total_len: query_total,
                query_range: 0..query_len,
                query_strand: Strand::Reverse,
            },
            cigar: CigarIndex::from_cigar(cg_ops, target_len, query_len, Strand::Reverse),
        };

        for item in AlignmentIter::new(&alignment, 0..30) {
            println!("{item:?}");
        }
    }
}
