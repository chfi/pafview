use rustc_hash::{FxHashMap, FxHashSet};

use ultraviolet::DVec2;

use crate::{
    sequences::{SeqId, Sequences},
    Cigar, CigarIndex, ProcessedCigar, Strand,
};

/// Location and orientation of an alignment of two sequences of
/// lengths `target_total_len` and `query_total_len`
pub struct AlignmentLocation {
    // target_total_len: u64,
    target_range: std::ops::Range<u64>,

    // query_total_len: u64,
    query_range: std::ops::Range<u64>,
    query_strand: Strand,
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
    cigar: &'cg CigarIndex,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
    // query_range: std::ops::Range<u64>,
}

pub struct AlignmentIterItem {
    op_ix: usize,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,

    query_rev: bool,
}

impl<'cg> AlignmentIter<'cg> {
    fn new(
        cigar: &'cg CigarIndex,
        //
        target_range: std::ops::Range<u64>,
    ) -> Self {
        let start_i = cigar
            .op_target_offsets
            .partition_point(|&t| t < target_range.start);
        let start_i = start_i.checked_sub(1).unwrap_or_default();

        let end_i = cigar
            .op_target_offsets
            .partition_point(|&t| t < target_range.end);
        // let query_range = q_start..q_end; //... hmmmm

        let op_index_range = start_i..end_i;

        Self {
            cigar,
            op_index_range: todo!(),
            target_range,
            // query_range: todo!(),
        }
    }
}

impl<'cg> Iterator for AlignmentIter<'cg> {
    type Item = AlignmentIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
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
        AlignmentIter::new(&self.cigar, target_range)
    }
}

pub struct PafInput {
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

    // pub fn parse_from_lines<'s>(lines: impl IntoIterator<Item = &'s str>) -> anyhow::Result<Self> {
    pub fn read_paf_file(paf_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        use std::io::prelude::*;

        let reader = std::fs::File::open(&paf_path).map(std::io::BufReader::new)?;

        // let mut sequences = FxHashMap::default();
        // let mut targets = Vec::new();
        // let mut queries = Vec::new();
        // let mut targets = FxHashMap::default();
        // let mut queries = FxHashMap::default();

        // let mut add_sequencesjj

        // fn print_paf(line: &PafLine<&str>) {
        //     println!(
        //         "{} - len {}, start {}, end {}",
        //         line.query_name, line.query_seq_len, line.query_seq_start, line.query_seq_end
        //     );
        // }

        let debug = ["SK1#1#chrI", "S288C#1#chrI"]
            .into_iter()
            .map(String::from)
            .collect::<FxHashSet<_>>();

        for line in reader.lines() {
            let line = line?;
            if let Some(paf_line) = parse_paf_line(line.split('\t')) {
                // println!("{paf_line:?}");

                // if let Some(existing) = targets.get(&paf_line.query_name) {
                // print!("found existing query ");
                // print_paf
                // println!("found query: {
                //
                // }

                let cigar = CigarIndex::from_paf_line(&paf_line);

                let target_total_len = paf_line.tgt_seq_len;
                let query_total_len = paf_line.query_seq_len;

                let target_align_len = paf_line.tgt_seq_end - paf_line.tgt_seq_start;
                let query_align_len = paf_line.query_seq_end - paf_line.query_seq_start;

                let [target_len, query_len] = cigar.target_and_query_len();

                if debug.contains(paf_line.tgt_name) {
                    print!(
                        "target: {}, query: {}\t",
                        paf_line.tgt_name, paf_line.query_name,
                    );
                    print!("total: {target_total_len}, {query_total_len}\t");
                    println!("aligned: {target_align_len}, {query_align_len}\tcigar: {target_len}, {query_len}");
                }

                // print!(
                //     "query: {}, target: {}\t",
                //     paf_line.query_name, paf_line.tgt_name
                // );
                // print!("total: {query_total_len}, {target_total_len}\t");
                // println!("aligned: {query_align_len}, {target_align_len}\tcigar: {query_len}, {target_len}");

                // if !sequences.contains_key(&paf_line.query_name) {
                //     //
                // }

                // println!(
                //     "query {} - len {}, start {}, end {}",
                //     paf_line.query_name,
                //     paf_line.query_seq_len,
                //     paf_line.query_seq_start,
                //     paf_line.query_seq_end
                // );

                // println!(
                //     "tgt {} - len {}, start {}, end {}",
                //     paf_line.tgt_name,
                //     paf_line.tgt_seq_len,
                //     paf_line.tgt_seq_start,
                //     paf_line.tgt_seq_end
                // );
            }
        }

        //
        todo!();
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
    }
}
