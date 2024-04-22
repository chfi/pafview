use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    sequences::{SeqId, Sequences},
    CigarIndex, ProcessedCigar,
};

pub struct Alignment {
    pub target_id: SeqId,
    // pub target_
    pub cigar: CigarIndex,
}

pub struct AlignmentIter<'cg, 'seq> {
    cigar: &'cg CigarIndex,
    seqs: &'seq Sequences,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
    // query_range: std::ops::Range<u64>,
}

pub struct AlignmentIterItem<'seq> {
    op_ix: usize,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,

    target_seq: Option<&'seq [u8]>,
    query_seq: Option<&'seq [u8]>,
    query_rev: bool,
}

impl<'cg, 'seq> AlignmentIter<'cg, 'seq> {
    fn new(
        cigar: &'cg CigarIndex,
        seqs: &'seq Sequences,
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
            seqs,
            op_index_range: todo!(),
            target_range,
            // query_range: todo!(),
        }
    }
}

impl<'cg, 'seq> Iterator for AlignmentIter<'cg, 'seq> {
    type Item = AlignmentIterItem<'seq>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

impl Alignment {
    pub fn iter_target_range<'cg, 'seq>(
        &'cg self,
        sequences: &'seq Sequences,
        target_range: std::ops::Range<u64>,
    ) -> AlignmentIter<'cg, 'seq> {
        AlignmentIter::new(&self.cigar, sequences, target_range)
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

                let cigar = CigarIndex::from_paf_line(&paf_line, 0, 0);

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
