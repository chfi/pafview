use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use ultraviolet::DVec2;

use crate::{
    cigar::implicit::ImpgIndex,
    sequences::{SeqId, Sequences},
    CigarIndex, CigarIter, CigarOp, IndexedCigar, Strand,
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

    pub fn map_from_local_target_offset(&self, target_offset: u64) -> u64 {
        self.target_range.start + target_offset
    }

    pub fn map_from_local_query_offset(&self, query_offset: u64) -> u64 {
        match self.query_strand {
            Strand::Forward => self.query_range.start + query_offset,
            Strand::Reverse => self.query_range.end.checked_sub(query_offset).unwrap_or(0),
        }
    }

    /// Maps `local_range` so that it is offset according to `self.target_range`
    /// `local_range.end` must be smaller than or equal to `self.aligned_target_len()`
    pub fn map_from_aligned_target_range(
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
    pub fn map_from_aligned_query_range(
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

// pub struct AlignmentNew {
//     pub target_id: SeqId,
//     pub query_id: SeqId,

//     pub location: AlignmentLocation,
//     pub cigar: std::sync::Arc<dyn IndexedCigar>,
//     // pub cigar_op_vertices: Vec<[DVec2; 2]>,
//     pub cigar_op_line_vertices: Vec<[DVec2; 2]>,
// }

pub struct Alignment {
    pub target_id: SeqId,
    pub query_id: SeqId,

    pub location: AlignmentLocation,
    // pub cigar: CigarIndex,
    pub cigar: std::sync::Arc<dyn IndexedCigar + Send + Sync + 'static>,
    // pub cigar_op_vertices: Vec<[DVec2; 2]>,
    pub cigar_op_line_vertices: Vec<[DVec2; 2]>,
}

pub struct AlignmentIter<'cg> {
    // cigar: &'cg CigarIndex,
    // cigar_iter: CigarIter<'cg>,
    cigar_iter: crate::cigar::BoxedCigarIter<'cg>,
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

        // the iterator on CigarIndex takes a range within the alignment,
        // while AlignmentIter takes a range within the entire sequence,
        // so we need to offset/clip it

        let start = target_range
            .start
            .checked_sub(alignment.location.target_range.start)
            .unwrap_or_default();
        let end = target_range
            .end
            .checked_sub(alignment.location.target_range.start)
            .unwrap_or_default();

        let cigar_iter = alignment.cigar.iter_target_range(start..end);

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

    pub fn strand(&self) -> Strand {
        if self.query_rev {
            Strand::Reverse
        } else {
            Strand::Forward
        }
    }
}

impl<'cg> Iterator for AlignmentIter<'cg> {
    type Item = AlignmentIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let cg_item = self.cigar_iter.next()?;
        let target_range = self
            .location
            .map_from_aligned_target_range(cg_item.target_range);
        let query_range = self
            .location
            .map_from_aligned_query_range(cg_item.query_range);

        Some(AlignmentIterItem {
            op: cg_item.op,
            op_count: cg_item.op_count,
            target_range,
            query_range,
            query_rev: self.location.query_strand.is_rev(),
        })
    }
}

/// Steps through the alignment operation one bp at a time, outputting
/// the target and query sequence offsets at each point
impl Iterator for AlignmentIterItem {
    type Item = [usize; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if self.op_count == 0 {
            return None;
        }

        let next_tgt = if self.op.consumes_target() {
            self.target_range.next()?
        } else {
            self.target_range.start
        };

        let next_qry = match (self.op.consumes_query(), self.strand()) {
            (true, Strand::Forward) => self.query_range.next()?,
            (true, Strand::Reverse) => self.query_range.next_back()?,
            (false, Strand::Forward) => self.query_range.start,
            (false, Strand::Reverse) => self.query_range.end,
        };
        self.op_count -= 1;
        Some([next_tgt as usize, next_qry as usize])
    }
}

impl Alignment {
    pub fn new(seq_names: &bimap::BiMap<String, SeqId>, paf_line: PafLine<&'_ str>) -> Self {
        let target_id = *seq_names.get_by_left(paf_line.tgt_name).unwrap();
        let query_id = *seq_names.get_by_left(paf_line.query_name).unwrap();

        let target_range = paf_line.tgt_seq_start..paf_line.tgt_seq_end;
        let query_range = paf_line.query_seq_start..paf_line.query_seq_end;

        let query_strand = if paf_line.strand_rev {
            Strand::Reverse
        } else {
            Strand::Forward
        };

        let location = AlignmentLocation {
            target_range,
            query_range,
            query_strand,
            target_total_len: paf_line.tgt_seq_len,
            query_total_len: paf_line.query_seq_len,
        };

        let cigar_index = CigarIndex::from_paf_line(&paf_line);

        let mut op_line_vertices = Vec::new();

        let offsets_iter = std::iter::zip(
            &cigar_index.op_target_offsets,
            &cigar_index.op_query_offsets,
        );

        let ops_iter = std::iter::zip(cigar_index.cigar.iter(), offsets_iter);

        for ((op, count), (&tgt_cg, &qry_cg)) in ops_iter {
            // tgt_cg and qry_cg are offsets from the start of the cigar

            let (tgt_end, qry_end) = match op {
                CigarOp::Eq | CigarOp::X | CigarOp::M => {
                    //
                    (tgt_cg + count as u64, qry_cg + count as u64)
                }
                CigarOp::I => {
                    //
                    (tgt_cg, qry_cg + count as u64)
                }
                CigarOp::D => {
                    //
                    (tgt_cg + count as u64, qry_cg)
                }
            };

            let tgt_range = location.map_from_aligned_target_range(tgt_cg..tgt_end);
            let qry_range = location.map_from_aligned_query_range(qry_cg..qry_end);

            let mut from = DVec2::new(tgt_range.start as f64, qry_range.start as f64);
            let mut to = DVec2::new(tgt_range.end as f64, qry_range.end as f64);

            if location.query_strand.is_rev() {
                std::mem::swap(&mut from.y, &mut to.y);
            }

            op_line_vertices.push([from, to]);
        }

        Self {
            target_id,
            query_id,
            location,
            // cigar: cigar_index,
            cigar: Arc::new(cigar_index),
            cigar_op_line_vertices: op_line_vertices,
        }
    }

    pub fn iter_target_range<'cg>(
        &'cg self,
        target_range: std::ops::Range<u64>,
    ) -> AlignmentIter<'cg> {
        AlignmentIter::new(&self, target_range)
    }

    /*
    pub fn query_offset_at_target(&self, target_offset: u64) -> u64 {
        /*
        use std::cmp::Ordering;
        let query_offset = match (
            target_offset.cmp(&self.location.target_range.start),
            self.location.query_strand,
        ) {
            (Ordering::Less, Strand::Forward)
            | (Ordering::Greater, Strand::Reverse) => {
                0
                // self.location.query_range.start
                // todo!()
            }
            (Ordering::Less, Strand::Reverse)
            | (Ordering::Greater, Strand::Forward) => {
                self.location.query_total_len
                // self.location.query_range.end
                // todo!()
            }
            (Ordering::Equal, Strand::Forward) => {
                todo!()
            }
            (Ordering::Equal, Strand::Reverse) => {
                todo!()
            }
        };
        */

        if target_offset < self.location.target_range.start {
            if self.location.query_strand.is_rev() {
                self.location.query_total_len
                // self.location.query_range.end
            } else {
                0
                // self.location.query_range.start
            }
        } else if target_offset >= self.location.target_range.end {
            if self.location.query_strand.is_rev() {
                0
                // self.location.query_range.start
            } else {
                self.location.query_total_len
                // self.location.query_range.end
            }
        } else {
            let op_ix = self
                .cigar
                .op_target_offsets
                .partition_point(|&offset| offset < target_offset);
            let ix = op_ix.min(self.cigar.op_query_offsets.len() - 1);
            let target_origin = self.cigar.op_target_offsets[ix];
            let mut query_offset = self.cigar.op_query_offsets[ix];

            if let Some((op, _count)) = self.cigar.cigar.get(ix) {
                if op.consumes_query() && target_offset > target_origin {
                    let delta = target_offset - target_origin;
                    query_offset += delta;
                }
            }

            self.location.map_from_local_query_offset(query_offset)
        }
    }
    */
}

pub struct Alignments {
    pub pairs: FxHashMap<(SeqId, SeqId), Alignment>,
}

pub fn load_input_files(cli: &crate::cli::Cli) -> anyhow::Result<(Alignments, Sequences)> {
    use std::io::prelude::*;

    let reader = std::fs::File::open(&cli.paf).map(std::io::BufReader::new)?;
    let mut lines = Vec::new();

    for line in reader.lines() {
        lines.push(line?);
    }

    let filter_line = {
        let target_names = cli
            .target_seqs
            .as_ref()
            .map(|names| names.iter().cloned().collect::<FxHashSet<_>>());

        let query_names = cli
            .query_seqs
            .as_ref()
            .map(|names| names.iter().cloned().collect::<FxHashSet<_>>());

        move |paf_line: &PafLine<&str>| -> bool {
            if let Some(targets) = &target_names {
                if !targets.contains(paf_line.tgt_name) {
                    return false;
                }
            }

            if let Some(queries) = &query_names {
                if !queries.contains(paf_line.query_name) {
                    return false;
                }
            }

            true
        }
    };
    println!("parsing {} lines", lines.len());
    let paf_lines = lines
        .iter()
        .filter_map(|s| {
            let line = parse_paf_line(s.split('\t'))?;
            filter_line(&line).then_some(line)
        })
        .collect::<Vec<_>>();

    let sequences = if let Some(fasta_path) = &cli.fasta {
        Sequences::from_fasta(fasta_path)?
    } else {
        Sequences::from_paf(&paf_lines).unwrap()
    };

    println!("using {} sequences", sequences.len());

    let alignments = if let Some(impg_path) = cli.impg.as_ref() {
        Alignments::from_impg(&sequences, impg_path, &cli.paf)?
    } else {
        Alignments::from_paf_lines(&sequences, paf_lines)
    };

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

    pub fn from_impg(
        sequences: &Sequences,
        impg_path: impl AsRef<std::path::Path>,
        paf_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        let impg_index = Arc::new(ImpgIndex::deserialize_file(sequences, impg_path, paf_path)?);

        let impg_cigars = ImpgIndex::impg_cigars(&impg_index, sequences);

        let pairs = impg_cigars
            .into_iter()
            .map(|(pair, impg_cg)| {
                let target_id = todo!();
                let query_id = todo!();

                // let location = AlignmentLocation {};

                // let cigar =

                //
                todo!();
            })
            .collect::<FxHashMap<_, _>>();

        Ok(Self { pairs })
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

pub fn parse_paf_line<'a>(mut fields: impl Iterator<Item = &'a str>) -> Option<PafLine<&'a str>> {
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

        let cg_str = "50=10I5X7D20M";
        // let cg_str = "5=5I5X5D5M";

        let cg_ops = crate::Cigar::parse_str(cg_str);
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
            cigar: std::sync::Arc::new(CigarIndex::from_cigar(
                cg_ops,
                target_len,
                query_len,
                Strand::Reverse,
            )),

            cigar_op_line_vertices: Vec::new(), // not needed for testing
        };

        for item in AlignmentIter::new(&alignment, 0..30) {
            println!("{item:?}");

            for step in item {
                println!("{step:?}");
            }
        }
    }
}
