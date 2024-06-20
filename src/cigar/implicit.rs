use std::any::Any;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::VecDeque, path::Path};

use coitrees::IntervalTree;
use impg::impg::{AdjustedInterval, Impg};
use rustc_hash::FxHashMap;

use crate::sequences::{SeqId, Sequences};

pub struct ImpgIndex {
    impg: Arc<Impg>,

    paf_path: PathBuf,
    // cigar_range_index: FxHashMap<(SeqId, SeqId), std::ops::Range<u64>>,
}

impl ImpgIndex {
    pub fn debug_print(&self) {
        println!("<ImpgIndex Debug>");
        println!("<<<<<<<< >>>>>>>>");
        println!(" >> treemap keys ");
        let mut keys = self.impg.trees.keys().collect::<Vec<_>>();
        keys.sort();
        println!();

        for &key in &keys {
            let tree = self.impg.trees.get(&key).unwrap();
            // for (key, tree) in self.impg.trees.iter() {
            // key ~ target id?
            let tgt_name = self.impg.seq_index.get_name(*key).unwrap();
            println!("{key} - {tgt_name}");
            // let queries = tree.
        }
        // println!("{keys:?}");
        println!("<End ImpgIndex Debug>");
        println!("<<<<<<<<<< >>>>>>>>>>");
        println!();
    }

    pub fn impg_cigars(
        index: &Arc<ImpgIndex>,
        sequences: &Sequences,
    ) -> FxHashMap<(SeqId, SeqId), Vec<ImpgCigar>> {
        let mut pair_cigars: FxHashMap<_, Vec<_>> = FxHashMap::default();

        for (&impg_target_id, tree) in index.impg.trees.iter() {
            // get name from impg
            let tgt_name = index.impg.seq_index.get_name(impg_target_id).unwrap();
            // then get pafview id from sequences
            let tgt_id = *sequences.names().get_by_left(tgt_name).unwrap();

            for query in tree.iter() {
                let meta = query.metadata;

                let impg_query_id = meta.query_id;
                let qry_name = index.impg.seq_index.get_name(impg_query_id).unwrap();
                let qry_id = *sequences.names().get_by_left(qry_name).unwrap();
                // let target_len = meta.target_end as u64 - meta.target_start as u64;
                // let query_len = meta.query_end as u64 - meta.query_start as u64;

                let target_range = (meta.target_start as u64)..(meta.target_end as u64);
                let query_range = (meta.query_start as u64)..(meta.query_end as u64);

                let impg_cigar = ImpgCigar {
                    target_range,
                    query_range,
                    query_strand: meta.strand.into(),
                    impg: index.clone(),
                    impg_target_id,
                    impg_query_id,
                };

                pair_cigars
                    .entry((tgt_id, qry_id))
                    .or_default()
                    .push(impg_cigar);
                // cigars.insert((tgt_id, qry_id), impg_cigar);
            }
        }

        for cigars in pair_cigars.values_mut() {
            cigars.sort_by_key(|cg| cg.target_range.start);
        }

        pair_cigars
    }

    // pub fn impg_cigar_for_pair(&self, target_id: SeqId, query_id: SeqId) -> Option<ImpgCigar> {
    //     todo!();
    // }

    pub fn deserialize_file(
        sequences: &Sequences,
        impg_path: impl AsRef<Path>,
        paf_path: impl AsRef<Path>,
        // paf_path: &str,
    ) -> anyhow::Result<Self> {
        use std::fs::File;
        use std::io::{self, prelude::*};

        let impg_reader = File::open(impg_path).map(io::BufReader::new)?;
        let serializable: impg::impg::SerializableImpg = bincode::deserialize_from(impg_reader)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to deserialize index: {:?}", e),
                )
            })?;

        let paf_str = paf_path.as_ref().to_str().unwrap();
        let impg = Impg::from_paf_and_serializable(paf_str, serializable);

        // let mut cigar_range_index = FxHashMap::default();
        let mut paf_reader = File::open(&paf_path).map(io::BufReader::new)?;

        let mut paf_buf = String::new();

        let mut file_offset = 0;

        loop {
            paf_buf.clear();

            let this_offset = file_offset;

            let read = paf_reader.read_line(&mut paf_buf)?;
            if read == 0 {
                break;
            }

            let line = &paf_buf[..read];

            let Some(paf_line) = crate::paf::parse_paf_line(line.split('\t')) else {
                continue;
            };

            file_offset += read;

            // paf_reader.
        }

        let paf_path = paf_path.as_ref().to_owned();

        Ok(Self {
            impg: impg.into(),
            paf_path,
            // cigar_range_index,
        })
    }
}

pub struct ImpgCigar {
    // pub target_len: u64,
    // pub query_len: u64,
    pub target_range: std::ops::Range<u64>,
    pub query_range: std::ops::Range<u64>,
    pub query_strand: super::Strand,
    // pub op_line_vertices: Vec<[ultraviolet::DVec2; 2]>,
    impg: Arc<ImpgIndex>,
    impg_target_id: u32,
    impg_query_id: u32,
}

impl ImpgCigar {
    fn query(&self, range: std::ops::Range<u64>) -> Vec<impg::impg::AdjustedInterval> {
        let start = (self.target_range.start + range.start) as i32;
        let end = (self.target_range.start + range.end) as i32;
        // let start = range.start as i32;
        // let end = range.end as i32;
        self.impg.impg.query(self.impg_target_id, start, end)
    }

    // pub fn

    // from_cigar?

    // iter_target_range(&self, target_range: std::ops::Range<u64>) ->

    fn iter_full_cigar_impl(&self) -> std::io::Result<Vec<impg::impg::CigarOp>> {
        let tree = self.impg.impg.trees.get(&self.impg_target_id).unwrap();

        // TODO support bgzipped paf

        let (interval, query_range) = tree
            .iter()
            .find_map(|interval| {
                let meta = interval.metadata;
                let start = meta.query_start as u64;
                let end = meta.query_end as u64;
                let strand = super::Strand::from(meta.strand);

                (interval.metadata.query_id == self.impg_query_id && strand == self.query_strand)
                    .then_some((interval, start..end))
            })
            .unwrap();

        let cigar = interval.metadata.get_cigar_ops(&self.impg.paf_path, None);

        Ok(cigar)
    }

    pub fn iter_target_range_impl(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Option<ImpgCigarIter> {
        let query = self.query(target_range);
        // let interval = self
        //     .query(target_range)

        let interval = query
            .into_iter()
            .find(|(query, _cigar, _target_i_guess)| query.metadata == self.impg_query_id);

        // dbg!(&interval);
        let (query, cigar, target) = interval?;

        // let target =

        let cigar = VecDeque::from(cigar);

        let target_offset = self.target_range.start;
        let query_offset = self.query_range.start;

        Some(ImpgCigarIter {
            target,
            query,
            cigar,
            target_offset,
            query_offset,
        })
    }
}

impl super::IndexedCigar for ImpgCigar {
    fn whole_cigar(&self) -> Box<dyn Iterator<Item = (crate::CigarOp, u32)> + '_> {
        let cigar = self.iter_full_cigar_impl().unwrap();
        let iter = cigar.into_iter().map(|i_op| {
            let op = crate::CigarOp::try_from(i_op.op()).unwrap();
            let len = i_op.len() as u32;
            (op, len)
        });

        // if self.query_strand.is_rev() {
        //     let iter = iter.rev();
        //     Box::new(iter)
        // } else {
        Box::new(iter)
        // }
    }

    fn iter_target_range(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Box<dyn Iterator<Item = crate::CigarIterItem> + '_> {
        if let Some(iter) = self.iter_target_range_impl(target_range) {
            Box::new(iter)
        } else {
            let iter = std::iter::empty();
            Box::new(iter)
        }
        // let iter = self.iter_target_range_impl(target_range).unwrap();
        // Box::new(iter)
    }
}

pub struct ImpgCigarIter {
    target_offset: u64,
    query_offset: u64,

    target: coitrees::Interval<u32>,
    query: coitrees::Interval<u32>,

    cigar: VecDeque<impg::impg::CigarOp>,
    // cigar: AdjustedInterval,
    // cigar: Vec<u8>,
}

impl Iterator for ImpgCigarIter {
    type Item = super::CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.target.first >= self.target.last {
            return None;
        }

        let next_op = self.cigar.pop_front()?;

        let op = super::CigarOp::try_from(next_op.op()).ok()?;
        let op_count = next_op.len() as u32;

        let tgt_start = self.target.first as u64 - self.target_offset;
        let tgt_end = tgt_start + op_count as u64;

        let qry_start = self.query.first as u64 - self.query_offset;
        let qry_end = qry_start + op_count as u64;

        // NB: maybe an off-by-one here; fix later
        let item = super::CigarIterItem {
            target_range: tgt_start..tgt_end,
            query_range: qry_start..qry_end,
            op,
            op_count,
        };

        self.target.first += op_count as i32;
        self.query.first += op_count as i32;

        Some(item)
    }
}

/*
pub struct ImpgCigarIter<'cg> {
    cigar: &'cg ImpgCigar,

    op_index_range: std::ops::Range<usize>,
    target_range: std::ops::Range<u64>,
}

impl<'cg> ImpgCigarIter<'cg> {}
*/

impl From<impg::paf::Strand> for super::Strand {
    fn from(value: impg::paf::Strand) -> Self {
        match value {
            impg::paf::Strand::Forward => Self::Forward,
            impg::paf::Strand::Reverse => Self::Reverse,
        }
    }
}

impl Into<impg::paf::Strand> for super::Strand {
    fn into(self) -> impg::paf::Strand {
        match self {
            Self::Forward => impg::paf::Strand::Forward,
            Self::Reverse => impg::paf::Strand::Reverse,
        }
    }
}
