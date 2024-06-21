use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::VecDeque, path::Path};

use coitrees::IntervalTree;
use impg::impg::Impg;
use rustc_hash::FxHashMap;

use crate::sequences::{SeqId, Sequences};

use anyhow::anyhow;

pub struct ImpgIndex {
    pub impg: Arc<Impg>,

    paf_bgz_index: Option<noodles::bgzf::gzi::Index>,

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
            // let tree = self.impg.trees.get(&key).unwrap();
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

                let qry_name = index.impg.seq_index.get_name(meta.query_id).unwrap();
                let qry_id = *sequences.names().get_by_left(qry_name).unwrap();

                let target_range = (meta.target_start as u64)..(meta.target_end as u64);
                let query_range = (meta.query_start as u64)..(meta.query_end as u64);

                let impg_cigar = ImpgCigar {
                    target_range,
                    query_range,
                    query_strand: meta.strand.into(),
                    impg: index.clone(),
                    impg_meta: meta.clone(),
                    impg_target_id,
                };

                pair_cigars
                    .entry((tgt_id, qry_id))
                    .or_default()
                    .push(impg_cigar);
            }
        }

        for cigars in pair_cigars.values_mut() {
            cigars.sort_by_key(|cg| cg.target_range.start);
        }

        pair_cigars
    }

    pub fn deserialize_file(
        impg_path: impl AsRef<Path>,
        paf_path: impl AsRef<Path>,
    ) -> anyhow::Result<Self> {
        use std::fs::File;
        use std::io;

        let impg_reader = File::open(impg_path).map(io::BufReader::new)?;
        let serializable: impg::impg::SerializableImpg = bincode::deserialize_from(impg_reader)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to deserialize index: {:?}", e),
                )
            })?;

        let paf_str = paf_path.as_ref().to_str().unwrap();
        let paf_path = paf_path.as_ref();
        let impg = Impg::from_paf_and_serializable(paf_str, serializable);

        let paf_bgz_index_path = paf_path.extension().and_then(|ext| {
            let new_ext = ["gz", "bgz"]
                .iter()
                .find_map(|e| (&ext == e).then(|| format!("{e}.gzi")))?;
            Some(paf_path.with_extension(new_ext))
        });

        let paf_bgz_index = if let Some(bgz_path) = paf_bgz_index_path.as_ref() {
            Some(
                noodles::bgzf::gzi::read(bgz_path)
                    .map_err(|e| anyhow!("Error loading bgzip index for PAF: {e:?}"))?,
            )
        } else {
            None
        };

        Ok(Self {
            impg: impg.into(),
            paf_bgz_index,
            paf_path: paf_path.to_path_buf(),
        })
    }
}

pub struct ImpgCigar {
    pub target_range: std::ops::Range<u64>,
    pub query_range: std::ops::Range<u64>,
    pub query_strand: super::Strand,
    impg: Arc<ImpgIndex>,
    impg_meta: impg::impg::QueryMetadata,
    impg_target_id: u32,
}

impl ImpgCigar {
    fn query(&self, range: std::ops::Range<u64>) -> Vec<impg::impg::AdjustedInterval> {
        let start = (self.target_range.start + range.start) as i32;
        let end = (self.target_range.start + range.end) as i32;
        self.impg.impg.query(self.impg_target_id, start, end)
    }

    fn iter_full_cigar_impl(&self) -> std::io::Result<Vec<impg::impg::CigarOp>> {
        let tree = self.impg.impg.trees.get(&self.impg_target_id).unwrap();

        let interval = tree
            .iter()
            .find(|interval| interval.metadata == &self.impg_meta)
            .expect("Interval not found in impg - should be impossible");

        let gzi_index = self.impg.paf_bgz_index.as_ref();
        let cigar = interval
            .metadata
            .get_cigar_ops(&self.impg.paf_path, gzi_index);

        Ok(cigar)
    }

    pub fn iter_target_range_impl(
        &self,
        local_target_range: std::ops::Range<u64>,
    ) -> Option<ImpgCigarIter> {
        let query = self.query(local_target_range.clone());

        let interval = query
            .into_iter()
            .find(|(query, _cigar, _target_i_guess)| query.metadata == self.impg_meta.query_id);

        let (query, cigar, _target) = interval?;

        let cigar = VecDeque::from(cigar);

        let target_range = local_target_range;

        let query_range = {
            let is_rev = query.first > query.last;
            let g_start = query.first.min(query.last) as u64;
            let g_end = query.first.max(query.last) as u64;
            let len = g_end - g_start;

            if !is_rev {
                let start = g_start - self.query_range.start;
                let end = g_end;
                start..end
            } else {
                let start = self.query_range.end - g_end;
                let end = start + len;
                start..end
            }
        };

        Some(ImpgCigarIter {
            target_range,
            query_range,
            cigar,
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

        Box::new(iter)
    }

    fn iter_target_range(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Box<dyn Iterator<Item = crate::CigarIterItem> + '_> {
        if let Some(iter) = self.iter_target_range_impl(target_range) {
            Box::new(iter)
        } else {
            Box::new(std::iter::empty())
        }
    }
}

pub struct ImpgCigarIter {
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    cigar: VecDeque<impg::impg::CigarOp>,
}

impl Iterator for ImpgCigarIter {
    type Item = super::CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.target_range.end == self.target_range.start {
            // if self.target.first >= self.target.last {
            return None;
        }

        let next_op = self.cigar.pop_front()?;

        let op = super::CigarOp::try_from(next_op.op()).ok()?;
        let op_count = next_op.len() as u32;

        let tgt_len = op.consumes_target().then_some(op_count).unwrap_or(0) as u64;
        let qry_len = op.consumes_query().then_some(op_count).unwrap_or(0) as u64;

        let tgt_start = self.target_range.start;
        let tgt_end = tgt_start + tgt_len;

        let qry_start = self.query_range.start;
        let qry_end = qry_start + qry_len;

        let item = super::CigarIterItem {
            target_range: tgt_start..tgt_end,
            query_range: qry_start..qry_end,
            op,
            op_count,
        };

        self.target_range.start = tgt_end.min(self.target_range.end);
        self.query_range.start = qry_end.min(self.query_range.end);

        Some(item)
    }
}

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
