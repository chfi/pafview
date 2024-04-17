use std::sync::Arc;

use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub usize);

pub struct Sequences {
    sequence_names: Arc<bimap::BiMap<String, SeqId>>,

    seqs: Option<FxHashMap<SeqId, Vec<u8>>>,
}

impl Sequences {
    // pub fn from_fasta(seq_names: &bimap::BiMap<String, SeqId>,
}
