use std::path::Path;
use std::sync::Arc;

use impg::impg::{AdjustedInterval, Impg};

pub struct ImpgIndex {
    impg: Arc<Impg>,
}

impl ImpgIndex {
    pub fn deserialize_file(
        impg_path: impl AsRef<Path>,
        paf_path: impl AsRef<Path>,
        // paf_path: &str,
    ) -> anyhow::Result<Self> {
        use std::io;

        let impg_reader = std::fs::File::open(impg_path).map(io::BufReader::new)?;
        let serializable: impg::impg::SerializableImpg = bincode::deserialize_from(impg_reader)
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to deserialize index: {:?}", e),
                )
            })?;

        let paf_path = paf_path.as_ref().to_str().unwrap();
        let impg = Impg::from_paf_and_serializable(paf_path, serializable);

        Ok(Self { impg: impg.into() })
    }
}

pub struct ImpgCigar {
    pub target_len: u64,
    pub query_len: u64,
    pub query_strand: super::Strand,
    pub op_line_vertices: Vec<[ultraviolet::DVec2; 2]>,

    impg: Arc<Impg>,
    impg_target_id: u32,
}

impl ImpgCigar {
    fn query(&self, range: std::ops::Range<u64>) -> Vec<impg::impg::AdjustedInterval> {
        let start = range.start as i32;
        let end = range.end as i32;
        self.impg.query(self.impg_target_id, start, end)
    }

    // pub fn

    // from_cigar?

    // iter_target_range(&self, target_range: std::ops::Range<u64>) ->
}

pub struct ImpgCigarIter {
    cigar: AdjustedInterval,
    // cigar: Vec<u8>,
}

impl Iterator for ImpgCigarIter {
    type Item = super::CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
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
