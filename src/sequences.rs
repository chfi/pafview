use std::sync::Arc;

use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(pub usize);

pub struct Sequences {
    sequence_names: Arc<bimap::BiMap<String, SeqId>>,

    sequences: Option<FxHashMap<SeqId, Vec<u8>>>,
}

impl Sequences {
    pub fn from_paf<'a>(
        lines: impl IntoIterator<Item = &'a crate::PafLine<&'a str>>,
    ) -> Option<Self> {
        let mut seq_names = bimap::BiMap::new();

        for line in lines {
            //
        }

        Some(Sequences {
            sequence_names: Arc::new(seq_names),
            sequences: None,
        })
    }

    pub fn from_paf_and_fasta<'a>(
        paf_lines: impl IntoIterator<Item = &'a crate::PafLine<&'a str>>,
        fasta_path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Self> {
        let mut seq_names = bimap::BiMap::new();
        let mut seqs: Vec<Vec<u8>> = Vec::new();

        let mut fasta_reader = std::fs::File::open(fasta_path)
            .map(std::io::BufReader::new)
            .map(noodles::fasta::Reader::new)?;

        for record in fasta_reader.records() {
            let record = record?;
            let name = std::str::from_utf8(record.name()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Sequence name was not valid UTF-8",
                )
            })?;
        }

        // for record in fasta_index {
        //     fasta_reader.
        //     //
        // }

        let seqs = seqs
            .into_iter()
            .enumerate()
            .map(|(i, seq)| (SeqId(i), seq))
            .collect::<FxHashMap<_, _>>();

        Ok(Sequences {
            sequence_names: Arc::new(seq_names),
            sequences: Some(seqs),
        })
    }
}
