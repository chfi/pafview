use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct SeqId(pub usize);

pub struct Sequences {
    sequence_names: Arc<bimap::BiMap<String, SeqId>>,

    sequences: FxHashMap<SeqId, SequenceData>,
    // sequences: Option<FxHashMap<SeqId, Vec<u8>>>,
}

impl Sequences {
    pub fn get(&self, seq: SeqId) -> Option<&SequenceData> {
        self.sequences.get(&seq)
    }

    pub fn names(&self) -> &Arc<bimap::BiMap<String, SeqId>> {
        &self.sequence_names
    }

    pub fn len(&self) -> usize {
        self.sequence_names.len()
    }

    pub fn get_bytes(&self, seq: SeqId) -> Option<&[u8]> {
        let seq = self.sequences.get(&seq)?.seq.as_ref()?;
        Some(seq.as_slice())
    }

    pub fn from_paf<'a>(
        lines: impl IntoIterator<Item = &'a crate::PafLine<&'a str>>,
    ) -> Option<Self> {
        let mut seq_names = bimap::BiMap::new();
        let mut sequences = FxHashMap::default();

        let mut add_sequence = |name: &str, len: u64| {
            if !seq_names.contains_left(name) {
                let id = SeqId(sequences.len());
                seq_names.insert(name.to_string(), id);
                sequences.insert(
                    id,
                    SequenceData {
                        name: name.to_string(),
                        len,
                        seq: None,
                    },
                );
            }
        };

        for line in lines {
            add_sequence(line.tgt_name, line.tgt_seq_len);
            add_sequence(line.query_name, line.query_seq_len);
        }

        Some(Sequences {
            sequence_names: Arc::new(seq_names),
            sequences,
        })
    }

    pub fn from_fasta<'a>(fasta_path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let mut seq_names = bimap::BiMap::new();

        let mut sequences = FxHashMap::default();

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

            let id = SeqId(sequences.len());

            let seq = record.sequence().as_ref().to_vec();

            let seq_data = SequenceData {
                name: name.into(),
                len: seq.len() as u64,
                seq: Some(seq),
            };

            seq_names.insert(seq_data.name.clone(), id);
            sequences.insert(id, seq_data);
        }

        Ok(Sequences {
            sequence_names: Arc::new(seq_names),
            sequences,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SequenceData {
    name: String,
    len: u64,

    // TODO maybe Box<dyn AsRef<[u8]>>
    seq: Option<Vec<u8>>,
}

impl SequenceData {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> u64 {
        debug_assert_eq!(
            self.seq
                .as_ref()
                .map(|seq| seq.len() as u64)
                .unwrap_or(self.len),
            self.len
        );
        self.len
    }

    pub fn seq(&self) -> Option<&[u8]> {
        self.seq.as_ref().map(|seq| seq.as_slice())
    }
}
