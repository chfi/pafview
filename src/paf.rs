use std::{collections::BTreeMap, hash::Hash, sync::Arc};

use rustc_hash::{FxHashMap, FxHashSet};

use ultraviolet::DVec2;

use anyhow::anyhow;

use crate::app::alignments::AlignmentIndex;
use crate::{
    cigar::implicit::ImpgIndex,
    sequences::{SeqId, Sequences},
    CigarIndex, CigarIter, CigarOp, IndexedCigar, Strand,
};

/// Location and orientation of an alignment of two sequences of
/// lengths `target_total_len` and `query_total_len`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Clone)]
pub struct Alignment {
    pub target_id: SeqId,
    pub query_id: SeqId,

    pub location: AlignmentLocation,
    // pub cigar: CigarIndex,
    pub cigar_file_byte_range: Option<std::ops::Range<u64>>,
    pub cigar: std::sync::Arc<dyn IndexedCigar + Send + Sync + 'static>,
}

impl std::fmt::Debug for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cigar = self
            .cigar
            .whole_cigar()
            .take(10)
            .map(|(op, len)| {
                let c = char::from(op);
                format!("{c}{len}")
            })
            .collect::<String>();

        f.debug_struct("Alignment")
            .field("target_id", &self.target_id)
            .field("query_id", &self.query_id)
            .field("location", &self.location)
            .field("cigar", &cigar)
            .finish()
    }
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
    pub fn new(seq_names: &bimap::BiMap<String, SeqId>, paf_line: &PafLine<&'_ str>) -> Self {
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

        let (cigar, cigar_file_byte_range) =
            if let Some(cigar) = CigarIndex::from_paf_line(&paf_line) {
                let range = paf_line.cigar_file_range.clone();
                (Arc::new(cigar) as _, range)
            } else {
                (Arc::new(crate::cigar::NoCigar) as _, None)
            };

        Self {
            target_id,
            query_id,
            location,
            // cigar: cigar_index,
            cigar_file_byte_range,
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

// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct AlignmentIndex {
//     pub pair: (SeqId, SeqId),
//     pub index: usize,
// }

#[derive(bevy::prelude::Resource)]
pub struct Alignments {
    // alignments in line-order with the input PAF file
    pub alignments: Arc<Vec<Alignment>>,

    // values are indices into `alignments` vec
    pub indices: Arc<FxHashMap<(SeqId, SeqId), Vec<usize>>>,

    // pub pairs: Arc<FxHashMap<(SeqId, SeqId), Vec<Alignment>>>,

    // byte ranges for the cigars in the PAF file
    cigar_range_index_map: bimap::BiHashMap<AlignmentIndex, std::ops::Range<u64>>,
}

impl Alignments {
    pub fn pair_alignments(
        &self,
        key: (SeqId, SeqId),
    ) -> Option<impl Iterator<Item = &'_ Alignment>> {
        let indices = self.indices.get(&key)?;
        let iter = indices.iter().filter_map(|&i| self.alignments.get(i));
        Some(iter)
    }

    pub fn pairs<'a>(
        &'a self,
    ) -> impl Iterator<Item = ((SeqId, SeqId), impl Iterator<Item = &'a Alignment>)> + 'a {
        self.indices.iter().map(|(pair, indices)| {
            let alignments = indices.iter().filter_map(|i| self.alignments.get(*i));
            (*pair, alignments)
        })
    }
}

pub struct AlignmentMetadata {
    pub residue_matches: usize,
    pub alignment_block_length: usize,
    pub mapping_quality: u8,

    pub optional_fields: FxHashMap<[u8; 2], (char, String)>,
}

#[derive(bevy::prelude::Resource)]
pub struct PafMetadata {
    metadata: FxHashMap<(SeqId, SeqId), Vec<AlignmentMetadata>>,
}

impl PafMetadata {
    // pub fn get(&self, alignment: &crate::app::alignments::Alignment) -> Option<&[String]> {
    pub fn get_optional_fields(
        &self,
        alignment: &crate::app::alignments::AlignmentIndex,
    ) -> Option<&FxHashMap<[u8; 2], (char, String)>> {
        let pair = self.metadata.get(&(alignment.target, alignment.query))?;
        let metadata = pair.get(alignment.pair_index)?;
        Some(&metadata.optional_fields)
    }

    pub fn get(
        &self,
        alignment: &crate::app::alignments::AlignmentIndex,
    ) -> Option<&AlignmentMetadata> {
        let pair = self.metadata.get(&(alignment.target, alignment.query))?;
        pair.get(alignment.pair_index)
    }

    pub fn from_paf(
        sequences: &Sequences,
        paf_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<Self> {
        use std::io::prelude::*;

        let reader = std::fs::File::open(paf_path)
            .map(std::io::BufReader::new)
            .map_err(|e| anyhow!("Error opening PAF file: {e:?}"))?;

        let mut pairs: FxHashMap<(SeqId, SeqId), Vec<(u64, AlignmentMetadata)>> =
            FxHashMap::default();

        for line in reader.lines() {
            let line = line?;
            let fields = line.trim().split('\t').collect::<Vec<_>>();

            let qry_name = fields[0];
            let tgt_name = fields[5];
            let tgt_start = fields[7];

            let residue_matches: usize = fields[9].parse().unwrap_or_default();
            let alignment_block_length: usize = fields[10].parse().unwrap_or_default();
            let mapping_quality: u8 = fields[11].parse().unwrap_or_default();

            let qry_id = sequences.sequence_names.get_by_left(qry_name).copied();
            let tgt_id = sequences.sequence_names.get_by_left(tgt_name).copied();

            let Some(pair @ (tgt_id, qry_id)) = tgt_id.zip(qry_id) else {
                continue;
            };

            let opt_fields = fields
                .iter()
                .skip(8)
                .filter_map(|field| {
                    if field.len() < 5 {
                        return None;
                    }

                    let bytes = field.as_bytes();
                    if bytes.get(2) == Some(&b':') && bytes.get(4) == Some(&b':') {
                        let tag = &bytes[..2];
                        if tag.eq_ignore_ascii_case(b"cg") {
                            return None;
                        }
                        let &[a, b] = tag else {
                            return None;
                        };
                        let ty = bytes[3] as char;
                        let key = [a, b];

                        let value = &field[5..];
                        Some((key, (ty, value.to_string())))
                    } else {
                        None
                    }
                })
                .collect::<FxHashMap<_, _>>();

            let tgt_start = tgt_start.parse::<u64>().unwrap();

            let metadata = AlignmentMetadata {
                residue_matches,
                alignment_block_length,
                mapping_quality,
                optional_fields: opt_fields,
            };

            pairs.entry(pair).or_default().push((tgt_start, metadata));
        }

        let pairs = pairs.into_iter().map(|(key, mut fields)| {
            fields.sort_by_key(|(s, _)| *s);
            let fields = fields.into_iter().map(|(_, f)| f).collect::<Vec<_>>();
            (key, fields)
        });

        Ok(Self {
            metadata: pairs.collect(),
        })
    }
}

pub fn load_input_files(cli: &crate::cli::Cli) -> anyhow::Result<(Alignments, Sequences)> {
    use std::io::prelude::*;

    if let Some(impg_path) = cli.impg.as_ref() {
        let (alignments, mut sequences) = Alignments::from_impg(impg_path, &cli.paf)
            .map_err(|e| anyhow!("Error loading PAF and impg index: {e:?}"))?;

        if let Some(fasta_path) = cli.fasta.as_ref() {
            sequences
                .extract_sequences_from_fasta(fasta_path)
                .map_err(|e| anyhow!("Error loading sequences from FASTA: {e:?}"))?;
        }

        println!("using {} sequences", sequences.len());
        Ok((alignments, sequences))
    } else {
        let reader = std::fs::File::open(&cli.paf)
            .map(std::io::BufReader::new)
            .map_err(|e| anyhow!("Error opening PAF file: {e:?}"))?;

        let mut lines = Vec::new();
        let mut offset = 0u64;

        for line in reader.lines() {
            let line = line?;
            let len = line.as_bytes().len() as u64;
            let end = offset + len;
            let range = offset..end;
            offset += len;

            lines.push((line, range));
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

        let paf_lines = lines
            .iter()
            .filter_map(|(raw_line, file_range)| {
                let line = parse_paf_line(raw_line, file_range.start)?;
                filter_line(&line).then_some(line)
            })
            .collect::<Vec<_>>();

        let sequences = if let Some(fasta_path) = &cli.fasta {
            Sequences::from_fasta(fasta_path)
                .map_err(|e| anyhow!("Error building sequence index from FASTA: {e:?}"))?
        } else {
            Sequences::from_paf(&paf_lines).unwrap()
        };

        println!("using {} sequences", sequences.len());

        let alignments = Alignments::from_paf_lines(&sequences, paf_lines);

        Ok((alignments, sequences))
    }
}

impl Alignments {
    pub fn get(&self, index: AlignmentIndex) -> Option<&Alignment> {
        let al_indices = self.indices.get(&(index.target, index.query))?;
        let ix = al_indices.get(index.pair_index)?;
        self.alignments.get(*ix)
    }

    pub fn from_paf_lines<'l>(
        // NB: construct Sequences from iterator over PafLines (or FASTA) before
        sequences: &Sequences,
        lines: impl IntoIterator<Item = PafLine<&'l str>>,
    ) -> Self {
        let mut alignments = Vec::new();
        let mut indices: FxHashMap<_, Vec<_>> = FxHashMap::default();

        // let mut pairs: FxHashMap<_, Vec<_>> = FxHashMap::default();

        for paf_line in lines {
            let target_id = sequences.names().get_by_left(paf_line.tgt_name).unwrap();
            let query_id = sequences.names().get_by_left(paf_line.query_name).unwrap();
            let alignment = Alignment::new(&sequences.names(), &paf_line);
            let al_ix = alignments.len();

            let pair_id = (*target_id, *query_id);

            indices.entry(pair_id).or_default().push(al_ix);
            alignments.push(alignment);
            // pairs.entry(pair_id).or_default().push(alignment);
        }

        let mut cigar_range_index_map: bimap::BiHashMap<AlignmentIndex, std::ops::Range<u64>> =
            Default::default();

        for (&(target, query), al_indices) in indices.iter_mut() {
            al_indices.sort_by_key(|al_ix| {
                let loc = &alignments[*al_ix].location;
                let tgt = &loc.target_range;
                let qry = &loc.query_range;
                (tgt.start, tgt.end, qry.start, qry.end)
            });

            // for (pair_index, al_ix) in al_indices.iter_mut().enumerate()

            for (index, al) in alignments.iter_mut().enumerate() {
                let al_ix = AlignmentIndex {
                    target,
                    query,
                    pair_index: index,
                };
                if let Some(range) = al.cigar_file_byte_range.clone() {
                    cigar_range_index_map.insert(al_ix, range);
                }
            }
        }

        let indices = indices.into_iter().map(|(sp, als)| (sp, als)).collect();

        Self {
            alignments: Arc::new(alignments),
            indices: Arc::new(indices),
            // pairs: Arc::new(pairs),
            cigar_range_index_map,
        }
    }

    pub fn from_impg(
        // sequences: &Sequences,
        impg_path: impl AsRef<std::path::Path>,
        paf_path: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<(Self, Sequences)> {
        let impg_index = Arc::new(ImpgIndex::deserialize_file(impg_path, paf_path)?);

        let cigar_range_index_map: bimap::BiHashMap<AlignmentIndex, std::ops::Range<u64>> =
            Default::default();

        let sequences = Sequences::from_impg(&impg_index)?;

        let impg_cigars = ImpgIndex::impg_cigars(&impg_index, &sequences);

        let mut alignments = Vec::new();
        let mut indices: FxHashMap<_, Vec<_>> = FxHashMap::default();

        // let mut pairs: FxHashMap<_, Vec<_>> = FxHashMap::default();

        for (pair @ (target_id, query_id), impg_cigars) in impg_cigars.into_iter() {
            for impg_cg in impg_cigars {
                let tgt_seq = sequences.get(target_id).unwrap();
                let target_total_len = tgt_seq.len();

                let qry_seq = sequences.get(query_id).unwrap();
                let query_total_len = qry_seq.len();

                let location = AlignmentLocation {
                    target_total_len,
                    target_range: impg_cg.target_range.clone(),
                    query_total_len,
                    query_range: impg_cg.query_range.clone(),
                    query_strand: impg_cg.query_strand,
                };

                let alignment = Alignment {
                    target_id,
                    query_id,
                    location,
                    cigar_file_byte_range: Some(impg_cg.cigar_file_byte_range.clone()),
                    cigar: Arc::new(impg_cg),
                };

                let al_ix = alignments.len();
                alignments.push(alignment);
                indices.entry(pair).or_default().push(al_ix);
            }
        }

        Ok((
            Self {
                alignments: alignments.into(),
                indices: indices.into(),
                cigar_range_index_map,
            },
            sequences,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub cigar: Option<S>,
    pub cigar_file_range: Option<std::ops::Range<u64>>,
}

pub fn parse_paf_line_bytes<'a>(line: &'a [u8], line_offset: u64) -> Option<PafLine<&'a [u8]>> {
    use bstr::ByteSlice;

    let mut ranged_fields = line.split_str(b"\t").scan(line_offset, |offset, field| {
        let len = field.len() as u64;
        let range = *offset..len;
        *offset += len;
        Some((field, range))
    });

    let (query_name, query_seq_len, query_seq_start, query_seq_end) =
        parse_ranged_paf_name_range_bs(&mut ranged_fields)?;
    let strand = ranged_fields.next()?.0;
    let (tgt_name, tgt_seq_len, tgt_seq_start, tgt_seq_end) =
        parse_ranged_paf_name_range_bs(&mut ranged_fields)?;

    // let (cigar, cigar_file_range) = ranged_fields
    let cigar = ranged_fields
        .skip(3)
        .find_map(|(s, r)| Some((s.strip_prefix(b"cg:Z:")?, r)));

    let (cigar, cigar_file_range) = if let Some((cg, r)) = cigar {
        (Some(cg), Some(r))
    } else {
        (None, None)
    };

    Some(PafLine {
        query_name,
        query_seq_len,
        query_seq_start,
        query_seq_end,

        tgt_name,
        tgt_seq_len,
        tgt_seq_start,
        tgt_seq_end,

        strand_rev: strand == b"-",
        cigar,
        cigar_file_range,
    })
}

pub fn parse_paf_line<'a>(line: &'a str, line_offset: u64) -> Option<PafLine<&'a str>> {
    let mut ranged_fields = line.split('\t').scan(line_offset, |offset, field| {
        let len = field.len() as u64;
        let range = *offset..len;
        *offset += len;
        Some((field, range))
    });

    let (query_name, query_seq_len, query_seq_start, query_seq_end) =
        parse_ranged_paf_name_range(&mut ranged_fields)?;
    let strand = ranged_fields.next()?.0;
    let (tgt_name, tgt_seq_len, tgt_seq_start, tgt_seq_end) =
        parse_ranged_paf_name_range(&mut ranged_fields)?;

    // let (cigar, cigar_file_range) = ranged_fields
    let cigar = ranged_fields
        .skip(3)
        .find_map(|(s, r)| Some((s.strip_prefix("cg:Z:")?, r)));

    let (cigar, cigar_file_range) = if let Some((cg, r)) = cigar {
        (Some(cg), Some(r))
    } else {
        (None, None)
    };

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
        cigar_file_range,
    })
}

pub fn parse_paf_name_range<'a>(
    mut fields: impl Iterator<Item = &'a str>,
) -> Option<(&'a str, u64, u64, u64)> {
    let name = fields.next()?;
    let len = fields.next()?.parse().ok()?;
    let start = fields.next()?.parse().ok()?;
    let end = fields.next()?.parse().ok()?;
    Some((name, len, start, end))
}

fn parse_ranged_paf_name_range<'a>(
    mut fields: impl Iterator<Item = (&'a str, std::ops::Range<u64>)>,
) -> Option<(&'a str, u64, u64, u64)> {
    let name = fields.next()?.0;
    let len = fields.next()?.0.parse().ok()?;
    let start = fields.next()?.0.parse().ok()?;
    let end = fields.next()?.0.parse().ok()?;
    Some((name, len, start, end))
}

fn parse_ranged_paf_name_range_bs<'a>(
    mut fields: impl Iterator<Item = (&'a [u8], std::ops::Range<u64>)>,
) -> Option<(&'a [u8], u64, u64, u64)> {
    use bstr::ByteSlice;
    let name = fields.next()?.0;
    let parse = |fd: &[u8]| fd.to_str().ok().and_then(|s| s.parse::<u64>().ok());
    let len = parse(fields.next()?.0)?;
    let start = parse(fields.next()?.0)?;
    let end = parse(fields.next()?.0)?;
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

            cigar_file_byte_range: Some(0..cg_str.as_bytes().len() as u64),
        };

        for item in AlignmentIter::new(&alignment, 0..30) {
            println!("{item:?}");

            for step in item {
                println!("{step:?}");
            }
        }
    }
}
