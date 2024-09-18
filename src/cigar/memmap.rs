use std::sync::Arc;
use std::{collections::BTreeMap, ops::Deref};

use std::io::{prelude::*, BufReader};

use bevy::utils::HashMap;
use bgzip::index::BGZFIndex;

pub struct IndexedPaf {
    data: PafData,
    byte_index: PafByteIndex,

    bgzi: Option<BGZFIndex>,
}

impl IndexedPaf {
    pub fn for_each_paf_line<F>(&self, mut f: F)
    where
        F: FnMut(crate::PafLine<&[u8]>),
    {
        if let Some(bgzi) = self.bgzi.as_ref() {
            use bstr::io::BufReadExt;

            let data_slice = self.data.bytes();

            let reader = bgzip::BGZFReader::new(std::io::Cursor::new(data_slice)).unwrap();
            let mut reader = bgzip::read::IndexedBGZFReader::new(reader, bgzi.clone()).unwrap();

            let mut line_ix = 0;

            reader
                .for_byte_line(|line_slice| {
                    let line_offset = self.byte_index.record_offsets[line_ix];
                    if let Some(paf_line) =
                        crate::paf::parse_paf_line_bytes(line_slice, line_offset)
                    {
                        f(paf_line);
                    }
                    line_ix += 1;
                    Ok(true)
                })
                .unwrap();
        } else {
            for (line_ix, &line_offset) in self.byte_index.record_offsets.iter().enumerate() {
                let Some(line_slice) = self.data.paf_line_slice(&self.byte_index, line_ix) else {
                    continue;
                };

                let Some(paf_line) = crate::paf::parse_paf_line_bytes(line_slice, line_offset)
                else {
                    continue;
                };

                f(paf_line);
            }
        }
    }

    pub fn cigar_reader_iter(
        &self,
        line_index: usize,
    ) -> std::io::Result<CigarReaderIter<Box<dyn BufRead + '_>>> {
        let reader = self.cigar_reader(line_index)?;

        let cg_range = self
            .byte_index
            .record_offsets
            .get(line_index)
            .and_then(|offset| {
                let inner = &self.byte_index.record_inner_offsets.get(line_index)?;
                Some((offset + inner.cigar_range.start)..(offset + inner.cigar_range.end))
            })
            .ok_or(std::io::Error::other("PAF line not found in index"))?;
        let cigar_bytes_len = (cg_range.end - cg_range.start) as usize;

        Ok(CigarReaderIter {
            done: false,
            buffer: vec![0u8; 4096],
            offset_in_buffer: 0,
            cigar_bytes_len,
            bytes_processed: 0,
            bytes_read: 0,
            reader,
        })
    }

    fn cigar_reader<'a>(&'a self, line_index: usize) -> std::io::Result<Box<dyn BufRead + 'a>> {
        let data = match &self.data {
            PafData::Memory(arc) => arc.as_ref(),
            PafData::Mmap(shared_mmap) => shared_mmap.as_ref(),
        };

        let cg_range = self
            .byte_index
            .record_offsets
            .get(line_index)
            .and_then(|offset| {
                let inner = &self.byte_index.record_inner_offsets.get(line_index)?;
                Some((offset + inner.cigar_range.start)..(offset + inner.cigar_range.end))
            })
            .ok_or(std::io::Error::other("PAF line not found in index"))?;

        let reader = std::io::Cursor::new(data);

        if let Some(bgzi) = self.bgzi.as_ref() {
            let mut reader =
                bgzip::BGZFReader::new(reader).map_err(|e| std::io::Error::other(e))?;
            let pos = bgzi
                .uncompressed_pos_to_bgzf_pos(cg_range.start)
                .map_err(|e| std::io::Error::other(e))?;
            reader.bgzf_seek(pos);
            Ok(Box::new(reader))
        } else {
            let mut reader = reader;
            reader.seek(std::io::SeekFrom::Start(cg_range.start));
            Ok(Box::new(reader))
        }
    }
}

enum PafData {
    Memory(Arc<[u8]>),
    Mmap(SharedMmap),
}

impl PafData {
    fn bytes(&self) -> &[u8] {
        match self {
            PafData::Memory(arc) => arc.as_ref(),
            PafData::Mmap(shared_mmap) => shared_mmap.as_ref(),
        }
    }

    fn bytes_len(&self) -> usize {
        match self {
            PafData::Memory(arc) => arc.len(),
            PafData::Mmap(mmap) => mmap.0.len(),
        }
    }

    fn paf_line_slice(&self, index: &PafByteIndex, line_index: usize) -> Option<&[u8]> {
        let line_offset = *index.record_offsets.get(line_index)? as usize;
        let next_line_offset = if line_index < index.line_count() - 1 {
            *index.record_offsets.get(line_index + 1)? as usize
        } else {
            self.bytes_len()
        };

        let start = line_offset;
        let end = next_line_offset;

        let data = match self {
            PafData::Memory(arc) => arc.as_ref(),
            PafData::Mmap(shared_mmap) => shared_mmap.as_ref(),
        };

        Some(&data[start..end])
    }

    fn paf_cigar_slice(&self, index: &PafByteIndex, line_index: usize) -> Option<&[u8]> {
        let line_offset = index.record_offsets.get(line_index)?;
        let offsets = index.record_inner_offsets.get(line_index)?;
        let cg_range = &offsets.cigar_range;
        let start = (line_offset + cg_range.start) as usize;
        let end = (line_offset + cg_range.end) as usize;
        match self {
            PafData::Memory(arc) => {
                //
                Some(&arc[start..end])
            }
            PafData::Mmap(shared_mmap) => {
                //
                Some(&shared_mmap.as_ref()[start..end])
            }
        }
    }
}

#[derive(Clone)]
enum PafSource {
    File(std::path::PathBuf),
    Memory(Arc<[u8]>),
    Mmap(SharedMmap),
}

#[derive(Clone)]
struct SharedMmap(Arc<memmap2::Mmap>);

enum PafLineData {
    Owned(Vec<u8>),
    Shared {
        buffer: Arc<dyn AsRef<[u8]>>,
        // full_buffer: Arc<[u8]>,
        line_range: std::ops::Range<usize>,
    },
}

impl AsRef<[u8]> for SharedMmap {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl IndexedPaf {
    pub fn memmap_paf(
        path: impl AsRef<std::path::Path>,
        bgzi: Option<BGZFIndex>,
    ) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        // let reader = std::fs::File::open(&path).map(BufReader::new)?;
        let file = std::fs::File::open(&path)?;

        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        let byte_index = if let Some(bgzi) = bgzi.as_ref() {
            let reader = bgzip::BGZFReader::new(std::io::Cursor::new(&mmap))
                .and_then(|r| bgzip::read::IndexedBGZFReader::new(r, bgzi.clone()))
                .map_err(|e| std::io::Error::other(e))?;
            PafByteIndex::from_paf(reader)?
        } else {
            let reader = std::io::Cursor::new(&mmap);
            PafByteIndex::from_paf(reader)?
        };

        Ok(Self {
            data: PafData::Mmap(SharedMmap(Arc::new(mmap))),
            byte_index,
            bgzi,
        })
    }

    pub fn from_bytes_vec(data: Vec<u8>) -> std::io::Result<Self> {
        let reader = std::io::Cursor::new(data.as_slice());
        let byte_index = PafByteIndex::from_paf(reader)?;

        Ok(Self {
            data: PafData::Memory(data.into()),
            byte_index,
            bgzi: None,
        })
    }
}

pub struct IndexedPafReader<S: Seek + BufRead> {
    byte_index: PafByteIndex,
    data: S,
}

impl<S: Seek + BufRead> IndexedPafReader<S> {
    pub fn new(mut data: S) -> std::io::Result<Self> {
        data.rewind()?;
        let byte_index = PafByteIndex::from_paf(&mut data)?;
        Ok(Self { byte_index, data })
    }
}

struct PafByteIndex {
    record_offsets: Vec<u64>,
    record_inner_offsets: Vec<PafRecordIndex>,
}

// offsets are relative to the start of the record in the file
struct PafRecordIndex {
    cigar_range: std::ops::Range<u64>,
    optional_fields: BTreeMap<[u8; 2], u64>,
}

impl PafByteIndex {
    fn line_count(&self) -> usize {
        self.record_offsets.len()
    }
}

impl PafByteIndex {
    fn from_paf<R: BufRead>(mut paf_reader: R) -> std::io::Result<Self> {
        use bstr::{io::BufReadExt, ByteSlice};

        let mut record_offsets = Vec::new();
        let mut record_indices = Vec::new();

        let mut buffer = Vec::new();
        let mut offset = 0u64;

        let mut count = 0;

        loop {
            buffer.clear();
            let bytes_read = paf_reader.read_until(b'\n', &mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            let line_offset = offset;
            offset += bytes_read as u64;

            let line = buffer[..bytes_read].trim_ascii();
            count += 1;

            if let Some(index) = PafRecordIndex::from_line(line) {
                record_indices.push(index);
                record_offsets.push(line_offset);
            }
        }

        println!("recorded offsets for {} lines", record_offsets.len());

        Ok(Self {
            record_offsets,
            record_inner_offsets: record_indices,
        })
    }
}

impl PafRecordIndex {
    fn from_line(line: &[u8]) -> Option<Self> {
        use bstr::ByteSlice;
        let fields = line.split_str("\t");

        let mut offset = 0u64;

        let mut cigar_range = None;
        let mut optional_fields = BTreeMap::default();

        let mut opt_field_buf = Vec::new();

        for field in fields {
            let field_offset = offset;
            offset += field.len() as u64 + 1;

            opt_field_buf.clear();
            opt_field_buf.extend(field.splitn_str(3, ":"));

            if opt_field_buf.len() < 3 {
                continue;
            }

            if opt_field_buf[0].eq_ignore_ascii_case(b"cg") {
                // NB: adding 5 to the offset to skip the `cg:Z:`
                let cg_start = field_offset + 5;
                let cg_end = cg_start + field.len() as u64;
                cigar_range = Some(cg_start..cg_end);
            } else if let &[a, b] = opt_field_buf[0] {
                let key = [a, b];
                optional_fields.insert(key, field_offset);
            }
        }

        Some(Self {
            cigar_range: cigar_range?,
            optional_fields,
        })
    }
}

pub struct CigarReaderIter<S: BufRead> {
    done: bool,

    buffer: Vec<u8>,
    offset_in_buffer: usize,
    cigar_bytes_len: usize,

    bytes_processed: usize,
    bytes_read: usize,

    reader: S,
    // reader: BufReader
}

impl<S: BufRead> CigarReaderIter<S> {
    pub fn read_op(&mut self) -> Option<std::io::Result<(super::CigarOp, u32)>> {
        use bstr::ByteSlice;
        // dbg!();

        if self.done {
            return None;
        }
        // dbg!();

        if self.offset_in_buffer >= self.buffer.len() || self.bytes_processed == self.bytes_read {
            self.buffer.resize(4096, 0);
            self.offset_in_buffer = 0;
            match self.reader.read(&mut self.buffer) {
                Ok(bytes_read) => {
                    // println!("read {bytes_read} bytes");
                    self.bytes_read += bytes_read;
                }
                Err(err) => {
                    return Some(Err(err));
                }
            }
        }

        let buf_slice = &self.buffer[self.offset_in_buffer..];
        // let bs = bstr::BStr::new(&buf_slice[..(100.min(buf_slice.len()))]);

        // dbg!();
        let Some(op_ix) = buf_slice.find_byteset(b"M=XIDN") else {
            self.done = true;
            return None;
        };

        // dbg!();
        let Some(count) = buf_slice[..op_ix]
            .to_str()
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            self.done = true;
            return None;
        };

        let op_char = buf_slice[op_ix] as char;
        let op = super::CigarOp::try_from(op_char).unwrap();

        self.bytes_processed += op_ix;
        if self.bytes_processed >= self.cigar_bytes_len {
            self.done = true;
        }

        Some(Ok((op, count)))
    }
}

pub struct PafCigarIndex {
    // same order/size as the lines in corresponding `PafByteIndex`
    cigar_indices: Vec<CigarPositionIndex>,
}

// enum PositionIndexedCigar {
//     Index(CigarPositionIndex),
//     Short {
//         tgt_start: u64,
//         qry_start: u64,
//         ops: Vec<(super::CigarOp, u32)>,
//     },
// }

struct CigarPositionIndex {
    tgt_offsets: Vec<u64>,
    qry_offsets: Vec<u64>,
    cigar_byte_offsets: Vec<u64>,
}

impl CigarPositionIndex {
    fn iter(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        std::iter::zip(&self.tgt_offsets, &self.qry_offsets)
            .zip(&self.cigar_byte_offsets)
            .map(|((tgt, qry), cg)| (*tgt, *qry, *cg))
    }

    fn index_cigar_bytes<S: BufRead>(
        mut cigar_reader: CigarReaderIter<S>,
    ) -> std::io::Result<Self> {
        let mut tgt_offsets: Vec<u64> = Vec::new();
        let mut qry_offsets: Vec<u64> = Vec::new();
        let mut cigar_byte_offsets: Vec<u64> = Vec::new();

        let mut tgt_offset = 0;
        let mut qry_offset = 0;

        let mut ops_in_bin = 0;

        loop {
            let op_byte_offset = cigar_reader.bytes_processed as u64;

            let Some(op) = cigar_reader.read_op() else {
                break;
            };

            let (op, count) = op?;

            ops_in_bin += 1;

            let push_bin = ops_in_bin > 2 << 10;
            // let push_bin = false; // TODO

            if push_bin {
                tgt_offsets.push(tgt_offset);
                qry_offsets.push(qry_offset);
                cigar_byte_offsets.push(op_byte_offset);
                ops_in_bin = 0;
            }

            let len = count as u64;
            tgt_offset += op.consumes_target().then_some(len).unwrap_or(0);
            qry_offset += op.consumes_query().then_some(len).unwrap_or(0);
        }

        Ok(Self {
            tgt_offsets,
            qry_offsets,
            cigar_byte_offsets,
        })
    }

    // fn index_cigar(step_size: u64, ops: impl Iterator<Item = (super::CigarOp, u32)>) -> Self {
    //     let mut tgt = 0;
    //     let mut qry = 0;

    //     for (op, count) in ops {

    //         //
    //     }

    //     todo!();
    // }
}
