use std::sync::Arc;
use std::{collections::BTreeMap, ops::Deref};

use std::io::{prelude::*, BufReader};

use bevy::utils::HashMap;
use bgzip::index::BGZFIndex;
use bstr::ByteSlice;

pub struct IndexedPaf {
    data: PafData,
    pub byte_index: PafByteIndex,

    bgzi: Option<BGZFIndex>,
}

impl IndexedPaf {
    pub fn paf_line_slice(&self, line_index: usize) -> Option<&[u8]> {
        self.data.paf_line_slice(&self.byte_index, line_index)
    }

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

    fn cigar_reader_iter_indexed<'a>(
        &'a self,
        position_index_map: &HashMap<usize, CigarPositionIndex>,
        cigar_line_index: usize,
        target_range: std::ops::Range<u64>,
    ) -> std::io::Result<CigarReaderIter<Box<dyn BufRead + 'a>>> {
        let Some((line_offset, record_offsets)) = self
            .byte_index
            .record_offsets
            .get(cigar_line_index)
            .zip(self.byte_index.record_inner_offsets.get(cigar_line_index))
        else {
            return Err(std::io::Error::other(format!(
                "Cigar not found for line {cigar_line_index}"
            )));
        };
        let Some(pos_index) = position_index_map.get(&cigar_line_index) else {
            return Err(std::io::Error::other("Position index not found for cigar"));
        };

        let start_ix = pos_index.index_for_target(target_range.start);
        let end_ix = pos_index.index_for_target(target_range.end);

        let Some((start_ix, mut end_ix)) = start_ix.zip(end_ix) else {
            return Err(std::io::Error::other(format!(
                "Range `{}..{}` out of bounds of cigar",
                target_range.start, target_range.end
            )));
        };

        let start_ix = start_ix.checked_sub(1).unwrap_or(0);

        if start_ix == end_ix {
            end_ix = (end_ix + 1).min(pos_index.byte_offsets.len());
        }

        let byte_start = pos_index.byte_offsets[start_ix];
        let byte_end = pos_index.byte_offsets[end_ix];

        let data = match &self.data {
            PafData::Memory(arc) => arc.as_ref(),
            PafData::Mmap(shared_mmap) => shared_mmap.as_ref(),
        };

        let reader = std::io::Cursor::new(data);

        let cg_reader: Box<dyn BufRead> = if let Some(bgzi) = self.bgzi.as_ref() {
            let mut reader = bgzip::BGZFReader::new(reader).map_err(std::io::Error::other)?;
            let pos = bgzi
                .uncompressed_pos_to_bgzf_pos(byte_start)
                .map_err(std::io::Error::other)?;
            reader.bgzf_seek(pos).map_err(std::io::Error::other)?;
            Box::new(reader)
        } else {
            let mut reader = reader;
            reader.seek(std::io::SeekFrom::Start(byte_start))?;
            Box::new(reader)
        };

        let cigar_bytes_len = (byte_end - byte_start) as usize;

        let mut cg_iter = CigarReaderIter::new(cg_reader, cigar_bytes_len);
        cg_iter.fill_buffer()?;

        let tgt_start = pos_index.target_offsets[start_ix];

        cg_iter.target_pos = tgt_start;
        cg_iter.query_pos = pos_index.query_offsets[start_ix];

        let mut skip_count = 0;
        // skip inside the block to the correct starting block
        if tgt_start < target_range.start {
            println!("skipping inside first block");
            loop {
                {
                    let buf_hd = bstr::BStr::new(&cg_iter.buffer[..200]);
                    println!("buffer: {buf_hd}");
                    println!(
                        "internal offset: {}\tprocessed: {}\tcigar byte len: {}",
                        cg_iter.offset_in_buffer, cg_iter.bytes_processed, cg_iter.cigar_bytes_len
                    )
                }
                let Some((op, count)) = cg_iter.get_current_op() else {
                    dbg!();
                    break;
                };

                let tgt_delta = op.target_delta(count) as u64;
                let tgt_pos = cg_iter.target_pos;

                println!("{tgt_pos} + {tgt_delta} cf. {}", target_range.start);

                if tgt_pos < target_range.start && tgt_pos + tgt_delta > target_range.start
                    || tgt_pos == target_range.start
                {
                    dbg!();
                    break;
                }

                dbg!();
                cg_iter.next_op()?;
                // cg_iter.step();
                skip_count += 1;
            }
        }
        println!("skipped {skip_count} ops");

        cg_iter.target_end = Some(target_range.end);

        Ok(cg_iter)
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
                let cigar_range = inner.cigar_range.as_ref()?;
                Some((offset + cigar_range.start)..(offset + cigar_range.end))
            })
            .ok_or(std::io::Error::other("PAF line not found in index"))?;
        // println!("iterating cigar {line_index} from byte range {cg_range:?}");
        let cigar_bytes_len = (cg_range.end - cg_range.start) as usize;

        Ok(CigarReaderIter::new(reader, cigar_bytes_len))
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
                let cigar_range = inner.cigar_range.as_ref()?;
                Some((offset + cigar_range.start)..(offset + cigar_range.end))
            })
            .ok_or(std::io::Error::other("PAF line not found in index"))?;

        let reader = std::io::Cursor::new(data);

        if let Some(bgzi) = self.bgzi.as_ref() {
            let mut reader = bgzip::BGZFReader::new(reader).map_err(std::io::Error::other)?;
            let pos = bgzi
                .uncompressed_pos_to_bgzf_pos(cg_range.start)
                .map_err(std::io::Error::other)?;
            reader.bgzf_seek(pos).map_err(std::io::Error::other)?;
            Ok(Box::new(reader))
        } else {
            let mut reader = reader;
            reader.seek(std::io::SeekFrom::Start(cg_range.start))?;
            Ok(Box::new(reader))
        }
    }
}

pub struct MmapCigar {
    pub target_range: std::ops::Range<u64>,
    pub query_range: std::ops::Range<u64>,
    pub query_strand: super::Strand,

    paf: Arc<IndexedPaf>,
    paf_line: usize,
}

impl MmapCigar {
    pub fn new(
        paf: Arc<IndexedPaf>,
        paf_line: usize,
        target_range: std::ops::Range<u64>,
        query_range: std::ops::Range<u64>,
        query_strand: super::Strand,
    ) -> Self {
        Self {
            target_range,
            query_range,
            query_strand,
            paf,
            paf_line,
        }
    }
}

impl super::IndexedCigar for MmapCigar {
    fn is_empty(&self) -> bool {
        let inner = self.paf.byte_index.record_inner_offsets.get(self.paf_line);
        inner
            .and_then(|record| {
                let r = record.cigar_range.as_ref()?;
                Some(r.end == r.start)
            })
            .unwrap_or(true)
    }

    fn whole_cigar(&self) -> Box<dyn Iterator<Item = (super::CigarOp, u32)> + '_> {
        Box::new(
            self.paf
                .cigar_reader_iter(self.paf_line)
                .unwrap()
                .map(|item| (item.op, item.op_count)),
        )
    }

    fn iter_target_range(
        &self,
        target_range: std::ops::Range<u64>,
    ) -> Box<dyn Iterator<Item = super::CigarIterItem> + '_> {
        let mut iter = self.paf.cigar_reader_iter(self.paf_line).unwrap();
        iter.target_start = Some(target_range.start);
        iter.target_end = Some(target_range.end);

        iter.fill_buffer().unwrap();
        loop {
            if iter.target_pos >= target_range.start {
                break;
            }

            let Some((op, count)) = iter.get_current_op() else {
                break;
            };

            let tgt_delta = op.target_delta(count) as u64;

            if iter.target_pos + tgt_delta > target_range.start {
                break;
            }
            let _ = iter.next_op();
        }

        Box::new(iter)
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

pub struct PafByteIndex {
    pub record_offsets: Vec<u64>,
    pub record_inner_offsets: Vec<PafRecordIndex>,
}

// offsets are relative to the start of the record in the file
#[derive(Debug)]
pub struct PafRecordIndex {
    pub cigar_range: Option<std::ops::Range<u64>>,
    pub optional_fields: BTreeMap<[u8; 2], u64>,
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
                let cg_end = cg_start + opt_field_buf[2].len() as u64;
                cigar_range = Some(cg_start..cg_end);
            } else if let &[a, b] = opt_field_buf[0] {
                let key = [a, b];
                optional_fields.insert(key, field_offset);
            }
        }

        Some(Self {
            cigar_range: cigar_range,
            optional_fields,
        })
    }
}

enum PafData {
    Memory(Arc<[u8]>),
    Mmap(SharedMmap),
}

#[derive(Clone)]
struct SharedMmap(Arc<memmap2::Mmap>);

impl AsRef<[u8]> for SharedMmap {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl PafData {
    pub fn bytes(&self) -> &[u8] {
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

    pub fn paf_line_slice(&self, index: &PafByteIndex, line_index: usize) -> Option<&[u8]> {
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

    // fn paf_cigar_slice(&self, index: &PafByteIndex, line_index: usize) -> Option<&[u8]> {
    //     let line_offset = index.record_offsets.get(line_index)?;
    //     let offsets = index.record_inner_offsets.get(line_index)?;
    //     let cg_range = &offsets.cigar_range;
    //     let start = (line_offset + cg_range.start) as usize;
    //     let end = (line_offset + cg_range.end) as usize;
    //     match self {
    //         PafData::Memory(arc) => {
    //             //
    //             Some(&arc[start..end])
    //         }
    //         PafData::Mmap(shared_mmap) => {
    //             //
    //             Some(&shared_mmap.as_ref()[start..end])
    //         }
    //     }
    // }
}

pub struct CigarReaderIter<S: BufRead> {
    cigar_bytes_len: usize,
    reader: S,
    done: bool,

    target_pos: u64,
    query_pos: u64,

    target_start: Option<u64>,
    target_end: Option<u64>,
    // query_start: Option<u64>,
    query_end: Option<u64>,

    buffer: Vec<u8>,
    buffer_bytes_used: usize,

    offset_in_buffer: usize,
    bytes_processed: usize,
}

impl<S: BufRead> CigarReaderIter<S> {
    const DEFAULT_BUFFER_SIZE: usize = 4096;

    fn new(reader: S, cigar_bytes_len: usize) -> Self {
        let buffer_len = Self::DEFAULT_BUFFER_SIZE.min(cigar_bytes_len);

        Self {
            cigar_bytes_len,
            reader,

            target_pos: 0,
            query_pos: 0,
            target_start: None,
            target_end: None,
            // query_start: None,
            query_end: None,

            done: false,
            buffer: vec![0u8; buffer_len],
            buffer_bytes_used: 0,
            offset_in_buffer: 0,
            bytes_processed: 0,
        }
    }

    // fills the internal buffer, ensuring that the next op is at the start of the buffer
    // by copying the remainder if necessary
    fn fill_buffer(&mut self) -> std::io::Result<()> {
        let remainder_range = self.offset_in_buffer..self.buffer_bytes_used;

        let mut tgt_offset = 0;

        if self.buffer_bytes_used > self.offset_in_buffer {
            tgt_offset = remainder_range.end - remainder_range.start;
            self.buffer.copy_within(remainder_range.clone(), 0);
        }

        let read_len = self.reader.read(&mut self.buffer[tgt_offset..])?;
        self.buffer_bytes_used = read_len + tgt_offset;
        self.offset_in_buffer = 0;

        Ok(())
    }

    fn get_current_op(&self) -> Option<(super::CigarOp, u32)> {
        let op_ix = self.buffer[self.offset_in_buffer..].find_byteset(b"M=XIDN")?;
        let buf_slice = &self.buffer[self.offset_in_buffer..];

        let count = buf_slice[..op_ix]
            .to_str()
            .ok()
            .and_then(|s| s.parse::<u32>().ok())?;

        let op_char = buf_slice[op_ix] as char;
        let op = super::CigarOp::try_from(op_char).ok()?;
        Some((op, count))
    }

    // step through the internal buffer, parsing the next op and its length and emitting it.
    // doesn't perform any read or touch the internal buffer
    // sets `self.done = true` if the entire cigar has been processed:
    // if this returns `None` and `self.done == false`, the buffer must be filled again
    // before parsing the next op, but if `self.done` has been set to `true` the iteration
    // is complete
    fn parse_next_op(&mut self) -> Option<(super::CigarOp, u32)> {
        let op_ix = self.buffer[self.offset_in_buffer..].find_byteset(b"M=XIDN")?;
        let buf_slice = &self.buffer[self.offset_in_buffer..];

        let count = buf_slice[..op_ix]
            .to_str()
            .ok()
            .and_then(|s| s.parse::<u32>().ok())?;

        let op_char = buf_slice[op_ix] as char;
        let op = super::CigarOp::try_from(op_char).ok()?;

        self.offset_in_buffer += op_ix + 1;
        self.bytes_processed += op_ix + 1;
        self.target_pos += op.target_delta(count) as u64;
        self.query_pos += op.query_delta(count) as u64;

        let target_done = self
            .target_end
            .map(|e| self.target_pos >= e)
            .unwrap_or(false);
        let query_done = self.query_end.map(|e| self.query_pos >= e).unwrap_or(false);

        if self.bytes_processed >= self.cigar_bytes_len || target_done || query_done {
            println!("target_pos: {}", self.target_pos);
            println!("target_done: {target_done}\tquery_done: {query_done}");
            self.done = true;
        }

        Some((op, count))
    }

    pub fn next_op(&mut self) -> std::io::Result<Option<(super::CigarOp, u32)>> {
        if self.done == true {
            return Ok(None);
        }

        let next = self.parse_next_op();

        if !self.done && next.is_none() {
            self.fill_buffer()?;
            // dbg!();
            return Ok(self.parse_next_op());
        }

        Ok(next)
    }
}

impl<S: BufRead> Iterator for CigarReaderIter<S> {
    type Item = super::CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        let tgt_start = self.target_pos;
        let qry_start = self.query_pos;
        if let Ok(next) = self.next_op() {
            let item = next.map(|(op, len)| {
                let tgt_end = tgt_start + op.target_delta(len) as u64;
                let qry_end = qry_start + op.query_delta(len) as u64;

                let mut target_range = tgt_start..tgt_end;
                let mut query_range = qry_start..qry_end;

                let mut op_count = len;

                if let Some(start_bound) = self.target_start {
                    if target_range.start < start_bound {
                        let clipped = start_bound - target_range.start;
                        target_range.start += clipped;
                        op_count -= clipped as u32;

                        if op.consumes_query() {
                            query_range.start += clipped;
                        }
                    }
                }

                if let Some(end_bound) = self.target_end {
                    if target_range.end > end_bound {
                        let clipped = target_range.end - end_bound;
                        target_range.end -= clipped;
                        op_count -= clipped as u32;

                        if op.consumes_query() {
                            query_range.end -= clipped;
                        }
                    }
                }

                super::CigarIterItem {
                    target_range,
                    query_range,
                    op,
                    op_count,
                }
            });

            return item;
        } else {
            self.done = true;
            return None;
        }
    }
}

// provides a cigar position (target (and/or query?) offset) to byte offset map,
// used to seek in the PAF line bytes to iterate only part of a cigar
#[derive(Debug)]
struct CigarPositionIndex {
    target_len: u64,
    query_len: u64,

    target_offsets: Vec<u64>,
    query_offsets: Vec<u64>,
    byte_offsets: Vec<u64>,
}

impl CigarPositionIndex {
    fn index_for_target(&self, target_offset: u64) -> Option<usize> {
        if target_offset >= self.target_len {
            return None;
        }
        let ix = self.target_offsets.partition_point(|&o| o < target_offset);
        Some(ix)
    }
}

impl CigarPositionIndex {
    fn index_cigar(
        //
        min_byte_distance: u64, // prob. want to use # of ops instead
        cigar: impl Iterator<Item = (super::CigarOp, u32)>,
    ) -> Self {
        // let mut target_offsets: Vec<u64> = Vec::new();
        // let mut query_offsets: Vec<u64> = Vec::new();
        // let mut byte_offsets: Vec<u64> = Vec::new();
        let mut target_offsets = vec![0u64];
        let mut query_offsets = vec![0u64];
        let mut byte_offsets = vec![0u64];

        let mut tgt_offset = 0u64;
        let mut qry_offset = 0u64;
        let mut byte_offset = 0u64;

        for (op, len) in cigar {
            let digits = len.ilog10();
            let byte_delta = (1 + digits) as u64;

            let tgt_delta = op.consumes_target().then_some(len as u64).unwrap_or(0);
            let qry_delta = op.consumes_query().then_some(len as u64).unwrap_or(0);

            let last_byte_end = byte_offsets.last().copied().unwrap_or(0);
            let next_byte_end = byte_offset + byte_delta;

            let byte_dist = byte_offset + byte_delta - last_byte_end;
            println!("byte dist {byte_dist}");

            if byte_offset + byte_delta - last_byte_end > min_byte_distance {
                target_offsets.push(tgt_offset);
                query_offsets.push(qry_offset);
                byte_offsets.push(byte_offset);
            }

            tgt_offset += tgt_delta;
            qry_offset += qry_delta;
            byte_offset += byte_delta;
        }

        target_offsets.push(tgt_offset);
        query_offsets.push(qry_offset);
        byte_offsets.push(byte_offset);

        let result = Self {
            target_len: tgt_offset,
            query_len: qry_offset,

            target_offsets,
            query_offsets,
            byte_offsets,
        };

        result
    }
}

/*
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
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    const CG1: &'static [u8] = b"107=1X89=1X5=1X1=1X56=1X5=1D34=1X8=1X36=1X8=2X42=1X486=1X10=1X10=1X45=1D1X14=1X85=1X106=1X11=1X5=1X5=1X123=1X55=1X140=1X6=1X85=1X9=1X35=1X3=1X18=8I15=3D15=1X25=8D1X67=1X52=1X9=1X14=1X1=1X2=1X27=1X5=1X129=1X21=1X43=1X282=1X53=1X823=1I5=1X134=1X5=1X2=1I28=1X1=1X360=1X4=1X62=1X56=1X212=1X9=1X32=1X358=12I597=1X14=1X20=1X1=1X56=1X2=1X94=2X387=1X209=1D1892=1X357=1X221=14D37=1X195=3D466=1X101=1X46=1X128=1D421=1X39=1X342=1X20=1X941=1X353=3D120=1X339=1X142=1X469=1D1050=1X464=1X600=3I29=1X210=1X54=9D17=1X3=4X50=2I16=1X26=2D6=1X57=2D1=1X3=1X35=1X12=1X6=1I59=2D99=1X1=1X45=1X23=1X11=5I12=1D104=1D21=1X2=1X34=5I2X23=1X28=1X5=1X15=2X19=1X10=1X25=1X17=1X46=1X60=1X9=1X31=1X1=1X3=1X24=1X5=1X11=1X60=1X5=1X16=1X20=1X38=19D1=1X3=1X7=1X11=1X19=1X4=1X2=1X2=1X2=1X5=1X33=1X10=1X24=1X2=1X11=1X5=1X11=1X29=1X17=1X2=1X39=1X3=1X1=1X3=1X21=1X1=1X25=1X6=4X2=1I14=1X2=1X47=1X26=1X25=2X1=1X7=1X1=1X1=1D1X2=1X10=1X25=2X9=4I1X4=2X18=1X1=1I1X34=4I1X2=1X1=1X31=1X2=1X2=1I4=1X10=1X32=1X5=1X47=1X6=1X8=1X2=1X9=2X31=1X24=1X6=1X38=1X6=1X21=1X34=1X3=1X2=1X20=1X3=1X15=1X4=2X11=1X42=1X63=1I1X6=1X5=1X11=1X8=1I20=1X4=4I2=1X4=1D23=1X9=1X2=2X3=1X6=1X14=1X2=1X2=1X12=1X13=1D7=2X6=1X8=2X3=1X24=1X2=1X1=1X2=1X5=2D4=1X17=1X3=1X18=1X1=1X4=1D13=8I6=1X5=2X1=1X1=1X3=1X1=1X1=1X2=1X13=1X7=1X3=1X1=1X38=1X39=3X25=2X6=1X1=1X2=1X2=1X32=1I2=1X8=1X9=1X20=1X20=1X37=1X107=12D43=1D13=1X52=1X7=2X3=1X2=1X21=1X17=1X12=1X3=1X7=1X2=1X7=1X1=1X128=1X25=2X2=1X75=1X39=1X63=1X8=1X17=1X50=1X203=1X37=1X14=1X52=1X25=1X74=1X7=1X15=1X137=1X62=1X128=1X37=1X26=1X8=1X27=1X6=1X32=1X4=2X46=2X70=1X45=1X6=1X3=1X6=1X17=1X11=1X31=1X64=1X1=1X5=1X7=1X20=1X6=1X2=3X2=1X44=1X32=1X31=1X3=1X10=6I10=1X9=1X9=1X6=1X2=1X10=45D35=1X9=1X42=1X4=1X27=1X21=1X113=1D22=1X35=1X34=1I1X36=1X17=1X5=1X25=1X1=1X31=1X44=1X16=1I4=1X46=1X5=1X2=3D27=1D36=1X21=1X30=1X7=1X13=26I1=1X1=1X3=1X10=1X4=3X19=2D6=1X4=2X5=2I2=1X1=2X2=1X4=6I1=1X4=1I4=2D2=2X13=1X7=1I15=1X2=1X2=2X12=37D1=1X5=1D3=3D2=1X6=2X17=2X5=1X5=1X6=1X2=1X4=5I4=1X4=1X11=1X3=1X1=1X5=2X16=1X1=2X7=4D3=1X14=1D5=1X1=1X4=2X11=1X4=1D1X4=1X4=1X1=2X4=1X5=1X1=1X3=3D3=1X4=1X2=1X1=1X10=1X5=1X4=1I11=2X3=1X1=1D4=1X25=243D4=1X1=1D2=1X1=1X2=1X3=3I1=4X1=2X2=2X1=3X2=1X3=1X2=1D3=7D4X5=3X2=1X1=2X1=1X2=1X1=1X2=1X2=1X1=1X3=1X1=1X1=4D1X3=2X1=2X1=2X2=1I1=1X1=3X2=3X2=2X1=1X3=1D2=1X2=2X1=1X4=2D1X3=4D5=2X1=1X1=8D2=1X3=4X2=1X3=2D1=2X2=4X3=2X1=4D1X3=1X1=1D1=4X1=1X4=1X3=1D2X1=1X1=2X2=2X2=1X3=2D5=3D2=1X5=3X1=1X2=5D1=1X1=1X2=1X1=1X4=1D2X2=2X2=2X1=1X2=5X1=1I1X1=2X2=3X3=2X1=4I3=1X7=1X2=2X12=1X3=1X7=1X2=1X4=1X11=1X2=1X1=1X12=1X4=1D1=1X20=1X9=2X8=1X5=2X28=20I6=1D24=1X25=1X1=29I14=2I12=1X10=1X21=1X2=1X13=1X6=1X3=1X15=1X12=908I6=2I26=1X12=1X19=1X3=1X39=1X14=1X1=2X9=1I9=1X9=2I1=1X3=2X26=1X2=1D22=6I1X12=1X25=1X63=1X27=1X5=2I111=1X283=1X56=17D24=12D9=1X53=1X162=1X225=1X108=1X9=11I4=1X144=1X68=1X1=1X134=1X181=3D124=1X18=1X29=1D68=1X72=1X26=1X22=1X68=1D189=1X13=1X8=1X10=2X1=1X2=1X1=9I1X2=1X3=2X2=4I1=1X1=1X2=1D1X1=1X1=2X1=1X70=1X156=1X39=2X68=1X16=1X50=1X58=1X9=1X32=1X24=1X4=1X77=1X8=1X6=1X4=2X36=1X11=1X120=1X14=1X54=1X1=2X6=3D13=1X23=7I8=1X15=1X3=1X25=1X1=3X1=1X1=3X1=1X2=1X6=1X29=21I18=1X7=1X4=1X50=1X2=1X1=1X13=1X25=1X170=1X9=1X24=2X28=1X144=1X178=1X38=1X33=1X44=1X152=1X35=1X36=1X10=1X2=1X15=1X1=1X17=1X9=1X7=1X20=1X2=1X7=1X1=1X12=1X54=1X188=1X38=1X26=1X5=1X425=1X10=1X11=1X25=1X2=26D19=1X5=1D1=1X3=1X5=1X35=1X49=1X8=1X5=1X8=1X8=1X8=1X17=1X8=1X98=1X95=1X85=1X37=1X83=1X48=6D2=1X3=1X17=2X8=1X9=1X14=2I7=1X23=1X128=1X27=1X16=1X53=9I13=1X4=1I45=1X18=4D15=1X11=1X18=1I46=1X16=1X1=1X2=1X1=1X7=1X11=1I9=1X4=1X1=1X3=1X2=1X5=1I5=1X3=1X2=1X5=2X5=1X7=2X1=1X10=1X5=1X5=4D1X5=1X42=1X14=1X10=2I15=1X7=1X2=1X21=1X3=1X25=1X10=1X15=1X23=1X12=8I53=1I1=1X4=1X39=1X178=1X36=1X48=1X177=1X25=1X87=3D105=1X19=1X24=1X106=4D2=1X1=1X7=1X4=1X4=5I18=1X150=1X5=1X69=1X2=1X11=1X39=1X113=1I1=1X6=1X1=1X2=2D10=1D20=1X77=1X33=1X11=1X128=1X679=1X266=1X172=1X154=8D162=1X89=1X75=1X42=1X60=1X85=1X16=1X144=1X50=2D9=1X51=1X41=1X30=2D35=1D12=1X4=1X14=1X190=1X68=1X59=1X14=1X5=1X17=1X23=1X41=1X2=1X29=1X2=1X5=1X8=1X41=1X2=1X182=1X56=1X20=1X26=1X5=1X74=1X59=1X5=1X20=1X11=1X5=1X59=1X14=1X17=1X77=1X8=1X38=1X573=1X10=1D4=1X8=1D12=2I5=1X44=1X5=1X9=1X3=1X15=1X1=1X3=1X1=1X5=1D4=3X1=4X5=454D3=1X2=1X1=1X1=1X5=1X1=1X3=1D1X1=1X1=1X1=1X1=1X1=3I8=1X8=1X4=1X4=1X10=1X75=1X20=1X1=1X67=1X12=1X23=1X24=2X3=1X4=1X26=1X18=1X15=1X3=1X18=1X15=1X3=1X9=1X1=1X3=1X4=1X4=1X3=1X3=1X15=1X2=6D1=1X16=1X2=1D9=1X6=4I13=1X25=1X16=1X41=1X55=1X12=1X3=1X18=1X2=1X8=15I32=2X5=1X46=2I1X1=2X40=1X19=1X11=1I16=1X9=1X7=1X54=1X27=1X2=1X10=1X8=1X40=1X10=1X7=1D1X71=1X16=1X9=1X1=1X2=1X10=1X4=1X8=1I23=1X46=1X5=1X3=1X58=2D11=1X8=1X8=1X9=1X18=1X13=1X38=1X4=1X4=1X17=1X14=1X28=1X7=1X3=1X1=2D1=2X3=1X28=1X19=1X4=1X9=1X7=1X5=1X6=1X8=1X7=1X2=1X2=1X15=1X5=1X3=21D1X3=1X4=1X10=1X88=1X72=1X10=1X33=2X4=1X26=1X4=1X3=1X5=1X7=1X4=1X19=1X6=17I1X1=1X31=1X3=1X31=1X2=1X2=1X11=1X2=1X73=1X26=1X9=1X1=1X31=3I1=1X35=1X3=1X8=1X17=1X18=1D4=1X17=1X1=1X3=1X6=1X5=1X1=1X3=1X5=1D13=1X8=105D6=1X1=1X9=1X46=1X15=1X14=1X2=1X1=1X1=1X24=1X11=1X6=1X4=3X5=4I4=1X37=2X48=2X3=2X10=1X14=1X14=1X17=1X14=1X58=1X4=1X35=1X10=1X16=1X3=1X23=1X14=1X4=1X11=1D5=1X4=1X3=1D8=1X31=1X43=1X2=1X6=1X46=1X11=1X26=2X19=1X29=1X15=1X3=1X15=1X14=1X17=1X20=1X5=1X35=1X13=1X19=1X8=1X22=1X2=1X2=1X47=2D1X25=1X16=1D177=1X34=1X124=1X162=1X99=1X1=1X18=1D120=1X82=1X210=1X18=1X137=1X8=1X92=1X70=1X17=2D99=1X54=1X53=1X2=1X2=1X73=1X91=1X62=1X61=1X22=1X12=1X81=1X138=1X36=1X8=1X72=1X24=1X28=1X145=1X10=1X47=1X11=1X18=1X3=1X57=1X7=2X17=1X29=1X142=1X4=1D30=1X28=1X26=1X23=1X81=1X69=1X5=1X22=1X75=1X91=1X108=1X53=2X24=1X121=1X62=1X54=1X38=1X6=2X10=1X9=1X180=1X100=1X39=1X160=1X44=1X63=1X40=1D101=1X22=1X43=1X73=1X43=1X83=1X178=1X25=1X296=1X109=1X39=2X123=1X16=1X93=1X65=2X1=1X35=1X77=1X11=1X41=1X8=1X31=1X2=1X8=1X24=2X1=2X1=1X11=1D1=1X6=1X8=1X20=1X53=1X11=2X10=3D4=1X5=1X4=1X41=1X9=1D7=1D106=1X25=1X111=1X10=1X22=1X90=1X16=1X32=1X11=1X26=1X24=1X36=30D2X65=1X4=1X59=1X32=1X1=6D1X15=17D4=2X8=1X34=2D14=1X2=1X1=1X32=1I1X3=1X42=1X10=8I3=1X5=1X7=1X11=10D6=1X8=1X9=1I17=1X14=1X9=1X1=3D1=1X8=1X11=1X10=1X5=1X1=1X1=1X11=1X6=1X28=1X7=1X10=1X4=1X13=3X4=3D4=1X3=9I5=1X1=1X12=1X26=1X33=1X22=1X17=1X7=1X6=1X13=1X6=1X17=1X17=1X1=1X3=1X2=1X14=1X17=1X22=2X39=3I1=2X5=1X86=1X2=1X34=1X7=1X4=3X3=2X8=1X259=1X20=1X33=1X114=1X22=1X90=1X39=1X6=1X12=1X2=1X30=1X1=1X11=1X9=1X2=2X3=1X7=1X42=2X5=2X5=1X16=1X79=1X21=1X43=1X2=1X3=1X16=1X1=2X52=1X5=1X45=1X9=4I4=1X9=1X13=1X36=1X12=1X14=1X3=2X29=1X2=1X13=1X6=1X1=1X7=1X8=1X6=1X12=1X11=5I12=9I47=5D43=1X3=1X3=1X2=1D3=1X18=1X7=1X1=2I24=1X4=1X6=5D9=2X10=2X13=4D8=7I5=164I2X1=2X5=5X1=1X6=1X6=1X22=1X19=1I23=766D2=1X27=1X14=1X2=1X8=1X18=1X7=1X16=1X28=4D9=1X61=2D70=7I21=1X46=1D7=1X13=2D36=1D8=1X7=1X1=1X28=1X11=1X102=1X2=1X34=1X17=43I1X1=1X57=1X14=1X18=2X72=2X71=1X89=1X97=1X7=";

    const CG2: &'static [u8] = b"1=22D34=1X6=4D22=1X10=1X7=1X24=1X3=1X52=2X38=1X8=1X22=1X101=1X56=1X27=1X10=1X14=1X23=1X11=1X17=1X85=1X96=1X100=1X165=1X23=1X203=1X62=1X29=1X26=1X47=1X17=3I75=1X171=1X85=1X2=1X37=1X6=1X5=1X2=1X10=1X30=1X23=1X11=1X62=1X233=1X82=1X3=1X87=1D12=1X8=1D20=1X9=1X9=1X51=1X37=1X24=1X320=1X57=1X171=1X513=3D210=8I21=1D67=1X12=1X9=13I90=1D18=1X1=1D3=5X4=2X4=2I2=2I40=1X12=1X36=1I8=1I28=1X17=1X81=62I28=1X102=5D10=2X3=1X2=1X24=1X14=1X11=1X3=1X16=1X5=1X33=1X77=1X192=1D45=29D4=1X105=1X3=1X8=1X67=1X17=1X19=1X9=60D5=1X1=1X5=1D8=1D2=1X1=1X22=1X9=1X7=1X11=1X2=1X17=1X9=2I15=1X8=2X3=10D3=1X11=1X2=1X10=3X15=1X17=164I3=1X2=2X3=4I1=1X5=1X16=4X2=1X11=1I2=4X8=1X1=1X1=2X1=1X4=1X4=1X2=1X7=1I2X7=1X6=1X3=13D1X4=1X1=1X2=1X3=1X2=1X4=1I9=1X1=1X8=1X5=2X3=1X35=1X7=1X2=1X13=1I20=1X15=4D1=1X35=1I12=1X2=1X1=1X4=1X108=1X209=1X227=1X2=1X75=1X158=1X29=1X75=1X57=1X118=7I1=1X28=1X2=1X23=1X12=28D13=1X2=1X9=1X44=1X9=1X6=1X1=3I1X19=17I4=1X6=1X6=1X11=1X8=16D53=1X244=1X167=1X630=1X26=1X93=1X176=1X247=1X54=1X86=1X253=1X14=1X1363=1X554=1X212=1X579=1X119=1X631=1X1039=1X65=1X534=1X906=1X1128=1X4=1X22=1X75=1X101=2X94=1X317=1X46=4D12=1X29=1I79=1X42=1X126=1X9=1X392=1X154=1X24=1X43=9I4=1X69=1X4=1X88=8I111=1X3=1X127=1X122=1X178=3D11=1X4=1D67=1X77=1X4=1X60=1X25=1X10=1X90=1X1=5D17=1X15=8I20=23I1=1X1=1X34=1X7=1X5=1X91=1X10=1X12=1X78=1I20=1X27=1X60=1X45=4D35=1X16=1D12=1X40=1X12=1X79=1X2=1X13=1X49=1X19=2X32=1D4=1X18=1X31=2X25=1X5=1X14=26D30=1X43=1X4=1X16=1X61=1X11=1X5=27I92=1X33=5D1X9=1X6=1X6=1X6=1X23=1X103=3I36=1X51=1X21=1X22=1X5=1X5=1X3=2I3X2=1I2=2X1=1X4=1X2=1X1=3X2=1X4=2X1=2X1=1X2=41D2=1I26=1X15=1X9=1X6=1X6=1X12=12D169=1X29=1X247=1X12=1X75=1X20=1X5=5I23=1X22=1X23=1D24=1X24=1I80=1X11=1X226=1D124=1X59=1X17=1X27=1X38=1X181=1X34=1X73=1X19=1X16=282D57=1X7=1X107=1X7=1X29=1X1=3D253=1X58=1X120=1X149=1X231=1X128=1X23=2X87=1X643=1X24=1X118=1I14=1X87=1X14=15D194=2X28=1X44=1X28=1X96=1X116=2X32=2I7=1X11=1X75=1X83=1X2=1X41=1X26=1X4=1X73=1X49=913I71=1X54=1I50=1X19=1X69=1X2=1X61=1X1=1X40=1X69=1X269=11I67=1X3=1X12=1X29=1X875=1X57=1X16=1X5=1X23=1X1=1X100=1X2=1X8=1X2=1X14=20D12=1X7=1X21=2X19=1I12=1X118=1X103=1X74=1X46=1X17=1X22=1X131=1X16=2D1=1X14=1X183=1X3=1X6=2X54=1X131=1X12=1X270=1X25=1X494=1X22=1X5=1X143=1X15=1X50=1X64=1X3=1X47=1D8=1X39=1X9=1X36=1X3=1X25=1X70=605I3X3=1X24=1X11=1X16=2X233=1X109=1I166=15D139=1X68=1I32=1X140=1I179=1X10=1X100=1X68=1X62=1X94=1X89=1X71=1X39=1D25=1X132=1X111=1X186=1X74=1X6=1X20=1X144=1X58=16D47=1X37=1X299=1X5=1X3=1D423=1X77=1D228=1X120=1X176=1X1=33I25=1I994=1X87=1X438=1X68=1X36=1X22=1X11=1X350=1X13=37D7=1X49=1X1=24I252=1X63=7D37=4I1=1X20=16I1X17=1X19=1I102=1X23=1X10=1X1=1X6=17D37=1X67=2I13=1X10=1D17=1X37=1X36=6I23=1I52=1X11=2X2=40D11=1X20=1X5=1X17=1X3=66D1X1=1X2=1X9=1X14=3X36=1I5=1I19=1X32=1X4=1X70=1X14=1X101=1X109=1D69=1X7=1X33=1X104=1X135=1X134=1X130=1X28=2X51=1X110=1X8=1X175=1X62=1X109=1X24=1X178=1X486=1X123=1X2=1X4=1X1=1X344=1X105=1X22=1X4=1X124=1X17=1X4=1X2=1X20=1X75=6D5=1X21=1X5=1D4=12I98=1D8=28D136=1X124=1X64=1D330=1D1X223=1X148=1X241=1X33=1X87=1X8=1X411=1X174=1X89=1X41=32D2X4=3I3=1X1=1X127=1X11=1X8=1X17=2I36=1X5=1X33=1X4=1X7=1X9=2D33=1X47=2X1326=36D2115=1X456=1X279=1D16=1X53=1X34=1X47=1X38=1X28=1X1=2X5=1X2=1X116=1X8=1X119=1X30=1X105=1X6=1I21=1X19=1X8=1X29=1X74=1X4=1X6=1X39=1X43=1D7=1X13=1X36=1X40=1X7=1I537=1X275=";

    #[test]
    fn test_cigar_reader_iter() {
        use std::io::prelude::*;

        // let cigar = b"150=10I50=5X5=12D50=";
        // let cigar = super::super::tests::TEST_CIGAR;
        // let cigar = b"578=1X922=1X1135=1X334=1X194=1X653=1X90=1X32=1X715=1X41=1X29=1X92=1X368=1X";
        // let cigar = b"1M";
        // let cigar = CG1;
        let cigar = CG2;

        let cg_str = std::str::from_utf8(cigar).unwrap();
        let test_ops = crate::cigar::Cigar::parse_str(cg_str);

        let reader = std::io::Cursor::new(&cigar);
        let mut iter = CigarReaderIter::new(reader, cigar.len());

        let mut ops_str = String::new();
        let mut count = 0;

        while let Ok(Some((op, len))) = iter.next_op() {
            ops_str.push_str(&format!("{len}{}", char::from(op)));
            count += 1;
        }

        println!("counted {count} ops");
        println!("expected {} ops", test_ops.0.len());

        assert_eq!(ops_str.as_bytes(), cigar);
    }

    #[test]
    fn test_cigar_reader_target_range_iter() {
        // let cigar_str = CG1;
        let cigar_str = CG2;

        println!("cigar string len: {}", cigar_str.len());
        let reader = std::io::Cursor::new(&cigar_str);
        let mut iter = CigarReaderIter::new(reader, cigar_str.len());

        // let pos_index = CigarPositionIndex::index_cigar(4096, iter);
        let pos_index =
            CigarPositionIndex::index_cigar(512, iter.map(|item| (item.op, item.op_count)));

        println!("{pos_index:?}");

        dbg!();

        let cigar_range = 0..cigar_str.len() as u64;

        let byte_index = PafByteIndex {
            record_offsets: vec![0],
            record_inner_offsets: vec![PafRecordIndex {
                cigar_range: Some(cigar_range),
                optional_fields: BTreeMap::default(),
            }],
        };

        let paf = IndexedPaf {
            data: PafData::Memory(cigar_str.to_vec().into()),
            byte_index,
            bgzi: None,
        };

        dbg!();
        let pos_index = [(0usize, pos_index)].into_iter().collect::<HashMap<_, _>>();

        dbg!();
        let mut cg_iter = paf
            .cigar_reader_iter_indexed(&pos_index, 0, 10..100)
            .unwrap();

        let mut ops_str = String::new();
        let mut count = 0;

        let mut tgt = 0;
        let mut qry = 0;

        println!("-------");

        while let Ok(Some((op, len))) = cg_iter.next_op() {
            let text = format!("{len}{}", char::from(op));
            println!("{text}");
            tgt += op.target_delta(len) as u64;
            qry += op.query_delta(len) as u64;
            ops_str.push_str(&text);
            count += 1;
        }

        println!("{ops_str}");
        println!("iterated {count} ops\ttarget delta {tgt}, query delta {qry}");
    }
}
