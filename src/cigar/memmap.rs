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
    pub fn cigar_reader(&self, line_index: usize) -> Option<Box<dyn BufRead>> {}
}

enum PafData {
    Memory(Arc<[u8]>),
    Mmap(SharedMmap),
}

impl PafData {
    fn paf_line_slice(&self, index: &PafByteIndex, line_index: usize) -> Option<&[u8]> {
        let line_offset = index.record_offsets.get(line_index)?;
        let next_line_offset = index.record_offsets.get(line_index + 1)?;
        let start = *line_offset as usize;
        let end = *next_line_offset as usize;

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

/*
trait SeekBufRead: Seek + BufRead {}
impl<S: Seek + BufRead> SeekBufRead for S {}

impl PafSource {
    fn reader(&self) -> std::io::Result<Box<dyn SeekBufRead>> {
        match self {
            PafSource::File(path_buf) => {
                let file = std::fs::File::open(path_buf)?;
                let reader = BufReader::new(file);
                Ok(Box::new(reader))
            }
            PafSource::Memory(vec) => {
                let cursor = std::io::Cursor::new(vec.clone());
                Ok(Box::new(cursor))
            }
            PafSource::Mmap(mmap) => {
                let mmap = mmap.clone();
                let cursor = std::io::Cursor::new(mmap);
                Ok(Box::new(cursor))
            }
        }
    }
}

struct PafData {
    source: PafSource,
    bgzi: Option<BGZFIndex>,
}
*/

// impl IndexedPaf {
//     pub fn iter_lines(&self) -> IndexedPafLineIter
// }

/*
pub struct IndexedPafCigarIter {
    reader: Box<dyn SeekBufRead>,

    buffer: Vec<u8>,
    buffer_offset: usize,

    read: usize,
    cigar_bytes_len: usize,
}

impl IndexedPafCigarIter {
    fn new(paf: &IndexedPaf, paf_line_ix: usize) -> std::io::Result<Self> {
        let index = &paf.byte_index;
        let line_offset = index.record_offsets[paf_line_ix];
        let cigar_range = &index.record_inner_offsets[paf_line_ix].cigar_range;
        let cigar_bytes_len = cigar_range.end - cigar_range.start;
        let buffer = Vec::with_capacity(4096);

        let mut reader = paf.data.reader()?;
        reader.seek(std::io::SeekFrom::Start(line_offset + cigar_range.start))?;
        Ok(Self {
            reader,
            buffer,
            buffer_offset: 0,
            read: 0,
            cigar_bytes_len: cigar_bytes_len as usize,
        })
    }
}

impl Iterator for IndexedPafCigarIter {
    type Item = super::CigarIterItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.read >= self.cigar_bytes_len {
            return None;
        }

        if self.buffer_offset == self.buffer.len() {
            self.buffer_offset = 0;
            self.buffer.clear();
            let read = self.reader.read(&mut self.buffer).unwrap();
        }

        todo!()
    }
}

impl IndexedPaf {
    pub fn iter_cigar(&self, paf_line_ix: usize) -> std::io::Result<IndexedPafCigarIter> {
        IndexedPafCigarIter::new(self, paf_line_ix)
    }
}
*/

impl IndexedPaf {
    pub fn memmap_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref();

        let reader = std::fs::File::open(path).map(BufReader::new)?;
        let byte_index = PafByteIndex::from_paf(reader)?;

        let mmap = {
            let file = std::fs::File::open(path)?;
            let opts = memmap2::MmapOptions::new();
            unsafe { opts.map(&file) }?
        };

        Ok(Self {
            data: PafSource::Mmap(SharedMmap(Arc::new(mmap))),
            byte_index,
        })
    }
}

impl IndexedPaf {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        let reader = std::fs::File::open(&path).map(BufReader::new)?;

        let byte_index = PafByteIndex::from_paf(reader)?;

        Ok(Self {
            data: PafSource::File(path),
            byte_index,
        })
    }

    pub fn from_bytes_vec(data: Vec<u8>) -> std::io::Result<Self> {
        let reader = std::io::Cursor::new(data.as_slice());
        let byte_index = PafByteIndex::from_paf(reader)?;

        Ok(Self {
            data: PafSource::Memory(data.into()),
            byte_index,
        })
    }
}

// pub struct IndexedPafSlice<S: Deref<Target = [u8]>> {
//     byte_index: PafByteIndex,
//     data: S,
// }

// impl<S: Deref<Target = [u8]>> IndexedPafSlice<S> {
// }

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

    // pub fn iter_paf_lines(&mut self) -> impl Iterator<Item = crate::PafLine<'_>> {
    //     self.data.rewind();
    // }
}

// struct IndexedPafMmap {
// }

struct PafByteIndex {
    record_offsets: Vec<u64>,
    record_inner_offsets: Vec<PafRecordIndex>,
}

// offsets are relative to the start of the record in the file
struct PafRecordIndex {
    cigar_range: std::ops::Range<u64>,
    optional_fields: BTreeMap<[u8; 2], u64>,
}

/*
impl PafPositionIndex {
    fn iter_lines<'a, R: 'a>(
        &'a self,
    ) -> impl Iterator<Item = crate::PafLine<&'d str>> + 'a {
        // for (line_offset, inner_offsets) in
        //     std::iter::zip(&self.record_offsets, &self.record_inner_offsets)
        // {
        //     //
        // }
        std::iter::zip(&self.record_offsets, &self.record_inner_offsets).map(
            |(line_offset, inner_offsets)| {
                //


                todo!()
            },
        )
    }
}
*/

impl PafByteIndex {
    fn from_paf<R: BufRead>(mut paf_reader: R) -> std::io::Result<Self> {
        use bstr::{io::BufReadExt, ByteSlice};

        let mut record_offsets = Vec::new();
        let mut record_indices = Vec::new();

        let mut buffer = Vec::new();
        let mut offset = 0u64;

        loop {
            buffer.clear();
            let bytes_read = paf_reader.read_until(b'\n', &mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            let line_offset = offset;
            offset += bytes_read as u64;

            let line = buffer[..bytes_read].trim_ascii();

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
                let field_end = field_offset + field.len() as u64;
                cigar_range = Some(field_offset..field_end);
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
    fn read_op(&mut self) -> Option<std::io::Result<(super::CigarOp, u32)>> {
        use bstr::ByteSlice;

        if self.done {
            return None;
        }

        if self.offset_in_buffer >= self.buffer.len() {
            self.buffer.resize(4096, 0);
            self.offset_in_buffer = 0;
            match self.reader.read(&mut self.buffer) {
                Ok(bytes_read) => {
                    self.bytes_read += bytes_read;
                }
                Err(err) => {
                    return Some(Err(err));
                }
            }
        }

        let buf_slice = &self.buffer[self.offset_in_buffer..];

        let Some(op_ix) = buf_slice.find_byteset(b"M=XIDN") else {
            self.done = true;
            return None;
        };

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

        Some(Ok((op, count)))
    }
}

/*
impl<S: BufRead> Iterator for CigarReaderIter<S> {
    type Item = std::io::Result<super::CigarIterItem>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes_processed >= self.cigar_bytes_len {
            return None;
        }

        let mut error = None;

        if self.offset_in_buffer >= self.buffer.len() {
            self.buffer.resize(4096, 0);
            self.offset_in_buffer = 0;
            match self.reader.read(&mut self.buffer) {
                Ok(bytes_read) => {
                    self.bytes_read += bytes_read;
                }
                Err(err) => {
                    return Some(Err(err));
                }
            }
        }

        todo!()
    }
}
*/

impl<S: BufRead> CigarReaderIter<S> {
    fn new(cigar_reader: S, cigar_bytes_len: usize) -> Self {
        Self {
            done: false,

            buffer: Vec::new(),
            offset_in_buffer: 0,
            cigar_bytes_len,

            bytes_processed: 0,
            bytes_read: 0,

            reader: cigar_reader,
        }
    }

    /*
    fn new(paf: &IndexedPaf, line_index: usize) -> Option<Self> {
        let line_offset = index.record_offsets.get(line_index)?;
        let offsets = index.record_inner_offsets.get(line_index)?;
        let cg_range = &offsets.cigar_range;
        let start = (line_offset + cg_range.start) as usize;
        let end = (line_offset + cg_range.end) as usize;
        let cg_range = start..end;

        let cigar_bytes_len = end - start;

        let reader =

        Some(Self {
            buffer: Vec::new(),
            offset_in_buffer: 0,
            cigar_bytes_len,
            bytes_processed: 0,
            bytes_read: 0,
            reader: reader
        })


        todo!();
    }
    */
}

// struct PafIndex {
//     file: memmap2::Mmap,
//     is_bgzip: bool,
// }

// struct BgzipPafIndex<B: std::ops::Deref<Target = [u8]>> {
//     contents: B,
//     index: bgzip::index::BGZFIndex,
// }

/*
impl<B: std::ops::Deref<Target = [u8]>> BgzipPafIndex<B> {

    fn open_file(path: impl AsRef<std::ops::Path>) -> std::io::Result<Self> {

    }

}
*/

/*
struct PafIndex<B: std::ops::Deref<Target = [u8]>> {
    file_contents: B,
    // alignment_index_offsets: HashMap<>
}

impl<B: Deref<Target = [u8]>> PafIndex<B> {
    fn new(source: B) -> Self {
        Self {
            file_contents: source,
        }
    }
}
*/

/*
enum PafSourceAlt {
    Reader(Box<dyn SeekBufRead>),
    Memory(Box<dyn AsRef<[u8]>>),
}

impl PafSourceAlt {
    fn from_paf_source(source: &PafSource) -> Self {
        match source {
            PafSource::File(path_buf) => {
                Self::File(path_buf.clone())
            },
            PafSource::Memory(arc) => {
                let data = Box::new(arc.clone()) as Box<dyn AsRef<[u8]>>;
                Self::Memory(data)
            }
            PafSource::Mmap(shared_mmap) => {
                let data = Box::new(shared_mmap.clone()) as Box<dyn AsRef<[u8]>>;
                Self::Memory(data)
            }
        }
    }
}
*/
