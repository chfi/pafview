use std::sync::Arc;
use std::{collections::BTreeMap, ops::Deref};

use std::io::{prelude::*, BufReader};

use bevy::utils::HashMap;
use bgzip::index::BGZFIndex;

pub struct IndexedPaf {
    data: PafSource,
    byte_index: PafByteIndex,
}

#[derive(Clone)]
enum PafSource {
    File(std::path::PathBuf),
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

impl PafSource {
    fn reader(&self) -> std::io::Result<Box<dyn BufRead>> {
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

// impl IndexedPaf {
//     pub fn iter_lines(&self) -> IndexedPafLineIter
// }

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
