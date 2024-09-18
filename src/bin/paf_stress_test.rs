use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long)]
    paf: PathBuf,

    /// Force use of a specific bgzip index file
    #[arg(long)]
    bgzi: Option<PathBuf>,

    #[arg(long)]
    memmap: bool,
}

use pafview::cigar::memmap::IndexedPaf;

fn timed<T>(text: &str, f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let val = f();
    println!("{text} in {}ms", t0.elapsed().as_millis());
    val
}

fn main() -> anyhow::Result<()> {
    use bstr::ByteSlice;
    let cli = Cli::parse();

    // let t0 = std::time::Instant::now();

    let derived_bgzi_path = cli
        .paf
        .extension()
        .filter(|ext| ext.eq_ignore_ascii_case("gz") || ext.eq_ignore_ascii_case("bgz"))
        .map(|ext| {
            let mut new_ext = ext.to_os_string();
            new_ext.push(".gzi");
            let mut bgzi_path = cli.paf.clone();
            bgzi_path.set_extension(new_ext);
            bgzi_path
        });

    let bgzi_path = cli.bgzi.or(derived_bgzi_path);

    let bgzi = bgzi_path.and_then(|path| {
        std::fs::File::open(&path)
            .and_then(bgzip::index::BGZFIndex::from_reader)
            .ok()
    });
    // let bgzi_path = cli
    //     .paf
    //     .extension()
    //     .filter(|ext| ext.eq_ignore_ascii_case("gz") || ext.eq_ignore_ascii_case("bgz"));

    // let bgzi = if let Some(ext) = cli.paf.extension() {
    //     if ext.eq_ignore_ascii_case("gz") || ext.eq_ignore_ascii_case("bgz") {
    //         let mut new_ext = ext.to_os_string();
    //         new_ext.push(".gzi");
    //         let mut bgzi_path = cli.paf.clone();
    //         bgzi_path.set_extension(new_ext);

    //         let bgzi =
    //             std::fs::File::open(&bgzi_path).and_then(bgzip::index::BGZFIndex::from_reader);

    //         bgzi.ok()
    //         // match bgzi {
    //         //     Ok(bgzi) => {

    //         //         indexed_paf.with_bgzi(bgzi);
    //         //     }
    //         //     Err(err) => {
    //         //         panic!("Error loading bgzip index for compressed PAF: {err:?}");
    //         //     }
    //         // }
    //     } else {
    //         None
    //     }
    // } else {
    //     None
    // };

    let mut indexed_paf = timed("built memmapped PAF index", || {
        IndexedPaf::memmap_paf(&cli.paf, bgzi)
    })?;

    let (line_count, cigar_bytes) = timed("iterated parsed PAF lines", || {
        let mut count = 0;
        let mut total_cigar_bytes = 0;
        indexed_paf.for_each_paf_line(|line| {
            count += 1;
            if let Some(cg) = line.cigar {
                total_cigar_bytes += cg.len();
            }
        });

        (count, total_cigar_bytes)
    });

    println!(" -- counted {line_count} lines, {cigar_bytes} total cigar bytes");

    Ok(())
}
