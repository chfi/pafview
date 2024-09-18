use clap::Parser;
use std::path::PathBuf;

use rand::prelude::*;

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

    let indexed_paf = timed("built memmapped PAF index", || {
        IndexedPaf::memmap_paf(&cli.paf, bgzi)
    })?;

    let mut line_is = Vec::new();

    let (line_count, cigar_bytes) = timed("iterated parsed PAF lines", || {
        let mut count = 0;
        let mut total_cigar_bytes = 0;
        indexed_paf.for_each_paf_line(|line| {
            line_is.push(count);
            count += 1;
            if let Some(cg) = line.cigar {
                total_cigar_bytes += cg.len();
            }
        });

        (count, total_cigar_bytes)
    });

    println!(" -- counted {line_count} lines, {cigar_bytes} total cigar bytes");

    let mut rng = rand::thread_rng();

    // line_is.shuffle(&mut rng);

    let order = line_is;

    let mut op_count = 0;
    let mut al_count = 0;
    let mut total_tgt = 0u64;
    let mut total_qry = 0u64;
    let t0 = std::time::Instant::now();
    for (i, &line_ix) in order.iter().enumerate() {
        if i % 500 == 0 {
            println!("{i} / {}", order.len());
        }
        // for line_ix in [0] {
        // let (tgt, qry) = timed("iterated cigar", || {
        let (tgt, qry) = {
            let mut cg_iter = indexed_paf.cigar_reader_iter(line_ix).unwrap();
            let mut tgt = 0;
            let mut qry = 0;
            // let op = cg_iter.read_op();
            // println!("{op:?}");
            while let Some(Ok((op, count))) = cg_iter.read_op() {
                op_count += 1;
                let len = count as u64;
                tgt += op.consumes_target().then_some(len).unwrap_or(0);
                qry += op.consumes_query().then_some(len).unwrap_or(0);
            }
            (tgt, qry)
            // });
        };
        al_count += 1;
        // println!(" -- TGT {tgt}\t QRY {qry}");
        total_tgt += tgt;
        total_qry += qry;
    }
    println!(
        "iterated {op_count} cigar ops across {} alignments in {}ms",
        al_count,
        t0.elapsed().as_millis()
    );

    Ok(())
}
