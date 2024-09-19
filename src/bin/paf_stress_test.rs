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

    #[arg(long)]
    impg: Option<PathBuf>,
}

use pafview::cigar::memmap::IndexedPaf;

fn timed<T>(text: &str, f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let val = f();
    println!("{text} in {}ms", t0.elapsed().as_millis());
    val
}

fn test_impg(impg: &PathBuf, paf: &PathBuf) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let (alignments, _sequences) = pafview::Alignments::from_impg(impg, paf)
        .map_err(|e| anyhow::anyhow!("Error loading PAF and impg index: {e:?}"))?;

    let als = &alignments.alignments;
    println!(
        "loaded {} alignments in {}ns via impg",
        als.len(),
        t0.elapsed().as_nanos()
    );

    let mut als = als.iter().collect::<Vec<_>>();
    let mut rng = rand::thread_rng();
    // als.shuffle(&mut rng);

    let mut op_count = 0;
    let mut total_tgt = 0u64;
    let mut total_qry = 0u64;
    let t1 = std::time::Instant::now();

    let mut op_counts_log10 = [0usize; 10];
    let mut ts_op_counts_lg10 = [0u128; 10];

    for (i, al) in als.iter().enumerate() {
        let t0 = std::time::Instant::now();
        // for (i, cg) in cigars.iter().enumerate() {
        // if i % 500 == 0 {
        //     println!("{i} / {}", als.len());
        // }

        let mut tgt = 0;
        let mut qry = 0;

        let tgt_len = al.location.aligned_target_len();

        // let start = 0;
        let start = tgt_len / 4;
        let end = (3 * start).min(tgt_len);

        let start = al.location.target_range.start + start;
        let end = al.location.target_range.start + end;

        let al_iter = al.iter_target_range(start..end);
        // let cg_iter = al.cigar.iter_target_range(start..end);
        // let cg_iter = al.cigar.whole_cigar();
        // println!("built alignment iter in {}us", t0.elapsed().as_micros());

        let mut inner_ops = 0;

        for item in al_iter {
            let op = item.op;
            op_count += 1;
            inner_ops += 1;
            let len = item.op_count as u64;
            tgt += op.consumes_target().then_some(len).unwrap_or(0);
            qry += op.consumes_query().then_some(len).unwrap_or(0);
        }

        let i = (inner_ops as f64).log10().clamp(0.0, 10.0) as usize;

        total_tgt += tgt;
        total_qry += qry;

        op_counts_log10[i] += 1;
        ts_op_counts_lg10[i] += t0.elapsed().as_nanos();
    }

    // for (i, &line_ix) in order.iter().enumerate() {
    //     // println!(" -- TGT {tgt}\t QRY {qry}");
    // }
    println!(
        "iterated {op_count} cigar ops across {} alignments in {}ms",
        als.len(),
        t1.elapsed().as_millis()
    );

    println!(
        "{:10} - {:6} - {:10} - {}",
        "mag of op count", "#", "time (us)", "avg time"
    );
    for (ix, (&count, &time)) in std::iter::zip(&op_counts_log10, &ts_op_counts_lg10).enumerate() {
        let tsec = time as f64 / 1_000.0;
        let tavg = (time as f64 / count as f64) / 1_000.0;
        let num = 10usize.pow(ix as u32);
        let text = format!("{num:10} - {count} - {tsec}\t{tavg}");
        println!("{text}");
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    use bstr::ByteSlice;
    let cli = Cli::parse();
    // let impg = if let Some(impg) = cli.impg.as_ref() {

    if let Some(impg) = cli.impg.as_ref() {
        test_impg(impg, &cli.paf)?;
        // return Ok(());
    }
    /*
        Some(alignments)


    } else {
        None
    };
    */

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

    let mut op_counts_log10 = [0usize; 10];
    let mut ts_op_counts_lg10 = [0u128; 10];

    let mut op_count = 0;
    let mut al_count = 0;
    let mut total_tgt = 0u64;
    let mut total_qry = 0u64;

    let t0 = std::time::Instant::now();
    for (i, &line_ix) in order.iter().enumerate() {
        let t0 = std::time::Instant::now();
        if i % 500 == 0 {
            println!("{i} / {}", order.len());
        }
        // for line_ix in [0] {
        // let (tgt, qry) = timed("iterated cigar", || {
        let mut inner_ops = 0;

        let (tgt, qry) = {
            let mut cg_iter = indexed_paf.cigar_reader_iter(line_ix).unwrap();
            let mut tgt = 0;
            let mut qry = 0;
            // let op = cg_iter.read_op();
            // println!("{op:?}");
            while let Some(Ok((op, count))) = cg_iter.read_op() {
                inner_ops += 1;
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

        let i = (inner_ops as f64).log10().clamp(0.0, 10.0) as usize;
        op_counts_log10[i] += 1;
        ts_op_counts_lg10[i] += t0.elapsed().as_nanos();
    }
    println!(
        "iterated {op_count} cigar ops across {} alignments in {}ms",
        al_count,
        t0.elapsed().as_millis()
    );

    println!(
        "{:10} - {:6} - {:10} - {}",
        "mag of op count", "#", "time (us)", "avg time"
    );
    for (ix, (&count, &time)) in std::iter::zip(&op_counts_log10, &ts_op_counts_lg10).enumerate() {
        let tsec = time as f64 / 1_000.0;
        let tavg = (time as f64 / count as f64) / 1_000.0;
        let num = 10usize.pow(ix as u32);
        let text = format!("{num:10} - {count} - {tsec}\t{tavg}");
        println!("{text}");
    }

    Ok(())
}
