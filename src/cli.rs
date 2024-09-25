use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Default, bevy::prelude::Resource)]
#[command(version, about)]
pub struct Cli {
    /// Path to input PAF
    pub paf: PathBuf,

    /// Path to input FASTA file
    #[arg(long = "seq")]
    pub fasta: Option<PathBuf>,

    /// Path to BED annotation file
    #[arg(long)]
    pub bed: Option<PathBuf>,

    /// Path to BEDPE annotation file
    #[arg(long)]
    pub bedpe: Option<PathBuf>,

    /// Path to impg index file
    #[arg(long)]
    pub impg: Option<PathBuf>,

    /// Path to alignment color schemes file
    ///
    /// File should be a tab-delimited text file, with the first 9
    /// columns of each line corresponding (exactly) to an alignment
    /// in the PAF.
    ///
    /// The last column is a comma-separated list of values in the
    /// format `<op>:<color>:<color>`, where `<op>` is one of `{M, =,
    /// X, I, D}`, and `<color>` is a hex-formatted RGB color (e.g.
    /// `#FF1100`).
    ///
    /// The first color is the background color, the second the
    /// foreground color, of the corresponding CIGAR operation.
    #[arg(long)]
    pub color_schemes: Option<PathBuf>,

    /// Optional list of sequences to include as targets (X axis)
    #[arg(long)]
    pub target_seqs: Option<Vec<String>>,

    /// Optional list of sequences to include as queries (Y axis)
    #[arg(long)]
    pub query_seqs: Option<Vec<String>>,

    /// Start in dark mode
    #[arg(long)]
    pub dark_mode: bool,

    /// Reduce memory usage by only showing mappings at high zoom level
    #[arg(long)]
    pub low_mem: bool,
}
