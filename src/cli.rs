use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
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

    /// Optional list of sequences to include as targets (X axis)
    #[arg(long)]
    pub target_seqs: Option<Vec<String>>,

    /// Optional list of sequences to include as queries (Y axis)
    #[arg(long)]
    pub query_seqs: Option<Vec<String>>,
}
