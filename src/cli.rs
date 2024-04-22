use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about)]
pub(crate) struct Cli {
    /// Path to input PAF
    pub paf: PathBuf,

    /// Path to input FASTA file
    #[arg(long = "seq")]
    pub fasta: Option<PathBuf>,

    /// Path to BED annotation file
    #[arg(long)]
    pub bed: Option<PathBuf>,
}
