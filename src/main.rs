use clap::Parser;
use grid::AlignmentGrid;

use pafview::PafViewerApp;

use pafview::config;

use pafview::grid;

use pafview::annotations::AnnotationStore;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AlignedSeq {
    // name of the given sequence
    name: String,
    // its length
    len: u64,
    // its start offset in the global all-to-all alignment matrix
    #[deprecated]
    offset: u64,
}

pub fn main() -> anyhow::Result<()> {
    let args = pafview::cli::Cli::parse();

    // Load PAF and optional FASTA
    let (alignments, sequences) = pafview::paf::load_input_files(&args)?;

    let align_count = alignments.pairs.values().map(|al| al.len()).sum::<usize>();
    println!(
        "drawing {align_count} alignments across {} sequence pairs",
        alignments.pairs.len()
    );

    let alignment_grid = AlignmentGrid::from_alignments(&alignments, sequences.names().clone());
    // let alignment_grid = AlignmentGrid::from_axes(&alignments, sequences.names().clone(), x_axis, y_axis);
    // let alignment_grid = AlignmentGrid {
    //     x_axis,
    //     y_axis,
    //     sequence_names: sequences.names().clone(),
    // };

    let app_config = config::load_app_config().unwrap_or_default();

    let app = PafViewerApp {
        app_config,
        alignments: alignments,
        alignment_grid: alignment_grid,
        sequences,
        // paf_input: todo!(),
        annotations: AnnotationStore::default(),
    };

    pafview::app::run(app)
}
