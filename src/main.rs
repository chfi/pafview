use clap::Parser;
use grid::AlignmentGrid;

use pafview::PafViewerApp;

use pafview::config;

use pafview::grid;

use pafview::annotations::AnnotationStore;

pub fn main() -> anyhow::Result<()> {
    // #[cfg(feature = "tracy")]
    // {
    //     eprintln!("setting up tracy layer");
    //     use tracing_subscriber::layer::SubscriberExt;

    //     bevy::utils::tracing::subscriber::set_global_default(
    //         tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    //     )
    //     .expect("error during tracy layer setup");
    // }

    let args = pafview::cli::Cli::parse();

    // Load PAF and optional FASTA
    let (alignments, sequences) = pafview::paf::load_input_files_mmap(&args)?;

    let alignment_grid = AlignmentGrid::from_alignments(&alignments, sequences.names().clone());

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
