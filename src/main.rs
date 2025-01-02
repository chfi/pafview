use clap::Parser;

use pafview::{annotations::AnnotationStore, config, PafViewerApp};

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

    let app_config = config::load_app_config().unwrap_or_default();

    let app = PafViewerApp {
        app_config,
        alignments,
        sequences,
        annotations: AnnotationStore::default(),
    };

    pafview::app::run(app)
}
