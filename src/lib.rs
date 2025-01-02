pub mod app;

pub mod cigar;
pub mod math_conv;
pub mod paf;
pub mod pixels;

pub mod config;

pub mod annotations;
pub mod cli;
pub mod gui;
pub mod regions;
pub mod render;
pub mod sequences;
pub mod view;

pub use cigar::*;
pub use paf::{Alignment, Alignments, PafLine};
pub(crate) use pixels::*;

pub use config::AppConfig;
pub use sequences::Sequences;

pub struct PafViewerApp {
    pub alignments: paf::Alignments,
    pub sequences: sequences::Sequences,

    pub annotations: annotations::AnnotationStore,

    pub app_config: config::AppConfig,
}
