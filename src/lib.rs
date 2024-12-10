use std::sync::Arc;

pub mod app;

pub mod cigar;
pub mod math_conv;
pub mod paf;
pub mod pixels;

pub mod config;

pub mod annotations;
pub mod cli;
pub mod grid;
pub mod gui;
pub mod regions;
pub mod render;
pub mod sequences;
pub mod view;

pub use cigar::*;
pub use paf::{Alignment, Alignments, PafLine};
pub(crate) use pixels::*;

pub use grid::AlignmentGrid;

pub use config::AppConfig;
pub use sequences::Sequences;

pub struct PafViewerApp {
    // pub alignments: Arc<paf::Alignments>,
    // pub alignment_grid: Arc<grid::AlignmentGrid>,
    pub alignments: paf::Alignments,
    pub alignment_grid: grid::AlignmentGrid,
    pub sequences: sequences::Sequences,

    pub annotations: annotations::AnnotationStore,

    pub app_config: config::AppConfig,
}
