use std::sync::Arc;

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
pub use pixels::*;

pub use config::AppConfig;
pub use grid::AlignmentGrid;

pub struct PafViewerApp {
    pub alignments: Arc<paf::Alignments>,
    pub alignment_grid: Arc<grid::AlignmentGrid>,
    pub sequences: sequences::Sequences,

    pub annotations: annotations::AnnotationStore,

    pub app_config: config::AppConfig,
}

#[derive(Clone)]
pub enum AppEvent {
    LoadAnnotationFile { path: std::path::PathBuf },
    // AnnotationShapeDisplay {
    //     shape_id: annotations::draw::AnnotShapeId,
    //     enable: Option<bool>,
    // },

    // idk if this is a good idea but worth a try
    RequestSelection { target: regions::SelectionTarget },
}
