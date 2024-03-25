use bimap::BiMap;
use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{annotations::AnnotationStore, view::View, AlignedSeq, PafInput, PafViewerApp};

use super::AppWindowStates;

#[derive(Default)]
pub struct RegionsOfInterestGui {
    selected_region_set: Option<SelectedRegionSet>,
}

enum SelectedRegionSet {
    Bookmarks,
    Annotations { list_id: usize },
}

impl RegionsOfInterestGui {
    pub fn show_window(
        &mut self,
        ctx: &egui::Context,
        annotations: &mut crate::annotations::AnnotationStore,
        alignment_grid: &crate::grid::AlignmentGrid,
        seq_names: &BiMap<String, usize>,
        input: &PafInput,
        view: &mut View,
        window_states: &mut AppWindowStates,
    ) {
        egui::Window::new("View")
            .open(&mut window_states.regions_of_interest_open)
            // .resizable(true)
            // .vscroll(true)
            // .default_open(false)
            .show(&ctx, |ui| {
                self.ui(annotations, alignment_grid, seq_names, input, view, ui)
            });
    }

    fn ui(
        &mut self,
        annotations: &mut crate::annotations::AnnotationStore,
        alignment_grid: &crate::grid::AlignmentGrid,
        seq_names: &BiMap<String, usize>,
        input: &PafInput,
        view: &mut View,
        ui: &mut Ui,
    ) {
        // 2 panes, left/right

        // left is list of annotation sources, including the "built-in" "bookmarks" set
        egui::SidePanel::left("sources_panel")
            .resizable(true)
            .show_inside(ui, |ui| {
                // Bookmarks + annotations
                // -- should also allow loading annotations
            });

        // right is list of annotations in selected source;
        // for the bookmarks set, new annotations can be added

        // for both bookmarks and BED annotations, controls for moving
        // the view and toggling highlights & labels are available
        egui::SidePanel::right("regions_panel")
            .resizable(true)
            .show_inside(ui, |ui| {
                //
            });
    }
}
