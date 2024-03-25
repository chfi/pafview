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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        egui::Window::new("Regions of Interest")
            .open(&mut window_states.regions_of_interest_open)
            .resizable(true)
            // .auto_sized()
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

                egui::Grid::new("region_set_grid")
                    .num_columns(1)
                    .striped(true)
                    .show(ui, |ui| {
                        // bookmarks
                        let region_set = Some(SelectedRegionSet::Bookmarks);
                        let bookmarks_open = region_set == self.selected_region_set;
                        let sel_bookmarks = ui.selectable_label(bookmarks_open, "Bookmarks");
                        ui.end_row();

                        if sel_bookmarks.clicked() {
                            self.selected_region_set = region_set;
                        }

                        // loaded annotation lists
                        for (list_id, source_name) in annotations.source_names_iter() {
                            let region_set = Some(SelectedRegionSet::Annotations { list_id });
                            let list_open = region_set == self.selected_region_set;
                            let sel_list = ui.selectable_label(list_open, source_name);

                            if sel_list.clicked() {
                                self.selected_region_set = region_set;
                            }

                            ui.end_row();
                        }

                        // TODO button to load new annotations
                    });
            });

        // needed to not have the central panel overflow the window bounds
        egui::TopBottomPanel::bottom(egui::Id::new("regions_of_interest_window_invis_panel"))
            .default_height(0.0)
            .show_separator_line(false)
            .show_inside(ui, |_ui| ());

        // right is list of annotations in selected source;
        // for the bookmarks set, new annotations can be added
        // for both bookmarks and BED annotations, controls for moving
        // the view and toggling highlights & labels are available
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(set) = self.selected_region_set else {
                return;
            };

            match set {
                SelectedRegionSet::Bookmarks => {
                    // TODO
                }
                SelectedRegionSet::Annotations { list_id } => {
                    let list = &annotations.list_by_id(list_id).unwrap();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for (record_id, record) in list.records.iter().enumerate() {
                            ui.horizontal(|ui| {
                                ui.label(format!("{}", record.label));

                                // let target_btn = ui.button("Target");
                                // let query_btn = ui.button("Query");

                                /*
                                let state = self.get_region_state(app, list_id, record_id);

                                if target_btn.clicked() {
                                    state.draw_target_region = !state.draw_target_region;
                                    log::info!(
                                        "drawing {} target region: {}\t(region {:?})",
                                        record.label,
                                        state.draw_target_region,
                                        state.seq_region
                                    );
                                }

                                if query_btn.clicked() {
                                    state.draw_query_region = !state.draw_query_region;
                                    log::info!(
                                        "drawing {} query region: {}\t(region {:?})",
                                        record.label,
                                        state.draw_query_region,
                                        state.seq_region
                                    );
                                }
                                */
                            });
                        }
                    });
                }
            }
        });
    }
}
