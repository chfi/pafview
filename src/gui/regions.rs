use bimap::BiMap;
use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{
    annotations::{
        draw::{AnnotShapeId, AnnotationPainter},
        AnnotationStore,
    },
    grid::{AlignmentGrid, AxisRange},
    view::View,
    AlignedSeq, PafInput, PafViewerApp,
};

use super::AppWindowStates;

#[derive(Default)]
pub struct RegionsOfInterestGui {
    selected_region_set: Option<SelectedRegionSet>,

    bookmarks: Vec<RegionOfInterest>,
}

#[derive(Clone)]
pub struct RegionOfInterest {
    x_range: Option<AxisRange>,
    y_range: Option<AxisRange>,

    label: String,
    color: Option<egui::Color32>,

    shape_id: AnnotShapeId,
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
        app: &PafViewerApp,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &mut crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
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
                self.ui(app, annotation_painter, view, ui)
                // self.ui(annotations, alignment_grid, seq_names, input, view, ui)
            });
    }

    fn ui(
        &mut self,
        app: &PafViewerApp,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
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
                        for (list_id, source_name) in app.annotations.source_names_iter() {
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
                    self.bookmark_panel_ui(app, annotation_painter, view, ui);
                    // self.bookmark_panel_ui(annotations, alignment_grid, seq_names, input, view, ui);
                }
                SelectedRegionSet::Annotations { list_id } => {
                    let list = &app.annotations.list_by_id(list_id).unwrap();

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

    fn bookmark_entry_widget(
        &mut self,
        // annotations: &crate::annotations::AnnotationStore,
        annotation_painter: &mut AnnotationPainter,
        alignment_grid: &crate::grid::AlignmentGrid,
        view: &mut View,
        ui: &mut Ui,
        bookmark_ix: usize,
        // book
    ) {
        let bookmark = &self.bookmarks[bookmark_ix];
        let label = egui::Label::new(&bookmark.label).sense(egui::Sense::click());
        let label = ui.add(label);

        if label.double_clicked() {
            // TODO zoom to region

            let x_range = bookmark
                .x_range
                .as_ref()
                .and_then(|r| alignment_grid.x_axis.axis_range_into_global(r));
            let y_range = bookmark
                .y_range
                .as_ref()
                .and_then(|r| alignment_grid.y_axis.axis_range_into_global(r));

            let new_view = view.fit_ranges_in_view_f64(x_range, y_range);

            *view = new_view;
        }

        if ui.button("Display").clicked() {
            *annotation_painter.enable_shape_mut(bookmark.shape_id) ^= true;
        }
    }

    fn bookmark_panel_ui(
        &mut self,
        app: &PafViewerApp,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
        view: &mut View,
        ui: &mut Ui,
    ) {
        ui.vertical(|ui| {
            // list of existing bookmarks + simple controls
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new(ui.id().with("bookmark_list"))
                    .striped(true)
                    .show(ui, |ui| {
                        for ix in 0..self.bookmarks.len() {
                            self.bookmark_entry_widget(
                                // &app.annotations,
                                annotation_painter,
                                &app.alignment_grid,
                                view,
                                ui,
                                ix,
                            );

                            ui.end_row();
                        }
                    });
                //
            });

            // panel for creating & modifying bookmarks

            // "bookmark view"
            // "select & mark region"
            // "custom mark" w/ target & query region inputs

            //
            ui.horizontal(|ui| {
                //
            });
        });
    }
}
