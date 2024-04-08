use bimap::BiMap;
use egui::{color_picker::Alpha, util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{
    annotations::{
        draw::{AnnotShapeId, AnnotationPainter, AnnotationWorldRegion},
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

    bookmark_list_selected_ix: Option<usize>,
    bookmarks: Vec<RegionOfInterest>,
}

#[derive(Default, Clone)]
struct BookmarkDetailsWidgetState {
    displayed_ix: Option<usize>,

    target_range: String,
    query_range: String,

    color: egui::Rgba,
    label: String,
}

impl BookmarkDetailsWidgetState {
    fn reset_from(
        &mut self,
        grid: &AlignmentGrid,
        bookmark: &RegionOfInterest,
        bookmark_ix: usize,
    ) {
        if let Some(range) = bookmark.x_range.as_ref() {
            self.target_range = range.to_string_with_names(&grid.sequence_names);
        }
        if let Some(range) = bookmark.y_range.as_ref() {
            self.query_range = range.to_string_with_names(&grid.sequence_names);
        }

        self.label = bookmark.label.clone();
        self.color = bookmark.color;
        self.displayed_ix = Some(bookmark_ix);
    }
}

#[derive(Clone)]
pub struct RegionOfInterest {
    x_range: Option<AxisRange>,
    y_range: Option<AxisRange>,

    label: String,
    color: egui::Rgba,

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
            .show(&ctx, |ui| self.ui(app, annotation_painter, view, ui));
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

        let label_selected = self
            .bookmark_list_selected_ix
            .is_some_and(|ix| ix == bookmark_ix);

        let label = egui::SelectableLabel::new(label_selected, &bookmark.label);

        let label = ui.add(label);

        if label.clicked() {
            self.bookmark_list_selected_ix = Some(bookmark_ix);
        }

        if label.double_clicked() {
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

    fn bookmark_create_widget(
        &mut self,
        annotation_painter: &mut AnnotationPainter,
        alignment_grid: &crate::grid::AlignmentGrid,
        view: &mut View,
        ui: &mut Ui,
    ) {
        if ui.button("Bookmark current view").clicked() {
            let x_range = AxisRange::Global(view.x_range());
            let y_range = AxisRange::Global(view.y_range());
            let label = "New bookmark".to_string();
            let color = egui::Color32::TRANSPARENT;

            let draw_shape = AnnotationWorldRegion {
                world_x_range: Some(view.x_range()),
                world_y_range: Some(view.y_range()),
                color,
            };
            let shape_id = annotation_painter.add_shape(Box::new(draw_shape));

            self.bookmarks.push(RegionOfInterest {
                x_range: Some(x_range),
                y_range: Some(y_range),
                label,
                color: egui::Rgba::TRANSPARENT,
                shape_id,
            });
        }

        if ui.button("Select region to mark").clicked() {
            // TODO... how to handle this?? need state & comms
        }

        // TODO buttons to add individual X and Y range marks
    }

    fn bookmark_details_widget(
        &mut self,
        annotation_painter: &mut AnnotationPainter,
        alignment_grid: &crate::grid::AlignmentGrid,
        view: &mut View,
        ui: &mut Ui,
    ) {
        // text boxes &c for inputting ranges to *edit* selected bookmark...
        // probably best to have an "apply" button rather than actually parse on the fly,
        // and there should be a button to reset changes before applying

        let data_id = egui::Id::new("bookmark_detail_widget");

        let mut widget_state = ui
            .data(|data| data.get_temp::<BookmarkDetailsWidgetState>(data_id))
            .unwrap_or_default();

        let same_selection = self.bookmark_list_selected_ix == widget_state.displayed_ix;

        let Some(bookmark_ix) = self.bookmark_list_selected_ix else {
            return;
        };

        if !same_selection {
            let bookmark = &self.bookmarks[bookmark_ix];
            widget_state.reset_from(&alignment_grid, bookmark, bookmark_ix);
        }

        ui.horizontal(|ui| {
            ui.label("Target");
            ui.text_edit_singleline(&mut widget_state.target_range);
        });
        ui.horizontal(|ui| {
            ui.label("Query");
            ui.text_edit_singleline(&mut widget_state.query_range);
        });

        ui.horizontal(|ui| {
            ui.label("Label");
            ui.text_edit_singleline(&mut widget_state.label);
        });
        let color_resp = egui::widgets::color_picker::color_edit_button_rgba(
            ui,
            &mut widget_state.color,
            Alpha::OnlyBlend,
        );
        // let color_changed = egui::widgets::color_picker::color_picker_color32(
        //     ui,
        //     &mut widget_state.color,
        //     Alpha::OnlyBlend,
        // );

        if ui.button("Reset").clicked() {
            let bookmark = &self.bookmarks[bookmark_ix];
            widget_state.reset_from(&alignment_grid, bookmark, bookmark_ix);
        }

        let bookmark = &mut self.bookmarks[bookmark_ix];
        if color_resp.changed() {
            bookmark.color = widget_state.color;
            annotation_painter.set_shape_color(bookmark.shape_id, bookmark.color.into());
        }

        if ui.button("Apply").clicked() {
            let x_range = crate::grid::parse_axis_range_into_global(
                &alignment_grid.sequence_names,
                &alignment_grid.x_axis,
                &widget_state.target_range,
            );
            let y_range = crate::grid::parse_axis_range_into_global(
                &alignment_grid.sequence_names,
                &alignment_grid.y_axis,
                &widget_state.query_range,
            );

            if !widget_state.label.is_empty() {
                bookmark.label = widget_state.label.clone();
            }

            bookmark.color = widget_state.color;
            annotation_painter.set_shape_color(bookmark.shape_id, bookmark.color.into());

            if let Some(range) = x_range {
                bookmark.x_range = Some(range);
            }
            if let Some(range) = y_range {
                bookmark.y_range = Some(range);
            }
        }

        ui.data_mut(|data| data.insert_temp(data_id, widget_state));
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

            self.bookmark_create_widget(annotation_painter, &app.alignment_grid, view, ui);

            ui.separator();
            self.bookmark_details_widget(annotation_painter, &app.alignment_grid, view, ui);

            //
            // ui.horizontal(|ui| {
            //     //
            // });
        });
    }
}
