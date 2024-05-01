use bimap::BiMap;
use egui::{color_picker::Alpha, util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};
use winit::event_loop::EventLoopProxy;

use crate::{
    annotations::{
        draw::{AnnotShapeId, AnnotationPainter, AnnotationWorldRegion},
        label_layout::compute_layout_for_labels,
        AnnotationStore,
    },
    grid::{AlignmentGrid, AxisRange},
    regions::SelectionTarget,
    view::View,
    AlignedSeq, PafViewerApp,
};

use super::AppWindowStates;

#[derive(Default)]
pub struct RegionsOfInterestGui {
    selected_region_set: Option<SelectedRegionSet>,

    bookmark_list_selected_ix: Option<usize>,
    bookmarks: Vec<RegionOfInterest>,

    selection_request: Option<SelectionTarget>,

    label_debug: Vec<(egui::Pos2, std::sync::Arc<egui::Galley>)>,
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
        event_loop: &EventLoopProxy<crate::AppEvent>,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &mut crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
        view: &mut View,
        window_states: &mut AppWindowStates,
    ) {
        let mut active_selection = false;
        let mut selected_view = None;
        if let Some(target) = self.selection_request.as_ref() {
            let result = target.result.lock();
            if let Some(view) = result.as_ref() {
                selected_view = Some(*view);
            } else {
                active_selection = true;
            }
        }
        if let Some(selected) = selected_view {
            let x_range = AxisRange::Global(selected.x_range());
            let y_range = AxisRange::Global(selected.y_range());
            let label = "New bookmark".to_string();
            let color = egui::Color32::TRANSPARENT;

            let draw_shape = AnnotationWorldRegion {
                world_x_range: Some(selected.x_range()),
                world_y_range: Some(selected.y_range()),
                color,
            };
            let shape_id = annotation_painter.add_shape(Box::new(draw_shape));

            let bookmark_ix = self.bookmarks.len();
            self.bookmarks.push(RegionOfInterest {
                x_range: Some(x_range),
                y_range: Some(y_range),
                label,
                color: egui::Rgba::TRANSPARENT,
                shape_id,
            });
            self.bookmark_list_selected_ix = Some(bookmark_ix);
            self.selection_request = None;
        }

        if active_selection {
            return;
        }

        egui::Window::new("Regions of Interest")
            .open(&mut window_states.regions_of_interest_open)
            .default_width(600.0)
            .resizable(true)
            // .auto_sized()
            // .vscroll(true)
            // .default_open(false)
            .show(&ctx, |ui| {
                self.ui(app, event_loop, annotation_painter, view, ui)
            });
    }

    fn ui(
        &mut self,
        app: &PafViewerApp,
        event_loop: &EventLoopProxy<crate::AppEvent>,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
        view: &mut View,
        ui: &mut Ui,
    ) {
        let painter = ui.ctx().debug_painter();

        for (pos, galley) in &self.label_debug {
            painter.galley(*pos, galley.clone(), egui::Color32::BLACK);
        }

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
                    self.bookmark_panel_ui(app, event_loop, annotation_painter, view, ui);
                    // self.bookmark_panel_ui(annotations, alignment_grid, seq_names, input, view, ui);
                }
                SelectedRegionSet::Annotations { list_id } => {
                    self.annotation_panel_ui(app, annotation_painter, view, ui, list_id);
                }
            }
        });
    }

    fn annotation_list_entry_widget(
        &mut self,
        app: &PafViewerApp,
        annotation_painter: &mut AnnotationPainter,
        // alignment_grid: &crate::grid::AlignmentGrid,
        view: &mut View,
        ui: &mut Ui,
        list_id: usize,
        record_id: usize,
    ) -> (bool, bool) {
        let Some(list) = app.annotations.list_by_id(list_id) else {
            return (false, false);
        };
        let record = &list.records[record_id];

        let label = ui.add(egui::Label::new(&record.label).sense(egui::Sense::click()));

        let target_enabled = app
            .annotations
            .target_shape_for(list_id, record_id)
            .map(|shape| annotation_painter.is_shape_enabled(shape))
            .unwrap_or(false);
        let query_enabled = app
            .annotations
            .query_shape_for(list_id, record_id)
            .map(|shape| annotation_painter.is_shape_enabled(shape))
            .unwrap_or(false);

        let x_axis = &app.alignment_grid.x_axis;
        let y_axis = &app.alignment_grid.y_axis;

        if label.double_clicked() {
            // zoom to region based on which (target and/or query) are being rendered

            let axis_range = AxisRange::Seq {
                seq_id: record.seq_id,
                range: record.seq_range.clone(),
            };
            let x_range = target_enabled
                .then(|| x_axis.axis_range_into_global(&axis_range))
                .flatten();
            let y_range = query_enabled
                .then(|| y_axis.axis_range_into_global(&axis_range))
                .flatten();

            if x_range.is_some() || y_range.is_some() {
                let size = ui.ctx().screen_rect().size();
                let mut new_view = view.fit_ranges_in_view_f64(x_range, y_range);
                new_view.apply_limits([size.x as u32, size.y as u32]);
                *view = new_view;
            }
        }

        let mut toggle_target = false;
        let mut toggle_query = false;

        if ui.button("Target").clicked() {
            toggle_target = true;
        }

        if ui.button("Query").clicked() {
            toggle_query = true;
        }

        (toggle_target, toggle_query)
    }

    fn bookmark_list_entry_widget(
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
        event_loop: &EventLoopProxy<crate::AppEvent>,
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
            let target = SelectionTarget::default();
            let _ = event_loop.send_event(crate::AppEvent::RequestSelection {
                target: target.clone(),
            });
            self.selection_request = Some(target);
        }

        // let mut selected_view = None;
        // if let Some(target) = self.selection_request.as_ref() {
        //     let result = target.result.lock();
        //     if let Some(view) = result.as_ref() {
        //         selected_view = Some(*view);
        //     }
        // }
        // if let Some(selected) = selected_view {
        //     let x_range = AxisRange::Global(selected.x_range());
        //     let y_range = AxisRange::Global(selected.y_range());
        //     let label = "New bookmark".to_string();
        //     let color = egui::Color32::TRANSPARENT;

        //     let draw_shape = AnnotationWorldRegion {
        //         world_x_range: Some(selected.x_range()),
        //         world_y_range: Some(selected.y_range()),
        //         color,
        //     };
        //     let shape_id = annotation_painter.add_shape(Box::new(draw_shape));

        //     self.bookmarks.push(RegionOfInterest {
        //         x_range: Some(x_range),
        //         y_range: Some(y_range),
        //         label,
        //         color: egui::Rgba::TRANSPARENT,
        //         shape_id,
        //     });
        //     self.selection_request = None;
        // }

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
            ui.label("Label");
            ui.text_edit_singleline(&mut widget_state.label);
        });

        ui.horizontal(|ui| {
            ui.label("Target");
            ui.text_edit_singleline(&mut widget_state.target_range);
        });
        ui.horizontal(|ui| {
            ui.label("Query");
            ui.text_edit_singleline(&mut widget_state.query_range);
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

    fn annotation_panel_ui(
        &mut self,
        app: &PafViewerApp,
        annotation_painter: &mut AnnotationPainter,
        // annotations: &crate::annotations::AnnotationStore,
        // alignment_grid: &crate::grid::AlignmentGrid,
        // seq_names: &BiMap<String, usize>,
        // input: &PafInput,
        view: &mut View,
        ui: &mut Ui,
        list_id: usize,
    ) {
        let Some(list) = app.annotations.list_by_id(list_id) else {
            return;
        };

        ui.vertical(|ui| {
            if ui.button("test label layout").clicked() {
                let ctx = ui.ctx();

                let labels = list.records.iter().map(|record| {
                    let galley = annotation_painter.cache_label(ctx, &record.label);

                    let axis_range = AxisRange::Seq {
                        seq_id: record.seq_id,
                        range: record.seq_range.clone(),
                    };
                    let x_range = app
                        .alignment_grid
                        .x_axis
                        .axis_range_into_global(&axis_range)
                        .unwrap();

                    crate::annotations::label_layout::LabelDef {
                        text: &record.label,
                        galley,
                        world_x_region: x_range,
                    }
                });

                self.label_debug =
                    compute_layout_for_labels(ctx.screen_rect().size(), view, labels);
            }

            let mut filter_text = ui.data(|data| {
                data.get_temp::<String>(ui.id().with("filter_text"))
                    .unwrap_or_default()
            });
            ui.horizontal(|ui| {
                ui.label("Filter");
                ui.text_edit_singleline(&mut filter_text);
                if ui.button("Clear").clicked() {
                    filter_text.clear();
                }
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new(ui.id().with("annotation_list"))
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Toggle display");
                        let tgl_all_target = ui.button("Target").clicked();
                        let tgl_all_query = ui.button("Query").clicked();

                        ui.end_row();

                        ui.separator();
                        ui.end_row();

                        for (record_id, record) in list.records.iter().enumerate() {
                            if !record.label.contains(&filter_text) {
                                continue;
                            }

                            let (toggle_target, toggle_query) = self.annotation_list_entry_widget(
                                app,
                                annotation_painter,
                                view,
                                ui,
                                list_id,
                                record_id,
                            );

                            // if let Some(shape) = (toggle_target || tgl_all_target)
                            //     .then(|| app.annotations.target_shape_for(list_id, record_id))
                            //     .flatten()
                            // {
                            //         *annotation_painter.enable_shape_mut(shape) ^= true;
                            // }

                            if toggle_target || tgl_all_target {
                                if let Some(shape) =
                                    app.annotations.target_shape_for(list_id, record_id)
                                {
                                    *annotation_painter.enable_shape_mut(shape) ^= true;
                                }
                            }

                            if toggle_query || tgl_all_query {
                                if let Some(shape) =
                                    app.annotations.query_shape_for(list_id, record_id)
                                {
                                    *annotation_painter.enable_shape_mut(shape) ^= true;
                                }
                            }

                            ui.end_row();
                        }
                    });
                //
            });

            ui.data_mut(|data| data.insert_temp(ui.id().with("filter_text"), filter_text));
        });
    }

    fn bookmark_panel_ui(
        &mut self,
        app: &PafViewerApp,
        event_loop: &EventLoopProxy<crate::AppEvent>,
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
                            self.bookmark_list_entry_widget(
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

            self.bookmark_create_widget(
                event_loop,
                annotation_painter,
                &app.alignment_grid,
                view,
                ui,
            );

            ui.separator();
            self.bookmark_details_widget(annotation_painter, &app.alignment_grid, view, ui);
        });
    }
}
