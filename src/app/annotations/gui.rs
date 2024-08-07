use crate::{annotations::RecordListId, gui::AppWindowStates};

use bevy::prelude::*;

pub struct AnnotationsWindow {
    selected_annotation_list: Option<RecordListId>,
}

impl AnnotationsWindow {
    pub fn show_window(
        &mut self,

        annotations: &crate::annotations::AnnotationStore,
        window_states: &mut AppWindowStates,

        annotation_query: Query<(Entity, &super::Annotation, &super::DisplayEntities)>,
        display_query: Query<&mut Visibility>,

        ctx: &egui::Context,
    ) {
        egui::Window::new("Annotations")
            .open(&mut window_states.regions_of_interest_open)
            .default_width(600.0)
            .resizable(true)
            // .auto_sized()
            // .vscroll(true)
            // .default_open(false)
            .show(&ctx, |ui| {
                self.ui(annotations, annotation_query, display_query, ui);
            });
    }

    fn ui(
        &mut self,
        annotations: &crate::annotations::AnnotationStore,

        annotation_query: Query<(Entity, &super::Annotation, &super::DisplayEntities)>,
        mut display_query: Query<&mut Visibility>,

        ui: &mut egui::Ui,
    ) {
        egui::SidePanel::left("sources_panel")
            .resizable(true)
            .show_inside(ui, |ui| {
                egui::Grid::new("annotation_lists_grid")
                    .num_columns(1)
                    .striped(true)
                    .show(ui, |ui| {
                        // loaded annotation lists
                        for (list_id, source_name) in annotations.source_names_iter() {
                            let list_open = Some(list_id) == self.selected_annotation_list;
                            let sel_list = ui.selectable_label(list_open, source_name);

                            if sel_list.clicked() {
                                self.selected_annotation_list = Some(list_id);
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

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(list_id) = self.selected_annotation_list else {
                return;
            };

            self.annotation_panel_ui(annotations, annotation_query, display_query, ui, list_id);

            // match set {
            //     SelectedRegionSet::Bookmarks => {
            //         self.bookmark_panel_ui(app, annotation_painter, view, ui);
            //         // self.bookmark_panel_ui(annotations, alignment_grid, seq_names, input, view, ui);
            //     }
            //     SelectedRegionSet::Annotations { list_id } => {
            //     }
            // }
        });
    }

    fn annotation_panel_ui(
        &mut self,
        annotations: &crate::annotations::AnnotationStore,

        annotation_query: Query<(Entity, &super::Annotation, &super::DisplayEntities)>,
        mut display_query: Query<&mut Visibility>,

        ui: &mut egui::Ui,

        list_id: RecordListId,
    ) {
        let Some(list) = annotations.list_by_id(list_id) else {
            return;
        };

        ui.vertical(|ui| {
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

                        /*
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
                        */
                    });
                //
            });

            ui.data_mut(|data| data.insert_temp(ui.id().with("filter_text"), filter_text));
        });
    }
}
