use crate::{
    annotations::{AnnotationId, RecordListId},
    gui::AppWindowStates,
};

use bevy::prelude::*;

#[derive(Resource, Default)]
pub struct AnnotationsWindow {
    selected_annotation_list: Option<RecordListId>,
}

impl AnnotationsWindow {
    pub fn show_window(
        &mut self,

        annotations: &crate::annotations::AnnotationStore,
        window_states: &mut AppWindowStates,
        annot_entity_map: &super::AnnotationEntityMap,

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
                self.ui(
                    annotations,
                    annot_entity_map,
                    annotation_query,
                    display_query,
                    ui,
                );
            });
    }

    fn ui(
        &mut self,
        annotations: &crate::annotations::AnnotationStore,
        annot_entity_map: &super::AnnotationEntityMap,

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

            self.annotation_panel_ui(
                annotations,
                annot_entity_map,
                annotation_query,
                display_query,
                ui,
                list_id,
            );

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
        annot_entity_map: &super::AnnotationEntityMap,

        mut annotation_query: Query<(Entity, &super::Annotation, &super::DisplayEntities)>,
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

                        for (record_id, record) in list.records.iter().enumerate() {
                            if !record.label.contains(&filter_text) {
                                continue;
                            }

                            let annot_id = (list_id, record_id);

                            self.list_entry_widget(
                                annotations,
                                annot_entity_map,
                                &annotation_query,
                                &mut display_query,
                                ui,
                                list,
                                annot_id,
                            );
                        }
                    });
                //
            });

            ui.data_mut(|data| data.insert_temp(ui.id().with("filter_text"), filter_text));
        });
    }

    fn list_entry_widget(
        &mut self,
        annotations: &crate::annotations::AnnotationStore,
        annot_entity_map: &super::AnnotationEntityMap,

        annotation_query: &Query<(Entity, &super::Annotation, &super::DisplayEntities)>,
        display_query: &mut Query<&mut Visibility>,

        ui: &mut egui::Ui,
        record_list: &crate::annotations::RecordList,
        annot_id @ (list_id, record_id): AnnotationId,
        // list_id: RecordListId,
        // record_id: RecordEntryId,
    ) {
        let Some(entity) = annot_entity_map.get(&annot_id).copied() else {
            return;
        };

        let Ok((_, _, display_ents)) = annotation_query.get(entity) else {
            return;
        };

        let record = &record_list.records[record_id];

        let mut query_enabled = display_query
            .get(display_ents.query_region)
            .map(|vis| *vis != Visibility::Hidden)
            .unwrap_or(false);
        let mut target_enabled = display_query
            .get(display_ents.target_region)
            .map(|vis| *vis != Visibility::Hidden)
            .unwrap_or(false);

        let annot_label = ui.add(egui::Label::new(&record.label).sense(egui::Sense::click()));

        if ui.toggle_value(&mut target_enabled, "Target").clicked() {
            let result = if target_enabled {
                Visibility::Hidden
            } else {
                Visibility::Visible
            };

            if let Ok(mut vis) = display_query.get_mut(display_ents.target_region) {
                *vis = result;
            }
        }

        if ui.toggle_value(&mut query_enabled, "Query").clicked() {
            let result = if query_enabled {
                Visibility::Hidden
            } else {
                Visibility::Visible
            };

            if let Ok(mut vis) = display_query.get_mut(display_ents.query_region) {
                *vis = result;
            }
        }

        ui.end_row();
    }
}
