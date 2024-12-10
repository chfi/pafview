use egui::{Color32, FontId};

use crate::annotations::AnnotationStore;

pub mod config;
// pub mod debug;
pub mod goto;

#[derive(Debug, Clone, Copy)]
pub struct AppWindowStates {
    pub annotation_list_open: Option<bool>,
    pub goto_region_open: bool,

    pub regions_of_interest_open: bool,

    pub config_open: bool,

    #[cfg(debug_assertions)]
    pub label_physics_debug_open: bool,
}

impl std::default::Default for AppWindowStates {
    fn default() -> Self {
        Self {
            annotation_list_open: None,
            goto_region_open: false,
            regions_of_interest_open: false,
            config_open: false,

            #[cfg(debug_assertions)]
            label_physics_debug_open: false,
        }
    }
}

impl AppWindowStates {
    pub fn new(annotations: &AnnotationStore) -> Self {
        let annotation_list_open = (!annotations.is_empty()).then_some(false);

        AppWindowStates {
            annotation_list_open,
            goto_region_open: false,
            regions_of_interest_open: false,

            config_open: false,

            #[cfg(debug_assertions)]
            label_physics_debug_open: false,
            // label_physics_debug_open: true,
        }
    }
}
