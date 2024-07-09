use bevy::{math::DVec2, prelude::*};

use crate::gui::AppWindowStates;

pub(super) struct RegionSelectionPlugin;

impl Plugin for RegionSelectionPlugin {
    fn build(&self, app: &mut App) {
        // app.init_resource::<RegionsOfInterest>()
        //     .add_systems(Startup, setup)
        //     .add_systems(Update, (menubar_system, settings_window));
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Component)]
pub struct SelectStart {
    pub world_pos: DVec2,
}

#[derive(Component)]
pub struct SelectEnd {
    pub world_pos: DVec2,
}
