use bevy::prelude::*;

use bevy_egui::EguiContexts;

use crate::gui::AppWindowStates;

pub(super) struct MenubarPlugin;

impl Plugin for MenubarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RegionsOfInterest>()
            .add_systems(Startup, setup)
            .add_systems(Update, (menubar_system, settings_window));
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Default, Resource)]
struct WindowStates {
    window_states: AppWindowStates,
}

fn setup(mut commands: Commands, viewer: Res<super::PafViewer>) {
    commands.insert_resource(WindowStates {
        window_states: AppWindowStates::new(&viewer.app.annotations),
    });
}

fn menubar_system(
    mut contexts: EguiContexts,
    viewer: Res<super::PafViewer>,
    mut window_states: ResMut<WindowStates>,
) {
    let ctx = contexts.ctx_mut();

    let _menubar_rect =
        crate::gui::MenuBar::show(ctx, &viewer.app, &mut window_states.window_states);
}

fn settings_window(
    mut contexts: EguiContexts,
    mut viewer: ResMut<super::PafViewer>,
    mut window_states: ResMut<WindowStates>,
) {
    let ctx = contexts.ctx_mut();
    crate::gui::config::application_settings_window(
        ctx,
        &mut window_states.window_states.config_open,
        &mut viewer.app.app_config,
    );
}

#[allow(dead_code)]
#[derive(Default, Resource)]
struct RegionsOfInterest {
    gui: crate::gui::regions::RegionsOfInterestGui,
}

/*
fn regions_of_interest_system(
    mut contexts: EguiContexts,
    viewer: Res<super::PafViewer>,
    mut window_states: ResMut<WindowStates>,
    mut roi_gui: ResMut<RegionsOfInterest>,
) {
    let ctx = contexts.ctx_mut();

    let roi_gui = &mut roi_gui.gui;

    // TODO: the RegionsOfInterestGui show_window needs an event loop; i need to handle AppEvent

    // roi_gui.show_window(
    //     ctx,
    //     &viewer.app,
}
*/
