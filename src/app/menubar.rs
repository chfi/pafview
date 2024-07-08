use bevy::prelude::*;

use bevy_egui::EguiContexts;

use crate::gui::AppWindowStates;

pub(super) struct MenubarPlugin;

impl Plugin for MenubarPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, menubar_system);
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
