use bevy::prelude::*;

use bevy_egui::EguiContexts;

use crate::gui::AppWindowStates;

use super::{annotations::gui::AnnotationsWindow, view::AlignmentViewport};

pub(super) struct MenubarPlugin;

impl Plugin for MenubarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenubarSize>()
            .init_resource::<AnnotationsWindow>()
            .add_systems(Startup, setup)
            .add_systems(
                PreUpdate,
                (
                    menubar_system,
                    annotations_window,
                    // regions_of_interest_system,
                    settings_window,
                    goto_region_window,
                )
                    .after(bevy_egui::EguiSet::BeginFrame),
            );
        // .add_systems(
        //     Update,
        //     (
        //         menubar_system,
        //         regions_of_interest_system,
        //         settings_window,
        //         goto_region_window,
        //     ),
        // );
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Default, Resource)]
pub(crate) struct WindowStates {
    pub(crate) window_states: AppWindowStates,
}

#[derive(Default, Resource)]
pub struct MenubarSize {
    pub height: f32,
}

fn setup(mut commands: Commands, annotations: Res<super::annotations::Annotations>) {
    commands.insert_resource(WindowStates {
        window_states: AppWindowStates::new(&annotations.0),
    });
}

pub(crate) fn menubar_system(
    mut contexts: EguiContexts,
    mut window_states: ResMut<WindowStates>,
    mut menubar_size: ResMut<MenubarSize>,
) {
    let ctx = contexts.ctx_mut();

    let menubar_rect = crate::gui::MenuBar::show(ctx, &mut window_states.window_states);
    menubar_size.height = menubar_rect.height();
}

fn settings_window(
    mut contexts: EguiContexts,

    mut app_config: ResMut<crate::AppConfig>,

    mut window_states: ResMut<WindowStates>,
) {
    let ctx = contexts.ctx_mut();
    crate::gui::config::application_settings_window(
        ctx,
        &mut window_states.window_states.config_open,
        app_config.as_mut(),
    );
}

fn goto_region_window(
    mut contexts: EguiContexts,
    alignment_grid: Res<crate::AlignmentGrid>,

    mut window_states: ResMut<WindowStates>,

    mut viewport: ResMut<AlignmentViewport>,
) {
    let ctx = contexts.ctx_mut();

    let mut view = viewport.view;

    crate::gui::goto::goto_region_window(
        ctx,
        &mut window_states.window_states.goto_region_open,
        &alignment_grid,
        &mut view,
    );

    if viewport.view != view {
        viewport.view = view;
    }
}

fn annotations_window(
    mut contexts: EguiContexts,

    annotations: Res<super::annotations::Annotations>,
    annot_entity_map: Res<super::annotations::AnnotationEntityMap>,
    mut window_states: ResMut<WindowStates>,

    mut annots_window: ResMut<AnnotationsWindow>,

    annotation_query: Query<(
        Entity,
        &super::annotations::Annotation,
        &super::annotations::DisplayEntities,
    )>,
    display_query: Query<&mut Visibility>,
    //
) {
    let ctx = contexts.ctx_mut();
    annots_window.show_window(
        &annotations.0,
        &mut window_states.window_states,
        annot_entity_map.as_ref(),
        annotation_query,
        display_query,
        ctx,
    );

    //
}

/*
#[allow(dead_code)]
#[derive(Default, Resource)]
struct RegionsOfInterest {
    gui: crate::gui::regions::RegionsOfInterestGui,
}

fn regions_of_interest_system(
    mut contexts: EguiContexts,
    // viewer: Res<super::PafViewer>,

    mut alignment_view: ResMut<AlignmentViewport>,
    mut annotation_painter: ResMut<AnnotationPainter>,
    mut window_states: ResMut<WindowStates>,
    mut roi_gui: ResMut<RegionsOfInterest>,
) {
    let ctx = contexts.ctx_mut();

    let roi_gui = &mut roi_gui.gui;

    let mut view = alignment_view.view;

    roi_gui.show_window(
        ctx,
        &viewer.app,
        &mut annotation_painter.0,
        &mut view,
        &mut window_states.window_states,
    );

    if view != alignment_view.view {
        alignment_view.view = view;
    }

    // roi_gui.show_window(
    //     ctx,
    //     &viewer.app,
}
*/
