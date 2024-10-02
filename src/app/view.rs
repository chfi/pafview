use std::collections::VecDeque;

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};
use leafwing_input_manager::action_state::ActionState;

use crate::sequences::SeqId;

use super::{
    selection::{Selection, SelectionActionTrait, SelectionComplete},
    AlignmentCamera,
};

/*

one plugin for the input-agnostic view update logic

another for view-related inputs...

*/

/*

The alignment viewport is defined using the grid of the sequence pairs,
and allows for the main camera to be updated in terms of world/base-level units.



*/
pub(super) struct AlignmentViewPlugin;

impl Plugin for AlignmentViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewHistoryCursor>()
            .add_event::<ViewEvent>()
            .add_systems(Startup, setup)
            .add_systems(
                PreUpdate,
                (
                    update_viewport_for_window_resize,
                    click_drag_pan_viewport,
                    input_update_viewport,
                    update_camera_from_viewport,
                )
                    .chain(),
            )
            .add_systems(Update, update_cursor_world)
            .add_systems(
                Update,
                (
                    super::selection::selection_action_input_system::<RectangleZoomSelection>,
                    rectangle_select_zoom_apply,
                )
                    .chain(),
            )
            .add_systems(Update, (handle_view_events, view_history_input));
    }
}

#[derive(Default, Resource)]
pub struct CursorAlignmentPosition {
    pub world_pos: Option<bevy::math::DVec2>,
    pub screen_pos: Option<bevy::math::Vec2>,

    pub target_pos: Option<(SeqId, u64)>,
    pub query_pos: Option<(SeqId, u64)>,
}

pub fn update_cursor_world(
    mut cursor_world: ResMut<CursorAlignmentPosition>,
    grid: Res<crate::AlignmentGrid>,
    view: Res<AlignmentViewport>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    let res = &window.resolution;
    let dims = [res.width(), res.height()];

    let mut new_al_cursor = CursorAlignmentPosition::default();

    if let Some(cursor_pos) = window.cursor_position() {
        let world_pos = {
            let p: [f32; 2] = cursor_pos.into();
            let wp: [f64; 2] = view.view.map_screen_to_world(dims, p).into();
            bevy::math::DVec2::from(wp)
        };

        let screen_pos = Vec2::new(
            cursor_pos.x - res.width() * 0.5,
            res.height() - cursor_pos.y - res.height() * 0.5,
        );
        new_al_cursor.screen_pos = Some(screen_pos);
        new_al_cursor.world_pos = Some(world_pos);

        new_al_cursor.target_pos = grid.x_axis.global_to_axis_exact(world_pos.x.round() as u64);
        new_al_cursor.query_pos = grid.y_axis.global_to_axis_exact(world_pos.y.round() as u64);
    }

    *cursor_world = new_al_cursor;
}

#[derive(Resource, Component)]
pub struct AlignmentViewport {
    pub view: crate::view::View,
    pub initial_view: crate::view::View,
}

impl AlignmentViewport {
    pub fn initial_view(&self) -> &crate::view::View {
        &self.initial_view
    }
}

fn setup(mut commands: Commands, grid: Res<crate::AlignmentGrid>) {
    let initial_view = crate::view::View {
        x_min: 0.0,
        x_max: grid.x_axis.total_len as f64,
        y_min: 0.0,
        y_max: grid.y_axis.total_len as f64,
    };

    let viewport = AlignmentViewport {
        view: initial_view,
        initial_view,
    };

    commands.insert_resource(viewport);
    commands.init_resource::<CursorAlignmentPosition>();
}

fn update_viewport_for_window_resize(
    mut alignment_view: ResMut<AlignmentViewport>,
    mut resize_reader: EventReader<bevy::window::WindowResized>,
) {
    let Some(new_res) = resize_reader.read().last() else {
        return;
    };

    let view = &mut alignment_view.view;

    let aspect_hw = new_res.height as f64 / new_res.width as f64;
    let old_aspect = view.height() / view.width();

    let center = view.center();

    if aspect_hw > old_aspect {
        // new is taller relative to old...
        let new_height = aspect_hw * view.width();

        view.y_min = center.y - new_height * 0.5;
        view.y_max = center.y + new_height * 0.5;
    } else {
        // new is wider relative to old
        let new_width = (new_res.width as f64 / new_res.height as f64) * view.height();

        view.x_min = center.x - new_width * 0.5;
        view.x_max = center.x + new_width * 0.5;
    }
}

pub(super) fn update_camera_from_viewport(
    alignment_view: Res<AlignmentViewport>,
    mut cameras: Query<(&mut Transform, &mut Projection, &Camera), With<AlignmentCamera>>,
) {
    let (mut transform, mut proj, camera) = cameras.single_mut();

    let Projection::Orthographic(proj) = proj.as_mut() else {
        return;
    };

    let view = &alignment_view.view;
    let mid = view.center();

    transform.translation.x = mid.x as f32;
    transform.translation.y = mid.y as f32;

    let scale = view.width() as f32 / camera.logical_target_size().unwrap().x;
    proj.scale = scale;
}

#[derive(Component, Default)]
pub(crate) struct RectangleZoomSelection;

impl SelectionActionTrait for RectangleZoomSelection {
    fn action() -> super::selection::SelectionAction {
        super::selection::SelectionAction::ZoomRectangle
    }
}

fn rectangle_select_zoom_apply(
    mut commands: Commands,
    app_view: Res<AlignmentViewport>,
    selections: Query<
        (Entity, &Selection),
        (With<RectangleZoomSelection>, With<SelectionComplete>),
    >,

    mut view_events: EventWriter<ViewEvent>,
) {
    for (sel_entity, selection) in selections.iter() {
        let Selection {
            start_world,
            end_world,
        } = selection;

        let min = start_world.min(*end_world);
        let max = start_world.max(*end_world);

        if max.x - min.x > 100.0 && max.y - min.y > 100.0 {
            let new_view = app_view
                .view
                .fit_ranges_in_view_f64(Some(min.x..=max.x), Some(min.y..=max.y));

            view_events.send(ViewEvent { view: new_view });
        }

        commands.entity(sel_entity).despawn();
    }
}

fn click_drag_pan_viewport(
    // mut click_origin: Local<Option<(bevy::math::DVec2, bevy::math::Vec2)>>,
    mut click_origin: Local<Option<bevy::math::Vec2>>,

    // TODO: this should be handled better; this system shouldn't depend
    // on a specific state/mode in the figure export plugin
    region_selection_mode: Res<super::figure_export::FigureRegionSelectionMode>,

    mut egui_contexts: bevy_egui::EguiContexts,
    menubar_size: Res<super::gui::MenubarSize>,

    mouse_button: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    alignment_cursor: Res<CursorAlignmentPosition>,

    windows: Query<&Window>,
    mut alignment_view: ResMut<AlignmentViewport>,
) {
    if region_selection_mode.user_is_selecting {
        *click_origin = None;
        return;
    }

    let window = windows.single();
    let win_size = window.resolution.size();
    let ptr_pos = window.cursor_position();

    let egui_using_cursor = egui_contexts.ctx_mut().wants_pointer_input()
        || ptr_pos.map(|p| p.y < menubar_size.height).unwrap_or(false);

    if egui_using_cursor {
        return;
    }

    if mouse_button.pressed(MouseButton::Left) && click_origin.is_none() {
        let origin = alignment_cursor.screen_pos;
        *click_origin = origin;
    }

    if !mouse_button.pressed(MouseButton::Left) {
        *click_origin = None;
    }

    let Some(cur_screen_pos) = alignment_cursor.screen_pos else {
        return;
    };
    if let Some(last_screen_pos) = click_origin.as_ref().copied() {
        // set the alignment view center so that the world positions
        // of the cursor at the start of the drag & the current frame
        // are the same

        let ctrl_down =
            keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);

        let shift_down =
            keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);

        let pan_factor = if ctrl_down {
            0.25
        } else if shift_down {
            5.0
        } else {
            1.0
        };

        let screen_delta = last_screen_pos - cur_screen_pos;
        let norm_delta = screen_delta / win_size;
        let view_size = alignment_view.view.size();
        let world_delta = ultraviolet::DVec2::new(
            norm_delta.x as f64 * view_size.x,
            norm_delta.y as f64 * view_size.y,
        ) * pan_factor;

        alignment_view.view.translate(world_delta.x, world_delta.y);
        *click_origin = Some(cur_screen_pos);
    }
}

fn input_update_viewport(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,

    windows: Query<&Window>,
    mut egui_contexts: bevy_egui::EguiContexts,

    mut mouse_wheel: EventReader<MouseWheel>,
    mut mouse_motion: EventReader<MouseMotion>,

    mut alignment_view: ResMut<AlignmentViewport>,
    mut view_events: EventWriter<ViewEvent>,

    // TODO: this should be handled better; this system shouldn't depend
    // on a specific state/mode in the figure export plugin
    region_selection_mode: Res<super::figure_export::FigureRegionSelectionMode>,
) {
    let egui_using_cursor = egui_contexts.ctx_mut().wants_pointer_input();

    let window = windows.single();

    if keyboard.just_pressed(KeyCode::Escape) && !region_selection_mode.user_is_selecting {
        view_events.send(ViewEvent {
            view: alignment_view.initial_view,
        });
    }

    let win_size = bevy::math::DVec2::new(
        window.resolution.width() as f64,
        window.resolution.height() as f64,
    );

    let mut scroll_delta = mouse_wheel
        .read()
        .map(|ev| {
            // TODO scale based on ev.unit
            match ev.unit {
                bevy::input::mouse::MouseScrollUnit::Line => {
                    ev.y as f64
                    //
                }
                bevy::input::mouse::MouseScrollUnit::Pixel => {
                    ev.y as f64 * 0.01f64
                    //
                }
            }
        })
        .sum::<f64>();

    let dt = time.delta_seconds();

    let mut mouse_delta = mouse_motion
        .read()
        .map(|ev| bevy::math::DVec2::new(ev.delta.x as f64, ev.delta.y as f64))
        .sum::<bevy::math::DVec2>();
    mouse_delta.y *= -1.0;

    if egui_using_cursor {
        mouse_delta = bevy::math::DVec2::ZERO;
        scroll_delta = 0.0;
    }

    // for (mut transform, mut proj) in camera_query.iter_mut() {
    // let Projection::Orthographic(proj) = proj.as_mut() else {
    //     continue;
    // };

    let view = alignment_view.view;
    let view_size = bevy::math::DVec2::new(view.size().x, view.size().y);

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };
    let cursor_norm = {
        let p = cursor_position;
        let x = p.x as f64 / window.resolution.width() as f64;
        let y = p.y as f64 / window.resolution.height() as f64;
        [x, y]
    };

    let ctrl_down =
        keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);

    let shift_down = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);

    let pan_factor = if ctrl_down {
        0.25
    } else if shift_down {
        5.0
    } else {
        1.0
    };

    let xv = view.width() * 0.05 * pan_factor;
    let yv = view.height() * 0.05 * pan_factor;

    let mut dv = bevy::math::DVec2::ZERO;

    if keyboard.pressed(KeyCode::ArrowLeft) {
        dv.x -= xv;
    }
    if keyboard.pressed(KeyCode::ArrowRight) {
        dv.x += xv;
    }

    if keyboard.pressed(KeyCode::ArrowUp) {
        dv.y += yv;
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        dv.y -= yv;
    }

    // if mouse_button.pressed(MouseButton::Left) {
    //     dv -= (mouse_delta / win_size) * view_size * pan_factor;
    // }

    if dv.length_squared() > 0.0 {
        alignment_view.view.translate(dv.x, dv.y);
    }

    if scroll_delta.abs() > 0.0 {
        let zoom_factor = scroll_delta;

        let base_zoom_speed = 0.05f64;

        let zoom_mult = if ctrl_down {
            0.1
        } else if shift_down {
            10.0
        } else {
            1.0
        };

        let delta_scale = 1.0 - zoom_factor * base_zoom_speed * zoom_mult;

        alignment_view
            .view
            .zoom_with_focus(cursor_norm, delta_scale as f64);
    }

    const KEY_ZOOM_FACTOR: f32 = 3.0;

    let mut key_zoom_delta = 0.0;
    if keyboard.pressed(KeyCode::PageUp) {
        key_zoom_delta += KEY_ZOOM_FACTOR;
    }
    if keyboard.pressed(KeyCode::PageDown) {
        key_zoom_delta -= KEY_ZOOM_FACTOR;
    }

    if key_zoom_delta.abs() > 0.0 {
        let zoom = if key_zoom_delta < 0.0 {
            1.0 + key_zoom_delta.abs() * dt
        } else {
            1.0 - key_zoom_delta.abs() * dt
        };

        alignment_view.view.zoom_with_focus([0.5, 0.5], zoom as f64);
    }
}

#[derive(Default, Resource)]
struct ViewHistoryCursor {
    past: VecDeque<crate::view::View>,
    future: VecDeque<crate::view::View>,
}

#[derive(Event)]
struct ViewEvent {
    view: crate::view::View,
}

fn handle_view_events(
    mut view_history: ResMut<ViewHistoryCursor>,
    mut app_view: ResMut<AlignmentViewport>,
    mut view_events: EventReader<ViewEvent>,
) {
    for view_ev in view_events.read() {
        view_history.future.clear();
        view_history.past.push_back(app_view.view);

        let new_view = app_view
            .view
            .fit_ranges_in_view_f64(Some(view_ev.view.x_range()), Some(view_ev.view.y_range()));
        app_view.view = new_view;
    }
}

fn view_history_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut view_history: ResMut<ViewHistoryCursor>,
    mut app_view: ResMut<AlignmentViewport>,
) {
    let ctrl = keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight);

    let back_key_input =
        ctrl && (keyboard.just_pressed(KeyCode::KeyZ) || keyboard.just_pressed(KeyCode::ArrowLeft));
    let forward_key_input = ctrl
        && (keyboard.just_pressed(KeyCode::KeyR) || keyboard.just_pressed(KeyCode::ArrowRight));

    let back_mouse_input = mouse.just_pressed(MouseButton::Back);
    let forward_mouse_input = mouse.just_pressed(MouseButton::Forward);

    if back_key_input || back_mouse_input {
        // move back in history
        if let Some(new_view) = view_history.past.pop_back() {
            view_history.future.push_front(app_view.view);
            app_view.view = new_view;
        }
    }

    if forward_key_input || forward_mouse_input {
        // move forward in history
        if let Some(new_view) = view_history.future.pop_front() {
            view_history.past.push_back(app_view.view);
            app_view.view = new_view;
        }
    }
}
