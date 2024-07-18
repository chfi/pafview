use std::collections::VecDeque;

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

use crate::sequences::SeqId;

use super::{
    selection::{Selection, SelectionComplete},
    AlignmentCamera,
};

pub(super) struct AlignmentViewPlugin;

impl Plugin for AlignmentViewPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewHistoryCursor>()
            .add_event::<ViewEvent>()
            .add_systems(Startup, setup)
            // .add_systems(PreUpdate, update_viewport_for_window_resize)
            .add_systems(
                Update,
                (
                    update_viewport_for_window_resize,
                    input_update_viewport,
                    update_camera_from_viewport,
                )
                    .chain()
                    .before(bevy::render::camera::camera_system::<OrthographicProjection>),
            )
            .add_systems(Update, update_cursor_world.after(input_update_viewport))
            .add_systems(
                Update,
                (rectangle_select_zoom_input, rectangle_select_zoom_apply).chain(),
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
    viewer: Res<super::PafViewer>,
    view: Res<AlignmentViewport>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    let res = &window.resolution;
    let dims = [res.width(), res.height()];

    let grid = &viewer.app.alignment_grid;

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

#[derive(Resource)]
pub struct AlignmentViewport {
    pub view: crate::view::View,
    initial_view: crate::view::View,
}

impl AlignmentViewport {
    pub fn initial_view(&self) -> &crate::view::View {
        &self.initial_view
    }
}

fn setup(mut commands: Commands, viewer: Res<super::PafViewer>) {
    let grid = &viewer.app.alignment_grid;

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

#[derive(Component)]
struct RectangleZoomSelection;

fn rectangle_select_zoom_input(
    mut commands: Commands,
    alignment_cursor: Res<CursorAlignmentPosition>,
    mouse_button: Res<ButtonInput<MouseButton>>,

    selections: Query<
        (Entity, &Selection),
        (With<RectangleZoomSelection>, Without<SelectionComplete>),
    >,
) {
    if let Ok((sel_entity, _selection)) = selections.get_single() {
        if mouse_button.just_released(MouseButton::Right) {
            commands.entity(sel_entity).insert(SelectionComplete);
        }
    } else {
        let Some(cursor) = alignment_cursor.world_pos else {
            return;
        };

        if mouse_button.just_pressed(MouseButton::Right) {
            commands.spawn((
                Selection {
                    start_world: cursor,
                    end_world: cursor,
                },
                RectangleZoomSelection,
            ));
        }
    }
}

fn rectangle_select_zoom_apply(
    mut commands: Commands,
    mut app_view: ResMut<AlignmentViewport>,
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
            let x_range = min.x..=max.x;
            let y_range = min.y..=max.y;

            let new_view = app_view
                .view
                .fit_ranges_in_view_f64(Some(min.x..=max.x), Some(min.y..=max.y));

            view_events.send(ViewEvent { view: new_view });
        }

        commands.entity(sel_entity).despawn();
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
) {
    let egui_using_cursor = egui_contexts.ctx_mut().wants_pointer_input();

    let window = windows.single();

    if keyboard.just_pressed(KeyCode::Escape) {
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
            ev.y
        })
        .sum::<f32>();

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

    let view = &mut alignment_view.view;
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

    let xv = view.width() * 0.05;
    let yv = view.height() * 0.05;

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

    if mouse_button.pressed(MouseButton::Left) {
        dv -= (mouse_delta / win_size) * view_size;
    }

    view.translate(dv.x, dv.y);

    if scroll_delta.abs() > 0.0 {
        let zoom = if scroll_delta < 0.0 {
            1.0 + scroll_delta.abs() * dt
        } else {
            1.0 - scroll_delta.abs() * dt
        };

        view.zoom_with_focus(cursor_norm, zoom as f64);
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

        view.zoom_with_focus([0.5, 0.5], zoom as f64);
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
        app_view.view = view_ev.view;
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
