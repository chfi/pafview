use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

use super::AlignmentCamera;

pub(super) struct AlignmentViewPlugin;

impl Plugin for AlignmentViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
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
            .add_systems(Update, update_cursor_world.after(input_update_viewport));
    }
}

#[derive(Default, Resource)]
pub struct CursorAlignmentPosition {
    pub world_pos: Option<bevy::math::DVec2>,
}

pub fn update_cursor_world(
    mut cursor_world: ResMut<CursorAlignmentPosition>,
    view: Res<AlignmentViewport>,
    windows: Query<&Window>,
) {
    let window = windows.single();
    let res = &window.resolution;
    let dims = [res.width(), res.height()];

    let world_pos = window.cursor_position().map(|p| {
        let p: [f32; 2] = p.into();
        let wp: [f64; 2] = view.view.map_screen_to_world(dims, p).into();
        bevy::math::DVec2::from(wp)
    });

    cursor_world.world_pos = world_pos;
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

fn input_update_viewport(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,

    mut egui_contexts: bevy_egui::EguiContexts,

    mut mouse_wheel: EventReader<MouseWheel>,
    mut mouse_motion: EventReader<MouseMotion>,

    mut alignment_view: ResMut<AlignmentViewport>,
    windows: Query<&Window>,
) {
    let egui_using_cursor = egui_contexts.ctx_mut().wants_pointer_input();

    let window = windows.single();

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
