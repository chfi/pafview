use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::camera::ScalingMode,
};

pub(super) struct AlignmentViewPlugin;

impl Plugin for AlignmentViewPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup).add_systems(
            Update,
            (input_update_viewport, update_camera_from_viewport)
                .chain()
                .before(bevy::render::camera::camera_system::<OrthographicProjection>),
        );
    }
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
}

pub(super) fn update_camera_from_viewport(
    alignment_view: Res<AlignmentViewport>,
    mut cameras: Query<(&mut Transform, &mut Projection), With<Camera>>,
) {
    let (mut transform, mut proj) = cameras.single_mut();

    let Projection::Orthographic(proj) = proj.as_mut() else {
        return;
    };

    // TODO update accordingly; need to move the input system here & work
    // on the viewport as well

    // proj.scaling_mode = ScalingMode::Fixed(())
}

fn input_update_viewport(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,

    mut egui_contexts: bevy_egui::EguiContexts,

    mut mouse_wheel: EventReader<MouseWheel>,
    mut mouse_motion: EventReader<MouseMotion>,

    mut alignment_view: ResMut<AlignmentViewport>,
    // mut camera_query: Query<(&mut Transform, &mut Projection), With<Camera>>,
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

    let xv = view.width() * 0.01;
    let yv = view.width() * 0.01;

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
    // transform.translation.x += dv.x;
    // transform.translation.y += dv.y;

    if scroll_delta.abs() > 0.0 {
        let zoom = if scroll_delta < 0.0 {
            1.0 + scroll_delta.abs() * dt
        } else {
            1.0 - scroll_delta.abs() * dt
        };

        view.zoom_with_focus(cursor_norm, zoom as f64);
    }
    // }
}
