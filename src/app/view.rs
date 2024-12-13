use std::collections::VecDeque;

use bevy::prelude::*;
use leafwing_input_manager::action_state::ActionState;

use super::{
    alignments::layout::{DefaultLayout, SeqPairLayout},
    input::ViewAction,
    AlignmentCamera,
};

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
                    pan_viewport_anchored,
                    (handle_input_update_viewport, handle_view_reset),
                )
                    .chain()
                    .before(enforce_alignment_viewport_limits)
                    .in_set(super::input::InputSet::HandleActions),
            )
            .add_systems(
                PreUpdate,
                (
                    update_viewport_for_window_resize,
                    enforce_alignment_viewport_limits,
                    update_camera_from_viewport,
                )
                    .chain(),
            )
            .add_systems(Update, (handle_view_events, view_history_input));

        app.add_plugins(rectangle_zoom::RectangleZoomViewPlugin);
    }
}

#[derive(Resource, Component, Debug, Clone, Copy)]
pub struct AlignmentViewport {
    pub view: crate::view::View,
}

fn setup(mut commands: Commands, grid: Res<crate::AlignmentGrid>) {
    let initial_view = crate::view::View {
        x_min: 0.0,
        x_max: grid.x_axis.total_len as f64,
        y_min: 0.0,
        y_max: grid.y_axis.total_len as f64,
    };

    let viewport = AlignmentViewport { view: initial_view };

    commands.insert_resource(viewport);
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

// TODO: this is pretty hacky but fine for now
pub(crate) fn enforce_alignment_viewport_limits(
    mut alignment_view: ResMut<AlignmentViewport>,
    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    const MAX_PIXELS_PER_BP: f64 = 64.0;

    let win_size = window.size().as_dvec2();
    let px_per_bp = win_size.x / alignment_view.view.width();

    if px_per_bp > MAX_PIXELS_PER_BP {
        let width = win_size.x / MAX_PIXELS_PER_BP;
        let height = width * (win_size.y / win_size.x);

        let center = alignment_view.view.center();

        alignment_view.view = crate::view::View {
            x_min: center.x - 0.5 * width,
            x_max: center.x + 0.5 * width,
            y_min: center.y - 0.5 * height,
            y_max: center.y + 0.5 * height,
        };
    }
}

pub(super) fn update_camera_from_viewport(
    alignment_view: Res<AlignmentViewport>,
    mut cameras: Query<(&mut Transform, &mut Projection, &Camera), With<AlignmentCamera>>,
) {
    let Ok((mut transform, mut proj, camera)) = cameras.get_single_mut() else {
        return;
    };

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

fn handle_view_reset(
    view_actions: Res<ActionState<ViewAction>>,
    mut view_events: EventWriter<ViewEvent>,

    layouts: Res<Assets<SeqPairLayout>>,
    default_layout: Res<DefaultLayout>,
) {
    if view_actions.just_pressed(&ViewAction::Reset) {
        if let Some(layout) = layouts.get(&default_layout.layout) {
            let bounds = crate::view::View {
                x_min: layout.mins.x,
                y_min: layout.mins.y,
                x_max: layout.maxs.x,
                y_max: layout.maxs.y,
            };

            view_events.send(ViewEvent { view: bounds });
        }
    }
}

fn handle_input_update_viewport(
    time: Res<Time>,

    view_actions: Res<ActionState<ViewAction>>,

    mut alignment_view: ResMut<AlignmentViewport>,
) {
    let dt = time.delta_seconds_f64();

    if let Some(pan_delta) = view_actions.dual_axis_data(&ViewAction::Pan) {
        let dv = pan_delta.pair.as_dvec2();
        let &[w, h] = alignment_view.view.size().as_array();

        alignment_view.view.translate(dv.x * w * dt, dv.y * h * dt);
    }
    let zoom_input = view_actions.value(&ViewAction::Zoom);

    let zoom_rate = 0.05;
    let zoom_delta = (1.0 - zoom_input * zoom_rate).clamp(0.1, 10.0);

    let zoom_center = view_actions
        .dual_axis_data(&ViewAction::ZoomOrigin)
        .map(|data| data.pair)
        .unwrap_or(Vec2::new(0.5, 0.5));

    if (zoom_delta - 1.0).abs() > 0.0 {
        // println!("zooming with {} around {:?}", zoom_delta.value, zoom_center);
        let center = zoom_center;
        let x0 = center.x as f64;
        let y0 = center.y as f64;
        alignment_view
            .view
            .zoom_with_focus([x0, y0], zoom_delta as f64);
    }
}

// run in `InputSet::HandleActions`
fn pan_viewport_anchored(
    cursor: Res<super::input::cursor::CursorPosition>,
    view_actions: Res<ActionState<ViewAction>>,

    mut alignment_view: ResMut<AlignmentViewport>,

    mut click_origin: Local<Option<bevy::math::Vec2>>,

    windows: Query<&Window>,
) {
    let Ok(win_size) = windows.get_single().map(|w| w.size()) else {
        return;
    };

    if view_actions.released(&ViewAction::AnchoredPan) {
        *click_origin = None;
    }

    let Some(cur_screen_pos) = cursor.screen else {
        return;
    };

    if view_actions.pressed(&ViewAction::AnchoredPan) && click_origin.is_none() {
        *click_origin = Some(cur_screen_pos);
    }

    if let Some(last_screen_pos) = click_origin.as_ref().copied() {
        // TODO: other actions (modifiers) for pan factor
        let pan_factor = 1.0;
        //
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
    view_actions: Res<ActionState<ViewAction>>,

    mut view_history: ResMut<ViewHistoryCursor>,
    mut app_view: ResMut<AlignmentViewport>,
) {
    if view_actions.just_pressed(&ViewAction::ViewHistoryBack) {
        if let Some(new_view) = view_history.past.pop_back() {
            view_history.future.push_front(app_view.view);
            app_view.view = new_view;
        }
    }
    if view_actions.just_pressed(&ViewAction::ViewHistoryForward) {
        if let Some(new_view) = view_history.future.pop_front() {
            view_history.past.push_back(app_view.view);
            app_view.view = new_view;
        }
    }
}

mod rectangle_zoom {
    use bevy::{math::DVec2, render::view::RenderLayers, sprite::Mesh2dHandle};

    use crate::{
        app::{
            input::{cursor::CursorPosition, InputSet, RectangleSelectAction, UserAction},
            render::bordered_rect::BorderedRectMaterial2d,
        },
        view::View,
    };

    use super::*;

    pub(super) struct RectangleZoomViewPlugin;

    impl Plugin for RectangleZoomViewPlugin {
        fn build(&self, app: &mut App) {
            // app.add_systems(Startup, prepare_assets);
            app.add_systems(Startup, spawn_zoom_entity)
                .add_systems(
                    PreUpdate,
                    forward_selection_cancel_action.in_set(InputSet::BuildUserActions),
                )
                .add_systems(
                    PreUpdate,
                    (
                        handle_actions.in_set(InputSet::HandleActions),
                        update_rectangle,
                    )
                        .chain(),
                );
        }
    }
    #[derive(Component)]
    struct RectangleZoomEntity;

    #[derive(Component)]
    struct RectangleZoomEndpoints {
        origin: DVec2,
        end: DVec2,
    }

    fn spawn_zoom_entity(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<BorderedRectMaterial2d>>,
    ) {
        let material = materials.add(BorderedRectMaterial2d {
            border_width_px: 1.0,
            ..default()
        });
        let mesh = meshes.add(Rectangle::from_length(1.0));

        commands.spawn((
            RectangleZoomEntity,
            RenderLayers::layer(1),
            Mesh2dHandle(mesh.clone()),
            material.clone(),
            SpatialBundle {
                visibility: Visibility::Hidden,
                ..default()
            },
        ));
    }

    fn forward_selection_cancel_action(
        mut user_actions: ResMut<ActionState<UserAction>>,
        mut view_actions: ResMut<ActionState<ViewAction>>,

        endpoints: Query<&RectangleZoomEndpoints>,

        mut debounce: Local<bool>,
    ) {
        let Ok(_endpoint) = endpoints.get_single() else {
            if *debounce {
                let cancel_data = user_actions.button_data_mut_or_default(&UserAction::Cancel);
                if cancel_data.released() {
                    *debounce = false;
                } else {
                    *cancel_data = default();
                }
            }
            return;
        };
        *debounce = false;

        let cancel_data = user_actions.button_data_mut_or_default(&UserAction::Cancel);
        let select_cancel_data = view_actions.button_data_mut_or_default(
            &ViewAction::RectangleZoom(RectangleSelectAction::CancelSelect),
        );
        *select_cancel_data = cancel_data.clone();

        if cancel_data.pressed() {
            *debounce = true;
        }
        *cancel_data = leafwing_input_manager::action_state::ButtonData::default();
    }

    fn update_rectangle(
        view: Res<AlignmentViewport>,
        cursor: Res<CursorPosition>,

        mut zoom_rect: Query<
            (Entity, &mut Transform, &mut RectangleZoomEndpoints),
            With<RectangleZoomEntity>,
        >,
        windows: Query<&Window>,
    ) {
        let Some(world_cursor) = cursor.world else {
            return;
        };
        let Ok(screen_dims) = windows.get_single().map(|w| w.size()) else {
            return;
        };

        for (_ent, mut transform, mut endpoints) in zoom_rect.iter_mut() {
            endpoints.end = world_cursor;

            let s0 = view.view.map_world_to_screen(screen_dims, endpoints.origin);
            let s1 = view.view.map_world_to_screen(screen_dims, endpoints.end);
            let mid = 0.5 * (s0 + s1);
            let dims = (s1 - s0).abs();

            let pt = Vec3::new(mid.x, screen_dims.y - mid.y, 1.0);

            transform.translation.x = pt.x;
            transform.translation.y = pt.y;
            transform.scale = Vec3::new(dims.x, dims.y, 1.0);
        }
    }

    fn handle_actions(
        mut commands: Commands,
        cursor: Res<CursorPosition>,
        actions: Res<ActionState<ViewAction>>,
        mut view_events: EventWriter<ViewEvent>,

        mut zoom_rect: Query<(Entity, &mut Visibility), With<RectangleZoomEntity>>,

        endpoints: Query<&RectangleZoomEndpoints>,
    ) {
        let Ok((zoom_ent, mut visibility)) = zoom_rect.get_single_mut() else {
            return;
        };

        let Some(world_cursor) = cursor.world else {
            return;
        };

        if let Ok(endpoints) = endpoints.get(zoom_ent) {
            // the user is currently moving the `endpoints.end` point with their cursor
            //
            //
            // make sure to hide the rectangle when there are no endpoints

            if actions.just_pressed(&ViewAction::RectangleZoom(
                RectangleSelectAction::CancelSelect,
            )) {
                println!("canceling");
                // if the selection is *canceled*, just remove the endpoints
                commands.entity(zoom_ent).remove::<RectangleZoomEndpoints>();
                *visibility = Visibility::Hidden;
            }

            if actions.just_released(&ViewAction::RectangleZoom(
                RectangleSelectAction::StartOrEndSelect,
            )) {
                println!("finishing rectangle zoom");
                // if the selection action is *released*, send a view event and remove the endpoints component
                let mins = endpoints.origin.min(endpoints.end);
                let maxs = endpoints.origin.max(endpoints.end);

                let new_view = View {
                    x_min: mins.x,
                    y_min: mins.y,
                    x_max: maxs.x,
                    y_max: maxs.y,
                };
                view_events.send(ViewEvent { view: new_view });

                commands.entity(zoom_ent).remove::<RectangleZoomEndpoints>();
                *visibility = Visibility::Hidden;
            }
        } else {
            if actions.just_pressed(&ViewAction::RectangleZoom(
                RectangleSelectAction::StartOrEndSelect,
            )) {
                println!("creating endpoints");
                // add the endpoints component
                commands.entity(zoom_ent).insert(RectangleZoomEndpoints {
                    origin: world_cursor,
                    end: world_cursor,
                });

                *visibility = Visibility::Visible;
            }
        }

        //
    }
}
