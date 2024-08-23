use bevy::{
    math::DVec2,
    prelude::*,
    render::view::RenderLayers,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};

use leafwing_input_manager::prelude::*;

use crate::gui::AppWindowStates;

use super::{
    view::{AlignmentViewport, CursorAlignmentPosition},
    ScreenspaceCamera,
};

pub(super) struct RegionSelectionPlugin;

impl Plugin for RegionSelectionPlugin {
    fn build(&self, app: &mut App) {
        app
            // .init_state::<SelectionState>()
            .init_gizmo_group::<SelectionGizmos>()
            .add_plugins(InputManagerPlugin::<SelectionAction>::default())
            .add_systems(Startup, setup_selection_input_map)
            .add_systems(Startup, setup_selection_gizmo_config)
            // .add_systems(Update, right_click_selection_test)
            .add_systems(Update, update_selection)
            // .add_systems(Update, (initialize_selection, update_selection).chain())
            .add_systems(
                Update,
                rectangle_zoom_selection_gizmos.after(update_selection),
            )
            .add_systems(PostUpdate, despawn_completed_selections);
        // app.init_resource::<RegionsOfInterest>()
        //     .add_systems(Startup, setup)
        //     .add_systems(Update, (menubar_system, settings_window));
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

// #[derive(States, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, Debug, Reflect)]
// pub enum SelectionState {
//     #[default]
//     Nothing,
//     WorldRegion,
//     Alignments,
//     SeqPairs,
//     Columns,
//     Rows,
// }

#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
pub enum SelectionAction {
    SelectionRelease,
    ZoomRectangle,
    DistanceMeasurement,
    RegionSelection,
}

fn setup_selection_input_map(mut commands: Commands) {
    let mut input_map: InputMap<SelectionAction> = InputMap::default();

    // this is pretty messed up, since the region selection is currently (8-23)
    // used only by the figure export, and the inputs filtered by `handle_selection_user_input`
    // in `figure_export.rs` unless the figure export window is waiting for
    // a region selection
    input_map.insert(SelectionAction::RegionSelection, MouseButton::Left);

    input_map.insert(SelectionAction::SelectionRelease, MouseButton::Right);
    input_map.insert(SelectionAction::ZoomRectangle, MouseButton::Right);

    let dist_chord = UserInput::Chord(vec![
        InputKind::PhysicalKey(KeyCode::ControlLeft),
        InputKind::Mouse(MouseButton::Right),
    ]);
    input_map.insert(SelectionAction::DistanceMeasurement, dist_chord);

    commands.init_resource::<ActionState<SelectionAction>>();
    commands.insert_resource(input_map);
    // commands.spawn(InputManagerBundle::with_map(input_map));
}

pub trait SelectionActionTrait {
    fn action() -> SelectionAction; // the Actionlike enum
}

// #[derive(Default)]
// pub struct SelectionActionInputPlugin<T: Component + SelectionActionTrait + Default> {
//     _component: std::marker::PhantomData<T>,
// }

// impl<T: Component + SelectionActionTrait + Default> Plugin for SelectionActionInputPlugin<T> {
//     fn build(&self, app: &mut App) {
//         app.add_systems(Update, selection_action_input_system::<T>);
//     }
// }

/// generic system that handles rectangular selections marked with the `T` component
pub fn selection_action_input_system<T: Component + SelectionActionTrait + Default>(
    mut commands: Commands,
    alignment_cursor: Res<CursorAlignmentPosition>,
    mut selection_actions: ResMut<ActionState<super::selection::SelectionAction>>,

    selections: Query<(Entity, &Selection), (With<T>, Without<SelectionComplete>)>,
) {
    use super::selection::SelectionAction as Action;

    if let Ok((sel_entity, _selection)) = selections.get_single() {
        if selection_actions.just_released(&Action::SelectionRelease)
            && !selection_actions.pressed(&T::action())
        {
            selection_actions.consume_all();
            commands.entity(sel_entity).insert(SelectionComplete);
        }
    } else {
        let Some(cursor) = alignment_cursor.world_pos else {
            return;
        };

        if selection_actions.just_pressed(&T::action())
            && selection_actions.get_just_released().is_empty()
        {
            commands.spawn((
                Selection {
                    start_world: cursor,
                    end_world: cursor,
                },
                T::default(),
            ));
        }
    }
}

pub fn despawn_completed_selections(
    mut commands: Commands,
    // alignment_cursor: Res<CursorAlignmentPosition>,
    // mut selection_actions: ResMut<ActionState<super::selection::SelectionAction>>,
    selections: Query<Entity, With<SelectionComplete>>,
) {
    for selection in selections.iter() {
        commands.entity(selection).despawn();
    }
}

#[derive(Component)]
pub struct Selection {
    pub start_world: DVec2,
    pub end_world: DVec2,
}

#[derive(Component)]
pub struct SelectionComplete;

#[derive(Default, Reflect, GizmoConfigGroup)]
struct SelectionGizmos {}

fn setup_selection_gizmo_config(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<SelectionGizmos>();
    config.render_layers = RenderLayers::layer(1);
}

fn update_selection(
    alignment_cursor: Res<CursorAlignmentPosition>,
    // mut endpoints: Query<(&SelectStart, &mut SelectEnd)>,
    mut endpoints: Query<&mut Selection>,
) {
    if let Some(pointer_world) = alignment_cursor.world_pos {
        for mut select in endpoints.iter_mut() {
            select.end_world = pointer_world;
        }
    }
}

// this should probably go in the module with the rect select logic,
// but that should also move, so lol
fn rectangle_zoom_selection_gizmos(
    mut gizmos: Gizmos<SelectionGizmos>,
    app_view: Res<AlignmentViewport>,

    selections: Query<
        &Selection,
        (
            Without<SelectionComplete>,
            With<super::view::RectangleZoomSelection>,
        ),
    >,
    windows: Query<&Window>,
) {
    let res = &windows.single().resolution;
    let dims = [res.width(), res.height()];

    let view = &app_view.view;

    for Selection {
        start_world,
        end_world,
    } in selections.iter()
    {
        let w0: [f64; 2] = (*start_world).into();
        let w1: [f64; 2] = (*end_world).into();
        let p0 = view.map_world_to_screen(dims, w0);
        let p1 = view.map_world_to_screen(dims, w1);

        let s0 = Vec2::new(
            p0.x - res.width() * 0.5,
            res.height() - p0.y - res.height() * 0.5,
        );
        let s1 = Vec2::new(
            p1.x - res.width() * 0.5,
            res.height() - p1.y - res.height() * 0.5,
        );

        let size = [(s1.x - s0.x).abs(), (s1.y - s0.y).abs()];

        let center = (s0 + s1) * 0.5;

        gizmos.rect_2d(center, 0.0, size.into(), LinearRgba::rgb(0.5, 0.5, 0.5));
    }
}
