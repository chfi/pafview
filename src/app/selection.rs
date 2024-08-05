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
        app.init_gizmo_group::<SelectionGizmos>()
            .add_plugins(InputManagerPlugin::<SelectionAction>::default())
            .add_systems(Startup, setup_selection_input_map)
            .add_systems(Startup, setup_selection_gizmo_config)
            // .add_systems(Update, right_click_selection_test)
            .add_systems(Update, update_selection)
            // .add_systems(Update, (initialize_selection, update_selection).chain())
            .add_systems(
                Update,
                rectangle_zoom_selection_gizmos.after(update_selection),
            );
        // app.init_resource::<RegionsOfInterest>()
        //     .add_systems(Startup, setup)
        //     .add_systems(Update, (menubar_system, settings_window));
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
pub enum SelectionAction {
    ZoomRectangle,
    DistanceMeasurement,
}

fn setup_selection_input_map(mut commands: Commands) {
    let mut input_map: InputMap<SelectionAction> = InputMap::default();

    input_map.insert(SelectionAction::ZoomRectangle, MouseButton::Right);

    let dist_chord = UserInput::Chord(vec![
        InputKind::PhysicalKey(KeyCode::ControlLeft),
        InputKind::Mouse(MouseButton::Right),
    ]);
    input_map.insert(SelectionAction::DistanceMeasurement, dist_chord);

    commands.spawn(InputManagerBundle::with_map(input_map));
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

fn initialize_selection(
    mut commands: Commands,
    // mut materials: ResMut<Assets<StandardMaterial>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,

    new_selection: Query<Entity, Added<Selection>>,
) {
    // create material & mesh
    /*
    let mesh = Mesh2dHandle(meshes.add(Rectangle::new(1.0, 1.0)));
    let material = materials.add(Color::srgba(0.7, 0.0, 0.0, 0.3));

    for select_ent in new_selection.iter() {
        if let Some(mut cmds) = commands.get_entity(select_ent) {
            cmds.insert((
                RenderLayers::layer(1),
                MaterialMesh2dBundle {
                    mesh: mesh.clone(),
                    material: material.clone(),
                    ..default()
                },
            ));
        }
    }
    */
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

/*
fn right_click_selection_test(
    mut commands: Commands,
    alignment_cursor: Res<CursorAlignmentPosition>,
    mouse_button: Res<ButtonInput<MouseButton>>,

    selections: Query<(Entity, &Selection), Without<SelectionComplete>>,
) {
    if let Ok((sel_entity, _selection)) = selections.get_single() {
        // let Some(cursor) = alignment_cursor.world_pos else {
        //     return;
        // };

        if mouse_button.just_released(MouseButton::Right) {
            commands.entity(sel_entity).insert(SelectionComplete);
        }
    } else {
        let Some(cursor) = alignment_cursor.world_pos else {
            return;
        };

        if mouse_button.just_pressed(MouseButton::Right) {
            commands.spawn(Selection {
                start_world: cursor,
                end_world: cursor,
            });
        }
    }
}
*/

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
    // gizmos.grid_2d(
    //     [0., 0.].into(),
    //     0.0,
    //     [10, 10].into(),
    //     [20.0, 20.0].into(),
    //     LinearRgba::rgb(0.5, 0.5, 0.5),
    // );

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

        // let pos = [s0.x.min(s1.x), s0.y.min(s1.y)];

        let size = [(s1.x - s0.x).abs(), (s1.y - s0.y).abs()];

        let center = (s0 + s1) * 0.5;

        gizmos.rect_2d(center, 0.0, size.into(), LinearRgba::rgb(0.5, 0.5, 0.5));

        // gizmos.circle_2d(s0, 5.0, LinearRgba::rgb(0.8, 0.0, 0.0));
        // gizmos.circle_2d(s1, 5.0, LinearRgba::rgb(0.8, 0.0, 0.0));
    }
}
