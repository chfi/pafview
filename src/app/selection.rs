use bevy::{
    math::DVec2,
    prelude::*,
    render::view::RenderLayers,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};

use crate::gui::AppWindowStates;

use super::{view::CursorAlignmentPosition, ScreenspaceCamera};

pub(super) struct RegionSelectionPlugin;

impl Plugin for RegionSelectionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (initialize_selection, update_selection).chain());
        // app.init_resource::<RegionsOfInterest>()
        //     .add_systems(Startup, setup)
        //     .add_systems(Update, (menubar_system, settings_window));
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Component)]
pub struct Selection {
    pub start_world: DVec2,
    pub end_world: DVec2,
}

/*
#[derive(Component)]
pub struct SelectStart {
    pub world_pos: DVec2,
}

#[derive(Component)]
pub struct SelectEnd {
    pub world_pos: DVec2,
}

#[derive(Bundle)]
pub struct SelectionBundle {
    pub start: SelectStart,
    pub end: SelectEnd,
}

impl SelectionBundle {
    pub fn at(world_pos: DVec2) -> Self {
        Self {
            start: SelectStart { world_pos },
            end: SelectEnd { world_pos },
        }
    }
}
*/

fn initialize_selection(
    mut commands: Commands,
    // mut materials: ResMut<Assets<StandardMaterial>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,

    new_selection: Query<Entity, Added<Selection>>,
) {
    // create material & mesh
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

fn update_selection_transform(
    // mut meshes: ResMut<Assets<Mesh>>,
    // view: Res<AlignmentV
    world_cameras: Query<&Camera, Without<ScreenspaceCamera>>,
    screenspace_cameras: Query<&Camera, With<ScreenspaceCamera>>,
    mut selection: Query<(&Selection, &mut Transform)>,
    // selection: Query<&Mesh2dHandle, With<Selection>>,
) {
    let world_camera = world_cameras.single();
    let screenspace_camera = screenspace_cameras.single();
    // let camera =

    for (selection, mut sel_transform) in selection.iter_mut() {
        todo!();
        // let p0 = world_camera.world_to_viewport(camera_transform, world_position)

        //
    }
    // for (selection, handle_2d) in selection.iter() {
    //     let Some(mesh) = meshes.get_mut(&handle_2d.0) else {
    //         log::error!("tried to update nonexistent mesh for selection");
    //         return;
    //     };
    // }
}
