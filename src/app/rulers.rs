use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::camera::ScalingMode,
};
use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
};

pub(super) struct ViewerRulersPlugin;

impl Plugin for ViewerRulersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup);
        // app.add_systems(Startup, setup)
        // .add_systems(PreUpdate, update_viewport_for_window_resize)
        // .add_systems(
        //     Update,
        //     (
        //         update_viewport_for_window_resize,
        //         input_update_viewport,
        //         update_camera_from_viewport,
        //     )
        //         .chain()
        //         .before(bevy::render::camera::camera_system::<OrthographicProjection>),
        // );
    }
}

#[derive(Component)]
struct AlignmentRuler;

fn setup(
    mut commands: Commands,
    // mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    // mut polylines: ResMut<Assets<Polyline>>,
) {
    commands
        .spawn((
            AlignmentRuler,
            NodeBundle {
                // node: todo!(),
                style: Style {
                    display: Display::Block,
                    width: Val::Percent(100.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
                focus_policy: bevy::ui::FocusPolicy::Pass,
                // visibility: Visibility::Hidden,
                visibility: Visibility::Visible,
                // background_color: Color::BLACK.into(),
                // background_color: todo!(),
                ..default()
            },
        ))
        .with_children(|parent| {
            // horizontal
            parent.spawn(NodeBundle {
                style: Style {
                    display: Display::Block,
                    width: Val::Percent(100.0),
                    // left: Val::Percent(0.0),
                    // right: Val::Percent(0.0),
                    top: Val::Percent(50.0),
                    border: UiRect {
                        top: Val::Px(1.5),
                        bottom: Val::Px(1.5),
                        ..default()
                    },
                    ..default()
                },
                border_color: Color::BLACK.into(),
                background_color: Color::BLACK.into(),
                ..default()
            });

            // vertical
        });

    /*
    let material = polyline_materials.add(PolylineMaterial {
        width: 1.0,
        color: Color::srgba(0.0, 0.0, 0.0, 0.5).into(),
        depth_bias: 0.0,
        perspective: false,
    });

    commands.spawn((
        AlignmentRuler,
        PolylineBundle {
            polyline: todo!(),
            material,
            transform: todo!(),
            global_transform: todo!(),
            visibility: todo!(),
            inherited_visibility: todo!(),
            view_visibility: todo!(),
        },
    ));
    */

    // commands.spawn((AlignmentRuler,
    //                 ));
}
