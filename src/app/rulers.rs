use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::{camera::ScalingMode, view::RenderLayers},
};
use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
};

use super::{view::CursorAlignmentPosition, PafViewer};

pub(super) struct ViewerRulersPlugin;

impl Plugin for ViewerRulersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, update_cursor_ruler);
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
struct AlignmentRuler {
    target_label: Entity,
    query_label: Entity,
}

fn setup(
    mut commands: Commands,
    // mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    // mut polylines: ResMut<Assets<Polyline>>,
) {
    let text_style = TextStyle {
        font_size: 32.0,
        color: Color::srgb(0.1, 0.1, 0.1),
        ..default()
    };

    // let target_label = commands.spawn_empty().id();
    // let query_label = commands.spawn_empty().id();

    let target_label = commands.spawn(RenderLayers::layer(1)).id();
    let query_label = commands.spawn(RenderLayers::layer(1)).id();

    let mut parent = commands.spawn((
        AlignmentRuler {
            target_label,
            query_label,
        },
        // SpatialBundle::default(),
        // RenderLayers::layer(1),
    ));

    // parent.insert_children(0, &[target_label, query_label]);

    // parent.insert_children(0, [target_label, query_label])

    // spawn text bundles on parent on update

    // parent.with_children(|parent| {
    // });

    // commands.spawn((
    //     Text2dBundle {
    //         text: Text::from_section("aoeu", text_style.clone()),
    //         transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
    //         ..default()
    //     },
    //     RenderLayers::layer(1),
    // ));

    /*
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
            RenderLayers::layer(1),
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
    */

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

fn update_cursor_ruler(
    mut commands: Commands,
    viewer: Res<PafViewer>,
    cursor: Res<CursorAlignmentPosition>,
    //
    ruler: Query<(Entity, &AlignmentRuler)>,
) {
    //
    let text_style = TextStyle {
        font_size: 32.0,
        color: Color::srgb(0.1, 0.1, 0.1),
        ..default()
    };

    let cursor_transform = cursor
        .screen_pos
        .map(|p| Transform::from_translation(Vec3::new(p.x, p.y, 0.0)))
        .unwrap_or_default();

    for (entity, ruler) in ruler.iter() {
        let t_label = ruler.target_label;
        let q_label = ruler.query_label;

        // if let Some

        commands.entity(entity).insert(cursor_transform);
        println!("cursor transform: {cursor_transform:?}");

        if let Some((tgt_seq, tgt_pos)) = cursor.target_pos {
            let tgt_name = viewer.app.sequences.sequence_names.get_by_right(&tgt_seq);
            // let tgt_text = tgt_name.map(|n| Text::from_section(n, text_style.clone()));
            let tgt_text =
                tgt_name.map(|n| Text::from_section(format!("{n}:{tgt_pos}"), text_style.clone()));

            let mut cmds = commands.entity(t_label);

            if let Some(text) = tgt_text {
                // println!("setting text: {text}");
                cmds.insert(Text2dBundle {
                    text,
                    transform: cursor_transform,
                    visibility: Visibility::Visible,
                    ..default()
                });
            } else {
                cmds.insert(Visibility::Hidden);
            }

            // commands.entity(t_label)
            //     .insert()
            //
        }

        // let t_text =

        // let mut cmds = commands.entity(entity);
        // cmds.replace_children(…)
    }
}
