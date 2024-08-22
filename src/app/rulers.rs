use bevy::{prelude::*, render::view::RenderLayers, sprite::Anchor};
use leafwing_input_manager::action_state::ActionState;

use super::{
    selection::{Selection, SelectionActionTrait, SelectionComplete},
    view::{AlignmentViewport, CursorAlignmentPosition},
};

pub(super) struct ViewerRulersPlugin;

impl Plugin for ViewerRulersPlugin {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<RulerGizmos>()
            .add_plugins(MeasurePlugin)
            .add_systems(Startup, setup_ruler_gizmo_config)
            .add_systems(Startup, setup)
            .add_systems(Update, update_cursor_ruler)
            .add_systems(PostUpdate, draw_cursor_ruler_gizmos);
    }
}

#[derive(Default, Reflect, GizmoConfigGroup)]
struct RulerGizmos {}

fn setup_ruler_gizmo_config(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<RulerGizmos>();
    config.render_layers = RenderLayers::layer(1);
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
    let target_label = commands.spawn(RenderLayers::layer(1)).id();
    let query_label = commands.spawn(RenderLayers::layer(1)).id();

    commands.spawn((AlignmentRuler {
        target_label,
        query_label,
    },));
}

fn draw_cursor_ruler_gizmos(
    mut gizmos: Gizmos<RulerGizmos>,
    cursor: Res<CursorAlignmentPosition>,
    windows: Query<&Window>,

    measure_selection: Query<&Selection, With<MeasurementSelection>>,
) {
    let Some(sp) = cursor.screen_pos else {
        return;
    };

    if !measure_selection.is_empty() {
        return;
    }

    let res = &windows.single().resolution;

    let color = Color::srgb(0.1, 0.1, 0.1);

    gizmos.linestrip_2d(
        [[sp.x, -res.height()].into(), [sp.x, res.height()].into()],
        color,
    );
    gizmos.linestrip_2d(
        [[-res.width(), sp.y].into(), [res.width(), sp.y].into()],
        color,
    );
}

fn update_cursor_ruler(
    mut commands: Commands,
    sequences: Res<crate::Sequences>,
    // viewer: Res<PafViewer>,
    cursor: Res<CursorAlignmentPosition>,

    ruler: Query<(Entity, &AlignmentRuler)>,
    windows: Query<&Window>,
) {
    let text_style = TextStyle {
        font_size: 22.0,
        color: Color::srgb(0.0, 0.0, 0.0),
        ..default()
    };

    let res = &windows.single().resolution;

    let cursor_transform = cursor
        .screen_pos
        .map(|p| Transform::from_translation(Vec3::new(p.x, p.y, 0.0)))
        .unwrap_or_default();

    for (entity, ruler) in ruler.iter() {
        let t_label = ruler.target_label;
        let q_label = ruler.query_label;

        commands.entity(entity).insert(cursor_transform);

        if let Some((tgt_seq, tgt_pos)) = cursor.target_pos {
            let tgt_name = sequences.sequence_names.get_by_right(&tgt_seq);
            let tgt_text = tgt_name
                .map(|n| Text::from_section(format!("TGT {n}:{tgt_pos}"), text_style.clone()));

            let mut cmds = commands.entity(t_label);
            let mut transform = cursor_transform;
            // TODO: still need to get the menu bar offset
            transform.translation.y = res.height() * 0.5 - 20.0;

            if let Some(text) = tgt_text {
                cmds.insert(Text2dBundle {
                    text,
                    text_anchor: Anchor::TopLeft,
                    transform,
                    visibility: Visibility::Visible,
                    ..default()
                });
            }
        } else {
            commands.entity(t_label).insert(Visibility::Hidden);
        }

        if let Some((qry_seq, qry_pos)) = cursor.query_pos {
            let qry_name = sequences.sequence_names.get_by_right(&qry_seq);
            let qry_text = qry_name
                .map(|n| Text::from_section(format!("QRY {n}:{qry_pos}"), text_style.clone()));

            let mut cmds = commands.entity(q_label);
            let mut transform = cursor_transform;
            transform.translation.x = -res.width() * 0.5;

            if let Some(text) = qry_text {
                cmds.insert(Text2dBundle {
                    text,
                    text_anchor: Anchor::CenterLeft,
                    transform,
                    visibility: Visibility::Visible,
                    ..default()
                });
            }
        } else {
            commands.entity(q_label).insert(Visibility::Hidden);
        }
    }
}

pub(super) struct MeasurePlugin;

impl Plugin for MeasurePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_measure_text).add_systems(
            Update,
            (
                super::selection::selection_action_input_system::<MeasurementSelection>,
                update_measure_display,
            )
                .chain(),
        );
    }
}

#[derive(Component, Default)]
struct MeasurementSelection;

impl SelectionActionTrait for MeasurementSelection {
    fn action() -> super::selection::SelectionAction {
        super::selection::SelectionAction::DistanceMeasurement
    }
}

#[derive(Component, Clone, Copy)]
enum MeasureRuler {
    Target,
    Query,
}

fn setup_measure_text(mut commands: Commands) {
    commands.spawn((
        MeasureRuler::Query,
        RenderLayers::layer(1),
        Text2dBundle {
            text: Text::from_section(
                "",
                TextStyle {
                    color: Color::BLACK,
                    ..default()
                },
            ),
            text_anchor: Anchor::CenterRight,
            visibility: Visibility::Hidden,
            ..default()
        },
    ));

    commands.spawn((
        MeasureRuler::Target,
        RenderLayers::layer(1),
        Text2dBundle {
            text: Text::from_section(
                "",
                TextStyle {
                    color: Color::BLACK,
                    ..default()
                },
            ),
            text_anchor: Anchor::BottomCenter,
            visibility: Visibility::Hidden,
            ..default()
        },
    ));
}

fn update_measure_display(
    alignment_view: Res<AlignmentViewport>,
    windows: Query<&Window>,

    mut gizmos: Gizmos<RulerGizmos>,

    mut measure_display_query: Query<(
        &mut Transform,
        &mut Text,
        &mut Anchor,
        &mut Visibility,
        &MeasureRuler,
    )>,

    selections: Query<
        (Entity, &Selection),
        (With<MeasurementSelection>, Without<SelectionComplete>),
    >,
) {
    let window = windows.single();
    let win_size = window.resolution.size();

    let Ok((_sel_entity, selection)) = selections.get_single() else {
        measure_display_query
            .iter_mut()
            .for_each(|(_, _, _, mut vis, _)| *vis = Visibility::Hidden);

        return;
    };

    let color = Color::srgb(0.05, 0.05, 0.05);

    let view = &alignment_view.view;

    let s0 = view.map_world_to_screen(win_size, selection.start_world.to_array());
    let s1 = view.map_world_to_screen(win_size, selection.end_world.to_array());

    let s0 = Vec2::new(
        s0.x - win_size.x * 0.5,
        win_size.y - s0.y - win_size.y * 0.5,
    );
    let s1 = Vec2::new(
        s1.x - win_size.x * 0.5,
        win_size.y - s1.y - win_size.y * 0.5,
    );

    gizmos.linestrip_2d([[s0.x, s0.y].into(), [s0.x, s1.y].into()], color);
    gizmos.linestrip_2d([[s0.x, s1.y].into(), [s1.x, s1.y].into()], color);

    // let s0 = Vec2::from(*s0.as_array());
    // let s1 = Vec2::from(*s1.as_array());

    let x_dist = (selection.start_world.x - selection.end_world.x).abs();
    let y_dist = (selection.start_world.y - selection.end_world.y).abs();

    // TODO update anchor & ensure that the text always stays on screen
    // while not overlapping the rulers
    for (mut transform, mut text, mut _anchor, mut vis, measure) in measure_display_query.iter_mut()
    {
        *vis = Visibility::Visible;

        match measure {
            MeasureRuler::Target => {
                text.sections[0].value = format!("{}", x_dist.floor());
                transform.translation.x = (s0.x + s1.x) * 0.5;
                transform.translation.y = s1.y;
            }
            MeasureRuler::Query => {
                text.sections[0].value = format!("{}", y_dist.floor());
                transform.translation.x = s0.x;
                transform.translation.y = (s0.y + s1.y) * 0.5;
            }
        }
    }
}
