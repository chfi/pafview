use bevy::{prelude::*, render::view::RenderLayers, sprite::Anchor};
use leafwing_input_manager::action_state::ActionState;

use super::{
    selection::{Selection, SelectionComplete},
    view::CursorAlignmentPosition,
    PafViewer,
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
) {
    let Some(sp) = cursor.screen_pos else {
        return;
    };

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
    viewer: Res<PafViewer>,
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
            let tgt_name = viewer.app.sequences.sequence_names.get_by_right(&tgt_seq);
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
            let qry_name = viewer.app.sequences.sequence_names.get_by_right(&qry_seq);
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
        app.add_systems(Update, (initialize_measure, update_measure).chain());
    }
}

#[derive(Component)]
struct MeasurementSelection;

fn initialize_measure(//
) {
    //
}

fn update_measure(
    //
    mut commands: Commands,
    alignment_cursor: Res<CursorAlignmentPosition>,
    selection_actions: Query<&ActionState<super::selection::SelectionAction>>,

    selections: Query<
        (Entity, &Selection),
        (With<MeasurementSelection>, Without<SelectionComplete>),
    >,
) {
    use super::selection::SelectionAction as Action;

    let selection_actions = selection_actions.single();

    if let Ok((sel_entity, _selection)) = selections.get_single() {
        // TODO: probably want a dedicated action for "completing" a selection;
        // as it is, the full chord must be held the entire time
        if selection_actions.just_released(&Action::DistanceMeasurement) {
            println!("completing distance measurement");
            commands.entity(sel_entity).insert(SelectionComplete);
        }
    } else {
        let Some(cursor) = alignment_cursor.world_pos else {
            return;
        };

        if selection_actions.just_pressed(&Action::DistanceMeasurement) {
            println!("spawning distance measurement");
            commands.spawn((
                Selection {
                    start_world: cursor,
                    end_world: cursor,
                },
                MeasurementSelection,
            ));
        }
    }
}
