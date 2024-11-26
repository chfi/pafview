use bevy::{prelude::*, render::view::RenderLayers, sprite::Anchor};
use leafwing_input_manager::action_state::ActionState;

use super::{
    selection::{Selection, SelectionActionTrait, SelectionComplete},
    view::{AlignmentViewport, CursorAlignmentPosition},
    ForegroundColor,
};

mod new_rulers {

    use bevy::{
        ecs::system::EntityCommands,
        math::DVec2,
        prelude::*,
        render::view::RenderLayers,
        sprite::{Anchor, Mesh2dHandle},
    };
    use leafwing_input_manager::prelude::*;

    // use super::super::{
    //     selection::{Selection, SelectionActionTrait, SelectionComplete},
    //     view::{AlignmentViewport, CursorAlignmentPosition},
    //     ForegroundColor,
    // };

    use crate::app::{
        input::{
            cursor::CursorPosition, ActiveTool, RectangleSelectAction, RulerAction, UserAction,
            ViewAction,
        },
        view::AlignmentViewport,
    };

    pub struct InteractiveRulersPlugin;

    impl Plugin for InteractiveRulersPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(
                PreUpdate,
                update_rulers.in_set(crate::app::input::InputSet::HandleActions),
            );
            // .add_systems(Update, draw_ruler_gizmos);
            // app.add_systems();
        }
    }

    #[derive(Component)]
    struct Ruler {
        start: Entity,
        end: Entity,
    }

    #[derive(Component)]
    struct RulerAxes {
        vertical: Entity,
        horizontal: Entity,
    }

    #[derive(Component)]
    struct RulerAxis;

    #[derive(Component)]
    struct RulerEndpoint {
        world: DVec2,
    }

    /*
    fn draw_ruler_gizmos(
        mut gizmos: Gizmos<super::RulerGizmos>,
        viewport: Res<AlignmentViewport>,
        fg_color: Res<crate::app::ForegroundColor>,

        endpoints: Query<&RulerEndpoint>,
        rulers: Query<(Entity, &Ruler)>,

        windows: Query<&Window>,
    ) {
        let Ok(window) = windows.get_single() else {
            return;
        };
        let win_size = window.size();

        for (entity, ruler) in rulers.iter() {
            let Some((start, end)) = endpoints
                .get(ruler.start)
                .ok()
                .zip(endpoints.get(ruler.end).ok())
            else {
                continue;
            };

            let s0 = viewport.view.map_world_to_screen(win_size, start.world);
            let s1 = viewport.view.map_world_to_screen(win_size, end.world);

            let corner = if s0.y > s1.y {
                [s0.x, s1.y]
            } else {
                [s1.x, s0.y]
            };

            println!("drawing ruler: {s0:?} - {corner:?} - {s1:?}");
            // gizmos.linestrip_2d(
            //     [[s0.x, s0.y].into(), corner.into(), [s1.x, s1.y].into()],
            //     fg_color.0,
            // );

            // gizmos.linestrip_2d([[[]]])

            // gizmos.linestrip_2d([[sp.x, -size.y].into(), [sp.x, size.y].into()], color);
            // gizmos.linestrip_2d([[-size.x, sp.y].into(), [size.x, sp.y].into()], color);
        }
    }
    */

    fn spawn_ruler<'a>(
        commands: &'a mut Commands,
        text_color: impl Into<Color>,
        start_point: DVec2,
        end_point: DVec2,
    ) -> (EntityCommands<'a>, Entity, Entity) {
        let mut start = Entity::PLACEHOLDER;
        let mut end = Entity::PLACEHOLDER;

        let mut root = commands.spawn_empty();

        root.with_children(|parent| {
            let bundle = (
                RenderLayers::layer(1),
                Text2dBundle {
                    text: Text::from_section(
                        "",
                        TextStyle {
                            color: text_color.into(),
                            ..default()
                        },
                    ),
                    text_anchor: Anchor::BottomCenter,
                    visibility: Visibility::Visible,
                    ..default()
                },
            );

            start = parent
                .spawn(RulerEndpoint { world: start_point })
                .insert(bundle.clone())
                .id();
            end = parent
                .spawn(RulerEndpoint { world: end_point })
                .insert(bundle)
                .id();
        })
        .insert(Ruler { start, end });

        (root, start, end)
    }

    fn add_ruler_visuals(
        mut commands: Commands,

        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,

        rulers: Query<(Entity, &Ruler), Without<RulerAxes>>,

        mut ruler_mesh_material: Local<Option<(Mesh2dHandle, Handle<ColorMaterial>)>>,
    ) {
        if ruler_mesh_material.is_none() {
            let mesh = Mesh2dHandle(meshes.add(Rectangle::new(1.0, 1.0)));
            let material = materials.add(Color::BLACK);
            *ruler_mesh_material = Some((mesh, material));
        }

        let Some((mesh, material)) = ruler_mesh_material.as_ref() else {
            return;
        };

        for (root_ent, _endpoints) in rulers.iter() {
            let vertical = commands
                .spawn((
                    RulerAxis,
                    RenderLayers::layer(1),
                    mesh.clone(),
                    material.clone(),
                    SpatialBundle::default(),
                ))
                .id();

            let horizontal = commands
                .spawn((
                    RulerAxis,
                    RenderLayers::layer(1),
                    mesh.clone(),
                    material.clone(),
                    SpatialBundle::default(),
                ))
                .id();

            let axes = RulerAxes {
                vertical,
                horizontal,
            };

            commands.entity(root_ent).insert(axes);
        }
    }

    fn update_ruler_axes(
        view: Res<AlignmentViewport>,

        rulers: Query<(&Ruler, &RulerAxes)>,
        endpoints: Query<&RulerEndpoint>,
        mut transforms: Query<&mut Transform, With<RulerAxis>>,

        windows: Query<&Window>,
    ) {
        let Ok(screen_dims) = windows.get_single().map(|w| w.size()) else {
            return;
        };

        for (ruler, axes) in rulers.iter() {
            let start = endpoints.get(ruler.start).map(|p| p.world);
            let end = endpoints.get(ruler.end).map(|p| p.world);

            let Some((start, end)) = start.ok().zip(end.ok()) else {
                continue;
            };

            let start_s = view.view.map_world_to_screen(screen_dims, start);
            let end_s = view.view.map_world_to_screen(screen_dims, end);

            let dims = (start_s - end_s).abs();
            let mid = (start_s + end_s) * 0.5;

            let height = dims.y;
            let width = dims.x;

            // the axes are *not* children of the ruler, so they're not influenced
            // by the transform hierarchy
            if let Ok(mut transform) = transforms.get_mut(axes.vertical) {
                transform.scale = Vec3::new(2.0, height, 1.0);
                transform.translation = Vec3::new(start_s.x, mid.y, 1.0);
            }

            if let Ok(mut transform) = transforms.get_mut(axes.horizontal) {
                transform.scale = Vec3::new(width, 2.0, 1.0);
                transform.translation = Vec3::new(mid.x, end_s.y, 1.0);
            }
        }
    }

    // #[derive(Component)]
    // struct HeldByCursor;

    // #[derive(Component, Clone, Copy, PartialEq)]
    // struct AtWorldPoint(DVec2);

    // TODO: probably better to use a marker component to track what is held,
    // and split this into two systems (handle_actions & update_rulers);
    // the `held_endpoint` `Local` is just to get started
    fn update_rulers(
        mut commands: Commands,
        ruler_actions: Res<ActionState<RulerAction>>,
        cursor: Res<CursorPosition>,

        mut endpoints: Query<(Entity, &mut RulerEndpoint, &Parent)>,

        mut held_endpoint: Local<Option<Entity>>,
    ) {
        if let Some((&held, world)) = held_endpoint.as_ref().zip(cursor.world) {
            // move the endpoint... maybe... idk
            if let Ok((_, mut endpoint, _)) = endpoints.get_mut(held) {
                endpoint.world = world;
            }
        }

        let mut picked_endpoint = None;
        if let Some(pos) = cursor.world {
            for (entity, endpoint, _) in endpoints.iter() {
                if (endpoint.world - pos).length_squared() < 100.0 {
                    picked_endpoint = Some(entity);
                }
            }
        }

        if ruler_actions.just_pressed(&RulerAction(RectangleSelectAction::StartOrEndSelect)) {
            if held_endpoint.is_none() {
                // spawn both endpoints, placing them under the cursor, but setting one of them to be "held"

                if picked_endpoint.is_none() {
                    if let Some(pos) = cursor.world {
                        let mut start = Entity::PLACEHOLDER;
                        let mut end = Entity::PLACEHOLDER;

                        let root = commands
                            .spawn_empty()
                            .with_children(|parent| {
                                start = parent.spawn(RulerEndpoint { world: pos }).id();
                                end = parent.spawn(RulerEndpoint { world: pos }).id();
                            })
                            .insert(Ruler { start, end })
                            .id();

                        *held_endpoint = Some(end);
                    }
                } else {
                    *held_endpoint = picked_endpoint;
                }
                // todo!();
            } else if let Some(held) = held_endpoint.take() {
                // TODO: place the held endpoint
                // don't need to do anything yet
            }
        }
        //
    }
}

pub(super) struct ViewerRulersPlugin;

impl Plugin for ViewerRulersPlugin {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<RulerGizmos>()
            .add_plugins(new_rulers::InteractiveRulersPlugin);
        // .add_plugins(MeasurePlugin)
        // .add_systems(Startup, setup_ruler_gizmo_config)
        // .add_systems(Startup, setup)
        // .add_systems(Update, update_cursor_ruler)
        // .add_systems(PostUpdate, draw_cursor_ruler_gizmos);
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

    fg_color: Res<ForegroundColor>,
    measure_selection: Query<&Selection, With<MeasurementSelection>>,
) {
    let Some(sp) = cursor.screen_pos else {
        return;
    };

    if !measure_selection.is_empty() {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };
    let res = &window.resolution;

    let color = fg_color.0;

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

    fg_color: Res<ForegroundColor>,
    ruler: Query<(Entity, &AlignmentRuler)>,
    windows: Query<&Window>,
) {
    let text_style = TextStyle {
        font_size: 22.0,
        color: fg_color.0,
        ..default()
    };

    let Ok(window) = windows.get_single() else {
        return;
    };
    let res = &window.resolution;

    let cursor_transform = cursor
        .screen_pos
        .map(|p| Transform::from_translation(Vec3::new(p.x, p.y, 0.0)))
        .unwrap_or_default();

    for (entity, ruler) in ruler.iter() {
        let t_label = ruler.target_label;
        let q_label = ruler.query_label;

        commands.entity(entity).insert(cursor_transform);

        // NB commenting to delete `target_pos` & `query_pos` from `CursorAlignmentPosition`
        // (since it doesn't make sense anymore)
        /*
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
         */
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

fn setup_measure_text(mut commands: Commands, fg_color: Res<ForegroundColor>) {
    commands.spawn((
        MeasureRuler::Query,
        RenderLayers::layer(1),
        Text2dBundle {
            text: Text::from_section(
                "",
                TextStyle {
                    color: fg_color.0,
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
                    color: fg_color.0,
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
    fg_color: Res<ForegroundColor>,

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
    let Ok(window) = windows.get_single() else {
        return;
    };
    let win_size = window.resolution.size();

    let Ok((_sel_entity, selection)) = selections.get_single() else {
        measure_display_query
            .iter_mut()
            .for_each(|(_, _, _, mut vis, _)| *vis = Visibility::Hidden);

        return;
    };

    let color = fg_color.0;

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
