use bevy::{prelude::*, render::view::RenderLayers, sprite::Anchor};
use leafwing_input_manager::action_state::ActionState;

use super::{
    selection::{Selection, SelectionActionTrait, SelectionComplete},
    view::{AlignmentViewport, CursorAlignmentPosition},
    ForegroundColor,
};

mod new_rulers {

    use bevy::{
        ecs::{event::ManualEventReader, system::EntityCommands},
        math::DVec2,
        prelude::*,
        render::view::RenderLayers,
        sprite::{Anchor, Mesh2dHandle},
    };
    use bevy_mod_picking::prelude::*;
    use leafwing_input_manager::{
        action_diff::{ActionDiff, ActionDiffEvent},
        prelude::*,
    };

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
            app.init_resource::<HeldRulerState>()
                .add_systems(
                    PreUpdate,
                    forward_ruler_cancel_action
                        .in_set(crate::app::input::InputSet::BuildUserActions),
                )
                .add_systems(
                    PreUpdate,
                    interact_with_rulers.in_set(crate::app::input::InputSet::HandleActions),
                )
                .add_systems(
                    Update,
                    (
                        (add_ruler_visuals, add_ruler_endpoint_interaction_sprite),
                        update_rulers,
                    )
                        .chain(),
                );
        }
    }

    #[derive(Component)]
    struct Ruler {
        start: Entity,
        end: Entity,

        buttons_root: Entity,
    }

    #[derive(Component)]
    struct RulerAxes {
        vertical: Entity,
        vertical_text: Entity,
        horizontal: Entity,
        horizontal_text: Entity,
    }

    #[derive(Component, Clone)]
    struct RulerAxis;

    #[derive(Component, Clone)]
    struct RulerText;

    #[derive(Component)]
    struct RulerEndpoint {
        world: DVec2,
    }

    #[derive(Component)]
    struct RulerButtonRoot;

    #[derive(Component)]
    struct DeleteRulerButton;

    #[derive(Resource, Default)]
    struct HeldRulerState {
        held_endpoint: Option<Entity>,
        original_position: Option<DVec2>,
    }

    type PositionedRulerFilter = Or<(
        With<Ruler>,
        With<RulerAxis>,
        With<RulerEndpoint>,
        With<RulerText>,
        With<RulerButtonRoot>,
        With<DeleteRulerButton>,
    )>;

    fn spawn_ruler<'a>(
        commands: &'a mut Commands,
        // text_color: impl Into<Color>,
        start_point: DVec2,
        end_point: DVec2,
    ) -> (EntityCommands<'a>, Entity, Entity) {
        let mut start = Entity::PLACEHOLDER;
        let mut end = Entity::PLACEHOLDER;

        let mut buttons_root = Entity::PLACEHOLDER;

        let mut root = commands.spawn((SpatialBundle { ..default() }, RenderLayers::layer(1)));

        root.with_children(|parent| {
            let bundle = (RenderLayers::layer(1), SpatialBundle::default());

            buttons_root = parent
                .spawn((RulerButtonRoot, SpatialBundle::default()))
                .id();
            start = parent
                .spawn(RulerEndpoint { world: start_point })
                .insert(bundle.clone())
                .id();
            end = parent
                .spawn(RulerEndpoint { world: end_point })
                .insert(bundle)
                .id();
        })
        .insert(Ruler {
            start,
            end,
            buttons_root,
        });

        (root, start, end)
    }

    fn add_ruler_endpoint_interaction_sprite(
        mut commands: Commands,

        endpoints: Query<Entity, (With<RulerEndpoint>, Without<Sprite>)>,
    ) {
        for entity in endpoints.iter() {
            //
            commands.entity(entity).insert((
                PickableBundle::default(),
                // {
                //     pickable: Pickable {
                //         should_block_lower: false,
                //         is_hoverable: false,
                //     },
                //     ..default()
                // },
                // TODO: this is ugly and hacky, and should be animated
                On::<Pointer<Over>>::target_component_mut(|_, sprite: &mut Sprite| {
                    sprite.color = Color::srgba_u8(0, 0, 0, 255);
                }),
                On::<Pointer<Out>>::target_component_mut(|_, sprite: &mut Sprite| {
                    sprite.color = Color::srgba_u8(0, 0, 0, 0);
                }),
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::srgba_u8(0, 0, 0, 255),
                        // color: Color::srgba_u8(0, 0, 0, 0), // should be invisible
                        ..default()
                    },
                    transform: Transform::from_scale(Vec3::new(10.0, 10.0, 100.0)),
                    ..default()
                },
            ));
        }
    }

    fn add_ruler_visuals(
        mut commands: Commands,

        icons: Res<crate::app::assets::Icons>,

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

        let text_color = Color::BLACK;

        let Some((mesh, material)) = ruler_mesh_material.as_ref() else {
            return;
        };

        for (root_ent, ruler) in rulers.iter() {
            let bundle = (
                RulerText,
                RenderLayers::layer(1),
                // text_bundle
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

            let mesh_bundle = (
                RulerAxis,
                RenderLayers::layer(1),
                mesh.clone(),
                material.clone(),
                SpatialBundle::default(),
            );

            let mut vertical_axis = Entity::PLACEHOLDER;
            let mut horizontal_axis = Entity::PLACEHOLDER;

            // the (visual) axis is a descendant of the ruler text,
            // since the axis is also scaled
            let vertical_text = commands
                .spawn(bundle.clone())
                .insert(Anchor::CenterRight)
                .with_children(|parent| {
                    vertical_axis = parent.spawn(mesh_bundle.clone()).id();
                })
                .id();

            let horizontal_text = commands
                .spawn(bundle)
                .insert(Anchor::CenterLeft)
                .with_children(|parent| {
                    horizontal_axis = parent.spawn(mesh_bundle.clone()).id();
                })
                .id();

            let axes = RulerAxes {
                vertical: vertical_axis,
                vertical_text,
                horizontal: horizontal_axis,
                horizontal_text,
            };

            /*
            commands.entity(ruler.buttons_root).with_children(|parent| {
                parent.spawn((
                    RenderLayers::layer(1),
                    DeleteRulerButton,
                    SpriteBundle {
                        // sprite: todo!(),
                        // transform: todo!(),
                        // global_transform: todo!(),
                        texture: icons.xmark.clone(),
                        // visibility: todo!(),
                        ..default()
                    },
                ));
            });
            */

            commands
                .entity(root_ent)
                .push_children(&[vertical_text, horizontal_text])
                .insert(axes);
            // .insert((axes, mesh.clone(), material.clone()));
        }
    }

    fn update_rulers(
        view: Res<AlignmentViewport>,

        rulers: Query<(Entity, &Ruler, &RulerAxes, &Children)>,
        endpoints: Query<&RulerEndpoint>,
        mut transforms: Query<&mut Transform, PositionedRulerFilter>,

        mut texts: Query<(&mut Text, &mut Anchor), With<RulerText>>,

        windows: Query<&Window>,
    ) {
        let Ok(screen_dims) = windows.get_single().map(|w| w.size()) else {
            return;
        };

        for (ruler_entity, ruler, axes, children) in rulers.iter() {
            // println!("ruler root has {} children", children.len());

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

            // set the endpoints to the screen position corresponding to their world position
            if let Ok(mut transform) = transforms.get_mut(ruler.start) {
                let x = if start_s.x > end_s.x { width } else { -width };
                let y = if start_s.y > end_s.y { -height } else { height };
                transform.translation = Vec3::new(x, y, 0.0) * 0.5;
            }
            if let Ok(mut transform) = transforms.get_mut(ruler.end) {
                let x = if start_s.x > end_s.x { -width } else { width };
                let y = if start_s.y > end_s.y { height } else { -height };
                transform.translation = Vec3::new(x, y, 0.0) * 0.5;
            }

            // the root of the ruler is at the middle of the rectangle defined by its endpoints
            if let Ok(mut transform) = transforms.get_mut(ruler_entity) {
                transform.translation = Vec3::new(mid.x, screen_dims.y - mid.y, 1.0)
                    - Vec3::new(screen_dims.x, screen_dims.y, 0.0) * 0.5;
            }

            // the text labels are placed on the outside of the corresponding rectangle side,
            // with the text anchors set accordingly
            if let Some(((mut text, mut anchor), mut transform)) = texts
                .get_mut(axes.vertical_text)
                .ok()
                .zip(transforms.get_mut(axes.vertical_text).ok())
            {
                text.sections[0].value = format!("{}", (start - end).y.abs().round() as u64);

                transform.translation = Vec3::new(width * 0.5, 0.0, 0.0);
                if end_s.x > start_s.x {
                    transform.translation.x *= -1.0;
                    *anchor = Anchor::CenterRight;
                } else {
                    *anchor = Anchor::CenterLeft;
                }
            }

            // if let Ok(mut text) = texts.get_mut(axes.horizontal_text) {
            if let Some(((mut text, mut anchor), mut transform)) = texts
                .get_mut(axes.horizontal_text)
                .ok()
                .zip(transforms.get_mut(axes.horizontal_text).ok())
            {
                text.sections[0].value = format!("{}", (start - end).x.abs().round() as u64);

                transform.translation = Vec3::new(0.0, height * 0.5, 0.0);
                if end_s.y > start_s.y {
                    transform.translation.y *= -1.0;
                    *anchor = Anchor::TopCenter;
                } else {
                    *anchor = Anchor::BottomCenter;
                };
            }

            // the axes are children of their corresponding texts, and only need to be
            // scaled to display appropriately as their parent transforms are positioned
            if let Ok(mut transform) = transforms.get_mut(axes.vertical) {
                transform.scale = Vec3::new(2.0, height, 1.0);
            }

            if let Ok(mut transform) = transforms.get_mut(axes.horizontal) {
                transform.scale = Vec3::new(width, 2.0, 1.0);
            }

            // the button(s) are on a child of the root entity, so its transform must also be set
            if let Ok(mut transform) = transforms.get_mut(ruler.buttons_root) {
                transform.translation = Vec3::new(width * 0.5, -1.0 * (height * 0.5 + 20.0), 0.0);
            }
        }
    }

    fn update_ruler_buttons(
        // held_ruler: Res<HeldRulerState>,
        rulers: Query<&Ruler>,
        mut visibilities: Query<&mut Visibility>,
    ) {
        for ruler in rulers.iter() {
            if let Ok(mut visibility) = visibilities.get_mut(ruler.buttons_root) {
                // let new_vis = ruler.
            }
        }
    }

    fn forward_ruler_cancel_action(
        mut user_actions: ResMut<ActionState<UserAction>>,
        mut ruler_actions: ResMut<ActionState<RulerAction>>,

        held_ruler: Res<HeldRulerState>,
    ) {
        if held_ruler.held_endpoint.is_some() {
            let cancel_data = user_actions.button_data_mut_or_default(&UserAction::Cancel);
            let ruler_data = ruler_actions
                .button_data_mut_or_default(&RulerAction(RectangleSelectAction::CancelSelect));

            *ruler_data = cancel_data.clone();
            *cancel_data = leafwing_input_manager::action_state::ButtonData::default();
        }
    }

    fn interact_with_rulers(
        mut commands: Commands,
        ruler_actions: Res<ActionState<RulerAction>>,
        cursor: Res<CursorPosition>,

        mut endpoints: Query<(Entity, &mut RulerEndpoint, &Parent)>,

        mut click_events: EventReader<Pointer<Click>>,

        mut held_ruler: ResMut<HeldRulerState>,
        // mut held_endpoint: Local<Option<Entity>>,
    ) {
        if let Some((held, world)) = held_ruler.held_endpoint.zip(cursor.world) {
            // if let Some((&held, world)) = held_endpoint.as_ref().zip(cursor.world) {
            // move the endpoint... maybe... idk
            if let Ok((_, mut endpoint, _)) = endpoints.get_mut(held) {
                endpoint.world = world;
            }
        }

        let mut picked_endpoint = None;
        // TODO: this isn't exactly correct; doesn't take distance from camera into account
        // (but all rulers are on the same plane currently anyway; might be better to use
        // the closest in the plane)
        // if held_ruler.held_endpoint.is_none() {
        for event in click_events.read() {
            if let Ok((target, endpoint, _)) = endpoints.get(event.target()) {
                if let Some(held) = held_ruler.held_endpoint {
                    if target == held {
                        println!("dropping endpoint {held:?}");
                        held_ruler.held_endpoint = None;
                        held_ruler.original_position = None;
                    }
                } else {
                    println!("picking up endpoint {target:?}");
                    held_ruler.held_endpoint = Some(target);
                    held_ruler.original_position = Some(endpoint.world);
                    picked_endpoint = Some(target);
                    break;
                }
            }
        }
        click_events.clear();

        if ruler_actions.just_pressed(&RulerAction(RectangleSelectAction::StartOrEndSelect)) {
            if held_ruler.held_endpoint.is_none() {
                // spawn both endpoints, placing them under the cursor, but setting one of them to be "held"
                if picked_endpoint.is_none() {
                    if let Some(pos) = cursor.world {
                        let (_root, _start, end) = spawn_ruler(&mut commands, pos, pos);
                        println!("placing endpoint {end:?}");
                        held_ruler.held_endpoint = Some(end);
                        held_ruler.original_position = None;
                    }
                }
            } else if held_ruler.held_endpoint.is_some() && picked_endpoint.is_none() {
                // this.. is probably not necessary
                held_ruler.held_endpoint = None;
                held_ruler.original_position = None;
            }
        }

        if let Some(held_endpoint) = held_ruler.held_endpoint {
            if let Ok((_, mut endpoint, parent)) = endpoints.get_mut(held_endpoint) {
                if let Some(world) = cursor.world {
                    endpoint.world = world;
                }

                if ruler_actions.just_pressed(&RulerAction(RectangleSelectAction::CancelSelect)) {
                    held_ruler.held_endpoint = None;
                    if let Some(origin) = held_ruler.original_position.take() {
                        // the endpoint had a position when it was picked up,
                        // so move it there
                        endpoint.world = origin;
                    } else {
                        // the endpoint didn't have a position (i.e. the ruler is being placed),
                        // so remove the entire thing
                        commands.entity(parent.get()).despawn_recursive();
                    }
                }
            }
        }
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
