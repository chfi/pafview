use super::{
    alignments::{AlignmentLayoutQuery, DefaultLayoutRoot},
    view::AlignmentViewport,
    SequencePairTile,
};
use crate::{
    app::{
        input::{
            cursor::CursorPosition, ActiveTool, RectangleSelectAction, RulerAction, UserAction,
            ViewAction,
        },
        AlignmentIndex,
    },
    Sequences,
};
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

use avian2d::parry::{
    self,
    bounding_volume::{Aabb, BoundingVolume},
    query::PointQuery,
};

struct InteractiveRulersPlugin;

impl Plugin for InteractiveRulersPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<DeleteRulerEvent>()
            .init_resource::<HeldRulerState>()
            .add_systems(
                PreUpdate,
                forward_ruler_cancel_action.in_set(crate::app::input::InputSet::BuildUserActions),
            )
            .add_systems(
                PreUpdate,
                interact_with_rulers.in_set(crate::app::input::InputSet::HandleActions),
            )
            .add_systems(
                Update,
                (copy_ruler_to_clipboard, paste_ruler_from_clipboard),
            )
            .add_systems(PreUpdate, delete_rulers)
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

#[derive(Component)]
struct CopyRulerBedpeButton;

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
    With<CopyRulerBedpeButton>,
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

        commands.entity(ruler.buttons_root).with_children(|parent| {
            parent.spawn((
                RenderLayers::layer(1),
                DeleteRulerButton,
                On::<Pointer<Click>>::send_event::<DeleteRulerEvent>(),
                SpriteBundle {
                    texture: icons.xmark.clone(),
                    transform: Transform::from_translation(Vec3::new(-15.0, 0.0, 100.0)),
                    ..default()
                },
            ));
            parent.spawn((
                RenderLayers::layer(1),
                CopyRulerBedpeButton,
                // On::<Pointer<Over>>::run(|| {
                //     println!("hovering copy button!");
                // }),
                SpriteBundle {
                    texture: icons.paste_clipboard.clone(),
                    transform: Transform::from_translation(Vec3::new(15.0, 0.0, 100.0)),
                    ..default()
                },
            ));
        });

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

fn snap_held_ruler(
    user_actions: Res<ActionState<UserAction>>,
    held_ruler: Res<HeldRulerState>,
    viewport: Res<AlignmentViewport>,

    mut endpoints: Query<&mut RulerEndpoint>,

    layouts: AlignmentLayoutQuery,

    windows: Query<&Window>,
) {
    if !user_actions.pressed(&UserAction::ModifierMinus) {
        return;
    }

    let Some(held_endpoint) = held_ruler.held_endpoint else {
        return;
    };

    let Ok(mut endpoint) = endpoints.get_mut(held_endpoint) else {
        return;
    };

    let Ok(win_size) = windows.get_single().map(|w| w.size()) else {
        return;
    };

    let wp = endpoint.world;
    let aabb_size = (10.0 / win_size.x) as f64 * viewport.view.width();

    struct Closest {
        layout_root: Entity,
        alignment_entity: Entity,
        alignment: AlignmentIndex,
        closest_world_point_on_aabb: DVec2,
        distance: f64,
    }

    let mut closest_alignment: Option<Closest> = None;

    for (layout_root, _transform, layout_handle, _) in layouts.layout_roots.iter() {
        let Some(layout) = layouts.layout_assets.get(layout_handle) else {
            continue;
        };

        let test_aabb = Aabb::from_half_extents([wp.x, wp.y].into(), [aabb_size, aabb_size].into());

        layout.layout_qbvh.aabbs_in_rect_callback(
            endpoint.world,
            [aabb_size, aabb_size],
            |seq_pair, aabb| {
                // TODO take layout transform into account here too
                let projected = aabb.project_local_point(&test_aabb.center(), true);

                // TODO idk if this is correct; don't want snapping when entirely inside an alignment... probably
                // if projected.is_inside {
                //     return true;
                // }

                let pt: nalgebra::OPoint<f64, nalgebra::Const<2>> = projected.point;
                let distance = (pt - test_aabb.center()).magnitude();

                let min_dist = closest_alignment
                    .as_ref()
                    .map(|c| c.distance)
                    .unwrap_or(std::f64::INFINITY);

                if distance < min_dist {
                    // TODO still need to check the alignments *in* the tile; don't have
                    // that qbvh yet
                    todo!();

                    let closest = Closest {
                        layout_root,
                        alignment_entity: todo!(),
                        alignment: todo!(),
                        closest_world_point_on_aabb: todo!(),
                        distance,
                    };

                    closest_alignment = Some(closest);
                }

                // let points = parry::query::closest_points(
                // )
                // compare distance to `closest_alignment`, build & store `Closest`
                // when appropriate
                todo!();
            },
        );

        // let (seq_pair, aabb) = layout.layout_qbvh.closest_tile_at_point(endpoint.world);
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

#[derive(Event)]
struct DeleteRulerEvent {
    button: Entity,
}

impl From<ListenerInput<Pointer<Click>>> for DeleteRulerEvent {
    fn from(value: ListenerInput<Pointer<Click>>) -> Self {
        DeleteRulerEvent {
            button: value.listener(),
        }
    }
}

// TODO: doesn't ensure that `HeldRulerState` is cleared on the off chance that
// a held ruler is deleted
fn delete_rulers(
    mut commands: Commands,
    mut delete_events: EventReader<DeleteRulerEvent>,
    parents: Query<&Parent>,
) {
    for event in delete_events.read() {
        let Some(ruler) = parents
            .get(event.button)
            .and_then(|r| parents.get(r.get()))
            .map(|r| r.get())
            .ok()
        else {
            continue;
        };

        commands.entity(ruler).despawn_recursive();
    }
}

// TODO: this is quite hacky and mostly a proof of concept -- this doesn't generalize
// and will need more focus information to even work with egui
fn paste_ruler_from_clipboard(
    mut commands: Commands,

    mut clipboard: ResMut<bevy_egui::EguiClipboard>,
    user_actions: Res<ActionState<UserAction>>,
    layout_root: Res<DefaultLayoutRoot>,
    layouts: AlignmentLayoutQuery,
    sequences: Res<Sequences>,

    mut last_pasted: Local<String>,
) {
    if user_actions.just_pressed(&UserAction::Paste) {
        let Some(data) = clipboard.get_contents() else {
            return;
        };

        if data == *last_pasted {
            return;
        }

        // parse as BEDPE
        let lines = data.lines();
        for line in lines {
            let fields = line.split('\t').take(6).collect::<Vec<_>>();
            if let [chrom1, start1, end1, chrom2, start2, end2] = fields.as_slice() {
                let start1 = start1.parse::<u64>().ok();
                let end1 = end1.parse::<u64>().ok();
                let start2 = start2.parse::<u64>().ok();
                let end2 = end2.parse::<u64>().ok();

                let Some(((start1, end1), (start2, end2))) =
                    (start1.zip(end1)).zip(start2.zip(end2))
                else {
                    return;
                };

                let tgt = sequences.names().get_by_left(*chrom1);
                let qry = sequences.names().get_by_left(*chrom2);

                let Some((&target, &query)) = tgt.zip(qry) else {
                    return;
                };

                // TODO transform here too (probably do this in a method on `AlignmentLayoutQuery`)
                let Ok((_, _transform, layout_handle, _)) = layouts.layout_roots.get(layout_root.0)
                else {
                    return;
                };

                let Some(tile_aabb) = layouts
                    .layout_assets
                    .get(layout_handle)
                    .and_then(|layout| layout.aabbs.get(&SequencePairTile { target, query }))
                else {
                    return;
                };
                // let Some(layout) = layouts.layout_assets.get(layout_handle) else {
                //     return;
                // };
                let offset = tile_aabb.mins;

                let p1 = DVec2::new(offset.x + start1 as f64, offset.y + start2 as f64);
                let p2 = DVec2::new(offset.x + end1 as f64, offset.y + end2 as f64);

                spawn_ruler(&mut commands, p1, p2);
                println!("spawned ruler `{data}`");
            }
        }

        *last_pasted = data;
    }
}

fn copy_ruler_to_clipboard(
    mut clipboard: ResMut<bevy_egui::EguiClipboard>,

    sequences: Res<Sequences>,

    layouts: AlignmentLayoutQuery,

    rulers: Query<&Ruler>,
    endpoints: Query<&RulerEndpoint>,

    mut click_events: EventReader<Pointer<Click>>,
    copy_buttons: Query<(Entity, &PickingInteraction), With<CopyRulerBedpeButton>>,

    parents: Query<&Parent, Or<(With<CopyRulerBedpeButton>, With<RulerButtonRoot>)>>,
) {
    // let Some((layout_root, hit_tile, local_offset)) = layouts.tile_and_local_offset_at_point(world_point)
    for (button, _interact) in copy_buttons.iter_many(click_events.read().map(|ev| ev.target())) {
        // if *interact == PickingInteraction::Pressed {
        // TODO: find *local* endpoints using seq. pair layout...
        // then format as BEDPE string & set to clipboard

        // let aa = parents.get(button).and_then(|btns| parents.get(btns.get())).and_then(|ruler| ruler);

        let ruler = parents
            .get(button)
            .and_then(|r| parents.get(r.get()))
            .and_then(|r| rulers.get(r.get()));

        let Ok(ruler) = ruler else {
            continue;
        };

        let start = endpoints.get(ruler.start).ok();
        let end = endpoints.get(ruler.end).ok();

        let Some((start, end)) = start.zip(end) else {
            continue;
        };

        let start_pt = layouts.tile_and_local_offset_at_point(start.world);
        let end_pt = layouts.tile_and_local_offset_at_point(end.world);

        // TODO: is the order right here?
        let Some(((_, start_tile, [start1, start2]), (_, end_tile, [end1, end2]))) =
            start_pt.zip(end_pt)
        else {
            continue;
        };

        // doesn't make sense to output BEDPE for a ruler crossing sequence bounds
        if start_tile != end_tile {
            continue;
        }
        let tile = start_tile;

        let chrom1 = sequences.get_name(tile.target).unwrap_or("UNKNOWNSEQ");
        let chrom2 = sequences.get_name(tile.query).unwrap_or("UNKNOWNSEQ");
        let record = format!("{chrom1}\t{start1}\t{end1}\t{chrom2}\t{start2}\t{end2}");
        clipboard.set_contents(&record);
        println!("copied ruler to clipboard: `{record}`");
        let retrieved = clipboard.get_contents();
        println!("read clipboard: {retrieved:?}");
    }
}

pub(super) struct ViewerRulersPlugin;

impl Plugin for ViewerRulersPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(InteractiveRulersPlugin)
            .add_plugins(cursor_information::CursorRulerPlugin);
        // .add_plugins(MeasurePlugin)
        // .add_systems(Startup, setup_ruler_gizmo_config)
        // .add_systems(Startup, setup)
        // .add_systems(Update, update_cursor_ruler)
        // .add_systems(PostUpdate, draw_cursor_ruler_gizmos);
    }
}

mod cursor_information {
    use crate::{
        app::{
            alignments::layout::{LayoutEntityIndex, SeqPairLayout},
            SequencePairTile,
        },
        Sequences,
    };

    use super::*;

    pub(super) struct CursorRulerPlugin;

    impl Plugin for CursorRulerPlugin {
        fn build(&self, app: &mut App) {
            app.add_systems(Startup, setup_cursor_sequence_labels)
                .add_systems(Update, update_cursor_crosshair);
        }
    }

    #[derive(Component)]
    struct CursorRulerLabels {
        target_seq: Entity,
        query_seq: Entity,
    }

    #[derive(Component)]
    struct SequenceLabel;

    // #[derive(Component)]
    // struct CursorCrosshairLine;

    fn setup_cursor_sequence_labels(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
        let text_color = Color::BLACK;

        let bundle = (
            Pickable::IGNORE,
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

        let target_seq = commands
            .spawn(bundle.clone())
            .insert((SequenceLabel, Anchor::TopLeft))
            .id();
        let query_seq = commands
            .spawn(bundle.clone())
            .insert((SequenceLabel, Anchor::TopLeft))
            .id();

        let mesh = Mesh2dHandle(meshes.add(Rectangle::new(1.0, 1.0)));
        let material = materials.add(Color::BLACK);

        commands
            .spawn((
                Pickable::IGNORE,
                CursorRulerLabels {
                    target_seq,
                    query_seq,
                },
                SpatialBundle::default(),
            ))
            .with_children(|parent| {
                parent.spawn((
                    Pickable::IGNORE,
                    RenderLayers::layer(1),
                    SpatialBundle::default(),
                    mesh.clone(),
                    material.clone(),
                ));
                parent.spawn((
                    Pickable::IGNORE,
                    RenderLayers::layer(1),
                    SpatialBundle::default(),
                    mesh,
                    material,
                ));
            });
    }

    fn update_cursor_crosshair(
        cursor: Res<CursorPosition>,
        layout_assets: Res<Assets<SeqPairLayout>>,
        sequences: Res<Sequences>,

        layout_roots: Query<(&Transform, &Handle<SeqPairLayout>, &LayoutEntityIndex)>,

        crosshair: Query<(Entity, &CursorRulerLabels, &Children)>,
        mut transforms: Query<&mut Transform, Without<Handle<SeqPairLayout>>>,
        mut texts: Query<&mut Text, With<SequenceLabel>>,
        mut label_visibilities: Query<&mut Visibility, With<SequenceLabel>>,
        windows: Query<&Window>,
    ) {
        let Ok(screen_dims) = windows.get_single().map(|w| w.size()) else {
            return;
        };

        let Ok((crosshair, cursor_labels, crosshair_children)) = crosshair.get_single() else {
            return;
        };

        let Some((world_point, screen_point)) = cursor.world.zip(cursor.screen) else {
            // hide crosshair/labels?
            return;
        };

        let mut hovered: Option<(SequencePairTile, [u64; 2])> = None;

        for (root_transform, layout_handle, _layout_entities) in layout_roots.iter() {
            let Some(layout) = layout_assets.get(layout_handle) else {
                continue;
            };

            // TODO: take layout transform into account
            let tiles = layout.layout_qbvh.aabbs_at_point(world_point);

            let Some(tile) = tiles.first() else {
                // TODO hide labels as there's no sequence tile under the cursor
                continue;
            };

            let Some(aabb) = layout.aabbs.get(tile) else {
                continue;
            };

            // TODO: transform here too
            let mins = bevy::math::DVec2::new(aabb.mins.x, aabb.mins.y);

            let local = (world_point - mins).as_u64vec2();
            hovered = Some((*tile, local.into()));
            break;
        }

        if let Ok(mut transform) = transforms.get_mut(crosshair) {
            transform.translation = Vec3::new(screen_point.x, screen_point.y, 50.0);
            // transform.translation = Vec3::new(mid.x, screen_dims.y - mid.y, 1.0)
            //     - Vec3::new(screen_dims.x, screen_dims.y, 0.0) * 0.5;
        }

        // first child is the vertical line, second horizontal
        if let Ok(mut transform) = transforms.get_mut(crosshair_children[0]) {
            transform.scale = Vec3::new(1.0, screen_dims.y * 2.0, 1.0);
        }
        if let Ok(mut transform) = transforms.get_mut(crosshair_children[1]) {
            transform.scale = Vec3::new(screen_dims.x * 2.0, 1.0, 1.0);
        }

        if let Some((tile, [local_tgt, local_qry])) = hovered {
            let build_label = |seq: crate::sequences::SeqId, offset: u64| {
                let name = sequences
                    .sequence_names
                    .get_by_right(&seq)
                    .map(|s| s.as_str())
                    .unwrap_or("<UNKNOWNSEQ>");
                format!("{name}:{offset}")
            };

            for mut vis in label_visibilities.iter_mut() {
                *vis = Visibility::Inherited;
            }

            if let Ok(mut transform) = transforms.get_mut(cursor_labels.target_seq) {
                transform.translation = Vec3::new(screen_point.x, screen_dims.y * 0.5 - 30.0, 0.0);
            }
            if let Ok(mut transform) = transforms.get_mut(cursor_labels.query_seq) {
                transform.translation = Vec3::new(10.0 - screen_dims.x * 0.5, screen_point.y, 0.0);
            }

            if let Ok(mut text) = texts.get_mut(cursor_labels.target_seq) {
                text.sections[0].value = build_label(tile.target, local_tgt);
            }
            if let Ok(mut text) = texts.get_mut(cursor_labels.query_seq) {
                text.sections[0].value = build_label(tile.query, local_qry);
            }
        } else {
            for mut vis in label_visibilities.iter_mut() {
                *vis = Visibility::Hidden;
            }
        }
    }
}
