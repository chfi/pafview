use std::time::Duration;

use bevy::{prelude::*, render::view::RenderLayers};
use bevy_mod_picking::prelude::Pickable;

pub struct ToastMessagePlugin;

impl Plugin for ToastMessagePlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ToastMessageEvent>()
            .insert_resource(ToastMessageConfig {
                timeout: Duration::from_millis(6000),
            })
            .add_systems(Startup, setup_toast_message_list)
            .add_systems(
                PreUpdate,
                (
                    spawn_toast_messages,
                    close_message_interaction,
                    tick_toast_messages,
                )
                    .chain(),
            );
    }
}

#[derive(Resource)]
pub struct ToastMessageConfig {
    pub timeout: Duration,
}

#[derive(Event)]
pub struct ToastMessageEvent {
    pub header: String,
    pub body: String,
}

fn setup_toast_message_list(mut commands: Commands) {
    commands.spawn((
        ToastMessageListRoot,
        RenderLayers::layer(1),
        Pickable::IGNORE,
        NodeBundle {
            focus_policy: bevy::ui::FocusPolicy::Pass,
            style: Style {
                display: Display::Flex,
                position_type: PositionType::Absolute,
                flex_direction: FlexDirection::Column,
                right: Val::Px(30.0),
                top: Val::Px(50.0),
                min_width: Val::Px(250.0),
                min_height: Val::Px(400.0),
                max_width: Val::Px(250.0),
                max_height: Val::Percent(100.0),
                ..default()
            },
            ..default()
        },
    ));
}

fn spawn_toast_messages(
    //
    mut commands: Commands,
    mut messages: EventReader<ToastMessageEvent>,

    toast_cfg: Res<ToastMessageConfig>,

    list: Query<Entity, With<ToastMessageListRoot>>,
) {
    let Ok(list) = list.get_single() else {
        messages.clear();
        return;
    };

    for msg in messages.read() {
        let msg_root = commands
            .spawn((
                Pickable::IGNORE,
                RenderLayers::layer(1),
                ToastMessageNode {
                    timer: Timer::new(toast_cfg.timeout.clone(), TimerMode::Once),
                },
                NodeBundle {
                    focus_policy: bevy::ui::FocusPolicy::Pass,
                    background_color: BackgroundColor(Color::hsl(0.0, 0.0, 0.8)),
                    border_color: BorderColor(Color::hsl(0.0, 0.0, 0.6)),
                    border_radius: BorderRadius::all(Val::Px(5.0)),
                    style: Style {
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        width: Val::Percent(100.0),
                        max_height: Val::Px(100.0),
                        border: UiRect::all(Val::Px(3.0)),
                        margin: UiRect::all(Val::Px(5.0)),
                        padding: UiRect::all(Val::Px(8.0)),
                        ..default()
                    },
                    ..default()
                },
            ))
            .with_children(|parent| {
                parent
                    .spawn((
                        RenderLayers::layer(1),
                        ButtonBundle {
                            // node: todo!(),
                            button: Button,
                            style: Style {
                                display: Display::Block,
                                position_type: PositionType::Absolute,
                                right: Val::Px(5.0),
                                top: Val::Px(5.0),
                                width: Val::Px(50.0),
                                height: Val::Px(30.0),
                                ..default()
                            },
                            ..default()
                        },
                    ))
                    .with_children(|parent| {
                        parent.spawn((
                            RenderLayers::layer(1),
                            TextBundle::from_section(
                                "Close",
                                TextStyle {
                                    font_size: 10.0,
                                    color: Color::hsl(0.0, 0.0, 0.1),
                                    ..default()
                                },
                            ),
                        ));
                    });

                parent.spawn((
                    RenderLayers::layer(1),
                    TextBundle::from_section(
                        &msg.header,
                        TextStyle {
                            font_size: 20.0,
                            color: Color::hsl(0.0, 0.0, 0.1),
                            ..default()
                        },
                    ),
                ));

                parent.spawn((
                    RenderLayers::layer(1),
                    TextBundle::from_section(
                        &msg.body,
                        TextStyle {
                            font_size: 14.0,
                            color: Color::hsl(0.0, 0.0, 0.1),
                            ..default()
                        },
                    ),
                ));
            })
            .id();

        commands.entity(list).add_child(msg_root);
    }
}

fn tick_toast_messages(
    mut commands: Commands,
    time: Res<Time>,

    mut toasts: Query<(Entity, &mut ToastMessageNode)>,
) {
    let delta = time.delta();

    for (entity, mut toast) in toasts.iter_mut() {
        let timer = toast.timer.tick(delta);

        if timer.finished() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn close_message_interaction(
    mut toasts: Query<(&Children, &mut ToastMessageNode)>,
    buttons: Query<&Interaction, With<Button>>,
) {
    for (children, mut toast) in toasts.iter_mut() {
        for interaction in buttons.iter_many(children) {
            if *interaction == Interaction::Pressed {
                let dur = toast.timer.duration();
                toast.timer.set_elapsed(dur);
            }
            //
        }
    }
}

#[derive(Component)]
struct ToastMessageListRoot;

#[derive(Component)]
struct ToastMessageNode {
    timer: Timer,
}
