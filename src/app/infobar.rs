use bevy::prelude::*;
use bevy_mod_picking::prelude::ListenerInput;

pub struct InfobarPlugin;

impl Plugin for InfobarPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<InfobarAlignmentEvent>()
            .add_systems(Startup, setup_infobar)
            .add_systems(Update, update_infobar);
    }
}

// #[derive(Debug, Resource)]
// struct InfobarEntity(Entity);

#[derive(Component)]
struct Infobar;

#[derive(Component)]
struct InfobarText;

#[derive(Debug, Clone, Event)]
pub struct InfobarAlignmentEvent {
    alignment_entity: Entity,
    // pub alignment: super::alignments::Alignment,
    cursor_over: bool,
}

fn setup_infobar(
    mut commands: Commands,
    //
) {
    let infobar = commands
        .spawn((
            Infobar,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.0),
                    height: Val::Px(30.0),
                    border: UiRect {
                        top: Val::Px(2.0),
                        ..default()
                    },
                    bottom: Val::Px(0.0),
                    position_type: PositionType::Absolute,
                    align_self: AlignSelf::Stretch,
                    justify_self: JustifySelf::Stretch,
                    flex_wrap: FlexWrap::Wrap,
                    justify_content: JustifyContent::FlexStart,
                    align_items: AlignItems::FlexStart,
                    align_content: AlignContent::FlexStart,
                    ..default()
                },
                background_color: Color::WHITE.into(),
                border_color: Color::BLACK.into(),
                // background_color: Color::RED.into(),
                // border_color: Color::WHITE.into(),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                TextBundle::from_section(
                    "<info>",
                    TextStyle {
                        font_size: 22.0,
                        color: Color::BLACK,
                        ..default()
                    },
                ),
                InfobarText,
            ));
        })
        .id();
}

fn update_infobar(
    mut commands: Commands,
    mut alignment_events: EventReader<InfobarAlignmentEvent>,
    // infobar: Query<Entity, With<Infobar>>,
    mut infobar_text: Query<&mut Text, With<InfobarText>>,

    alignments: Res<crate::Alignments>,
    alignment_query: Query<&super::alignments::Alignment>,
) {
    // let last_visible = alignment_events.read().fold(None, |last, ev| {
    //     if let Some(last) = last {
    //         if ev.cursor_over == false && last == ev.alignment {
    //             *last = ev.alignment;
    //         }
    //     }

    //     if ev.cursor_over = true {
    //         *last = ev.alignment;
    //     }
    //     last
    // });

    let mut last_hovered = None;

    for ev in alignment_events.read() {
        if let Some(last) = last_hovered {
            if ev.cursor_over == false && last == ev.alignment_entity {
                last_hovered = None;
            }
        }

        if ev.cursor_over == true {
            last_hovered = Some(ev.alignment_entity);
        }
    }

    // let infobar = infobar.single();

    let Some(al_ent) = last_hovered else {
        return;
    };

    let Ok(alignment) = alignment_query.get(al_ent) else {
        return;
    };

    if let Ok(mut text) = infobar_text.get_single_mut() {
        //
        text.sections[0].value = format!(
            "Hovered ({:?}, {:?})[{}]",
            alignment.target, alignment.query, alignment.pair_index
        );
        println!("setting infobar text: {}", text.sections[0].value);
    }
}

use bevy_mod_picking::prelude::*;

impl From<ListenerInput<Pointer<Over>>> for InfobarAlignmentEvent {
    fn from(value: ListenerInput<Pointer<Over>>) -> Self {
        InfobarAlignmentEvent {
            alignment_entity: value.target(),
            cursor_over: true,
        }
    }
}

impl From<ListenerInput<Pointer<Out>>> for InfobarAlignmentEvent {
    fn from(value: ListenerInput<Pointer<Out>>) -> Self {
        InfobarAlignmentEvent {
            alignment_entity: value.target(),
            cursor_over: false,
        }
    }
}
