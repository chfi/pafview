use bevy::prelude::*;
use bevy_mod_picking::prelude::ListenerInput;

pub struct InfobarPlugin;

impl Plugin for InfobarPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<InfobarAlignmentEvent>()
            .add_systems(Startup, setup_infobar)
            .add_systems(
                Update,
                update_infobar.run_if(resource_exists::<crate::paf::PafMetadata>),
            );
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
                    "",
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
    paf_opt_fields: Res<crate::paf::PafMetadata>,
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
        if let Some(metadata) = paf_opt_fields.get(alignment) {
            // if let Some(opt_fields) = paf_opt_fields.get_optional_fields(alignment) {
            let opt_fields = &metadata.optional_fields;
            let mut floats = opt_fields
                .iter()
                .filter_map(|(tag, (ty, val))| {
                    if *ty != 'f' {
                        return None;
                    }

                    Some((tag, val.parse::<f32>().ok()?))
                })
                .collect::<Vec<_>>();

            let mut ints = opt_fields
                .iter()
                .filter_map(|(tag, (ty, val))| {
                    if *ty != 'i' {
                        return None;
                    }

                    Some((tag, val.parse::<i32>().ok()?))
                })
                .collect::<Vec<_>>();

            floats.sort_by_key(|(tag, _)| *tag);
            ints.sort_by_key(|(tag, _)| *tag);

            let new_text = floats
                .iter()
                .map(|(tag, val)| {
                    let tag_txt = std::str::from_utf8(tag.as_slice()).unwrap();
                    format!("{tag_txt}:f:{val}")
                })
                .chain(ints.iter().map(|(tag, val)| {
                    let tag_txt = std::str::from_utf8(tag.as_slice()).unwrap();
                    format!("{tag_txt}:i:{val}")
                }));

            let (pfx, val) = {
                let len = metadata.alignment_block_length;
                if len > 1_000_000 {
                    ("M", len / 1_000_000)
                } else if len > 1_000 {
                    ("k", len / 1_000)
                } else {
                    ("", len)
                }
            };

            text.sections[0].value = format!("Length: {val}{pfx}B ");

            for (_ix, field) in new_text.into_iter().enumerate() {
                text.sections[0].value.push_str(" \t");
                text.sections[0].value.extend(field.chars());
            }
        }
        //
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
