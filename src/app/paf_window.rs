use bevy::prelude::*;
use bevy_egui::EguiContexts;
use egui::Sense;

use crate::{Alignment, Alignments, Sequences};

use super::{
    alignments::{
        goto::GotoAlignmentEvent, AlignmentLayoutQuery, DefaultLayoutRoot,
        SequencePairAlignmentEntities,
    },
    render::bordered_rect::BorderedRectMaterial,
    AlignmentIndex, SequencePairTile,
};

pub struct PafListWindowPlugin;

impl Plugin for PafListWindowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PafListWindowOpen>().add_systems(
            PreUpdate,
            show_paf_list_window.after(bevy_egui::EguiSet::BeginPass),
        );

        app.add_event::<HighlightAlignmentEvent>()
            .add_systems(Startup, prepare_highlight_assets)
            .add_systems(PreUpdate, highlight_alignments_from_events);
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct PafListWindowOpen(pub bool);

fn show_paf_list_window(
    mut contexts: EguiContexts,
    sequences: Res<Sequences>,
    alignments: Res<Alignments>,

    mut window_open: ResMut<PafListWindowOpen>,

    mut goto_view_events: EventWriter<GotoAlignmentEvent>,
    mut highlight_events: EventWriter<HighlightAlignmentEvent>,
) {
    let ctx = contexts.ctx_mut();

    egui::Window::new("PAF")
        .open(window_open.as_mut())
        .show(ctx, |ui| {
            let row_height = ui.text_style_height(&egui::TextStyle::Body);
            let total_rows = alignments.alignments.len();

            egui::ScrollArea::vertical().show_rows(ui, row_height, total_rows + 1, |ui, range| {
                egui::Grid::new("paf_list_window_record_grid")
                    .start_row(range.start)
                    .striped(true)
                    .show(ui, |ui| {
                        // header is always first row
                        ui.label("QRY");
                        ui.label("Seq. length");
                        ui.label("Start");
                        ui.label("End");
                        ui.label("TGT");
                        ui.label("Seq. length");
                        ui.label("Start");
                        ui.label("End");
                        ui.end_row();

                        let mut full_range = range.clone();
                        let _header_row = full_range.next();
                        let alignment_range = (full_range.start - 1)..(full_range.end - 1);
                        let rows = &alignments.alignments[alignment_range.clone()];

                        for (alignment_offset, alignment) in std::iter::zip(alignment_range, rows) {
                            let qry_seq = sequences.get(alignment.query_id);
                            let tgt_seq = sequences.get(alignment.target_id);
                            let Some((qry_seq, tgt_seq)) = qry_seq.zip(tgt_seq) else {
                                continue;
                            };

                            let qry_name = qry_seq.name();
                            let tgt_name = tgt_seq.name();
                            let loc = &alignment.location;

                            let qry_len = loc.query_total_len;
                            let qry_start = loc.query_range.start;
                            let qry_end = loc.query_range.end;

                            let tgt_len = loc.target_total_len;
                            let tgt_start = loc.target_range.start;
                            let tgt_end = loc.target_range.end;

                            let mut row_rect = egui::Rect::NOTHING;

                            row_rect = row_rect.union(ui.label(qry_name).rect);
                            row_rect = row_rect.union(ui.label(qry_len.to_string()).rect);
                            row_rect = row_rect.union(ui.label(qry_start.to_string()).rect);
                            row_rect = row_rect.union(ui.label(qry_end.to_string()).rect);
                            row_rect = row_rect.union(ui.label(tgt_name).rect);
                            row_rect = row_rect.union(ui.label(tgt_len.to_string()).rect);
                            row_rect = row_rect.union(ui.label(tgt_start.to_string()).rect);
                            row_rect = row_rect.union(ui.label(tgt_end.to_string()).rect);

                            let label = ui.interact(
                                row_rect,
                                ui.id().with(alignment_offset),
                                Sense::click(),
                            );

                            ui.end_row();
                            // TODO rest of PAF record

                            // this kinda sucks since it sends an event for every rendered row,
                            // rather than just on hover/exit, but w/e
                            highlight_events.send(HighlightAlignmentEvent {
                                alignment_offset,
                                is_highlighted: label.hovered(),
                            });

                            if label.clicked() {
                                goto_view_events
                                    .send(GotoAlignmentEvent::Alignment { alignment_offset });
                            };
                        }
                    });
            });

            //
        });
}

#[derive(Resource)]
struct AlignmentHighlightAssets {
    material: Handle<BorderedRectMaterial>,
    mesh: Handle<Mesh>,
}

fn prepare_highlight_assets(
    mut commands: Commands,
    mut materials: ResMut<Assets<BorderedRectMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let material = materials.add(BorderedRectMaterial {
        fill_color: LinearRgba::new(0.3, 0.05, 0.05, 0.3),
        border_color: LinearRgba::new(0.8, 0.2, 0.2, 0.8),
        border_width_px: 2.0,
        // border_opacities: todo!(),
        // border_width_modifiers: todo!(),
        // alpha_mode: todo!(),
        ..default()
    });

    let mesh = meshes.add(Rectangle::from_length(1.0));

    commands.insert_resource(AlignmentHighlightAssets { material, mesh });
}

#[derive(Event)]
struct HighlightAlignmentEvent {
    alignment_offset: usize,
    // alignment_entity: Entity,
    is_highlighted: bool,
}

#[derive(Component)]
struct AlignmentHighlightEntity(Entity);

#[derive(Component)]
struct AlignmentHighlight;

fn highlight_alignments_from_events(
    mut commands: Commands,
    mut events: EventReader<HighlightAlignmentEvent>,

    highlight_assets: Res<AlignmentHighlightAssets>,

    alignments: Res<Alignments>,

    layouts: AlignmentLayoutQuery,
    default_layout_root: Res<DefaultLayoutRoot>,

    seq_pair_alignments_map: Query<&SequencePairAlignmentEntities>,

    // alignment_aabbs: Res<super::alignments::AlignmentAabbs>,
    alignment_entities: Query<(Entity, Option<&AlignmentHighlightEntity>), With<AlignmentIndex>>,
) {
    let layout_root: Entity = default_layout_root.0;
    let Ok((_, layout_transform, layout_handle, layout_tiles)) =
        layouts.layout_roots.get(layout_root)
    else {
        events.clear();
        return;
    };

    let Some(layout) = layouts.layout_assets.get(layout_handle) else {
        events.clear();
        return;
    };

    for event in events.read() {
        let alignment = &alignments.alignments[event.alignment_offset];
        let pair_index = alignments.pair_indices[event.alignment_offset];

        let target = alignment.target_id;
        let query = alignment.query_id;

        let seq_pair = SequencePairTile { target, query };

        let Some(tile_entity) = layout_tiles.get(&seq_pair) else {
            continue;
        };

        let Ok(tile_alignment_entities) = seq_pair_alignments_map.get(*tile_entity) else {
            continue;
        };

        let Some(alignment_entity) = tile_alignment_entities.get(pair_index) else {
            continue;
        };

        let Ok((_, highlight_entity)) = alignment_entities.get(*alignment_entity) else {
            continue;
        };

        let Some(tile_aabb) = layout.aabbs.get(&seq_pair) else {
            continue;
        };

        let loc = &alignment.location;
        let loc_mid_x = (loc.target_range.start as f64 + loc.target_range.end as f64) * 0.5;
        let loc_mid_y = (loc.query_range.start as f64 + loc.query_range.end as f64) * 0.5;

        let mid_x = tile_aabb.mins.x + loc_mid_x;
        let mid_y = tile_aabb.mins.y + loc_mid_y;
        // let mid_x = tile_aabb.mins.x +

        let tgt_len = loc.aligned_target_len() as f32;
        let qry_len = loc.aligned_query_len() as f32;
        // let alignment_aabb = alignment_aabbs.get()

        if let Some(entity) = highlight_entity {
            if !event.is_highlighted {
                commands.entity(entity.0).despawn();
                commands
                    .entity(*alignment_entity)
                    .remove::<AlignmentHighlightEntity>();
            }
        } else {
            if event.is_highlighted {
                let transform = Transform::from_xyz(mid_x as f32, mid_y as f32, -20.0)
                    .with_scale(Vec3::new(tgt_len, qry_len, 1.0));

                let highlight_entity = commands
                    .spawn((
                        AlignmentHighlight,
                        SpatialBundle {
                            transform,
                            ..default() // visibility: todo!(),
                                        // inherited_visibility: todo!(),
                                        // view_visibility: todo!(),
                                        // transform: todo!(),
                                        // global_transform: todo!(),
                        },
                        highlight_assets.material.clone(),
                        highlight_assets.mesh.clone(),
                    ))
                    .id();
                // commands.entity(entity).despawn();
                commands
                    .entity(*alignment_entity)
                    .insert(AlignmentHighlightEntity(highlight_entity));
            }

            //
        }
    }
}
