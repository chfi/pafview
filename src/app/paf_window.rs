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

                            if label.hovered() {
                                // TODO highlight alignment
                            }

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
