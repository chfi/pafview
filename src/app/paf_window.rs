use bevy::prelude::*;
use bevy_egui::EguiContexts;

use crate::{Alignments, Sequences};

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
) {
    let ctx = contexts.ctx_mut();

    egui::Window::new("Go to region").show(ctx, |ui| {
        //

        let row_height = ui.text_style_height(&egui::TextStyle::Body);
        let total_rows = alignments.alignments.len();

        egui::ScrollArea::vertical().show_rows(ui, row_height, total_rows, |ui, range| {
            let rows = &alignments.alignments[range];

            for alignment in rows {
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

                let qry_text = format!("{qry_name}\t{qry_len}\t{qry_start}\t{qry_end}");
                let tgt_text = format!("{tgt_name}\t{tgt_len}\t{tgt_start}\t{tgt_end}");

                // TODO rest of PAF record

                ui.label(format!("{qry_text}\t{tgt_text}"));
            }
        });

        //
    });
}
