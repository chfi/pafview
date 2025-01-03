use bevy::prelude::*;

use crate::{
    app::{view::ViewEvent, SequencePairTile},
    sequences::SeqId,
    view::View,
    Alignments,
};

use super::{AlignmentAabbs, AlignmentIndex, AlignmentLayoutQuery};

pub struct GotoAlignmentPlugin;

impl Plugin for GotoAlignmentPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<GotoAlignmentEvent>().add_systems(
            PreUpdate,
            send_view_events.before(crate::app::view::handle_view_events),
        );
    }
}

#[derive(Event, Clone, Copy)]
pub enum GotoAlignmentEvent {
    SequencePair { target: SeqId, query: SeqId },
    Alignment { alignment_index: AlignmentIndex },
}

pub(crate) fn send_view_events(
    alignments: Res<Alignments>,
    layouts: AlignmentLayoutQuery,

    mut goto_events: EventReader<GotoAlignmentEvent>,
    mut view_events: EventWriter<ViewEvent>,
) {
    // TODO take layout into account, eventually
    let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
        goto_events.clear();
        return;
    };

    for goto_event in goto_events.read() {
        match *goto_event {
            GotoAlignmentEvent::SequencePair { target, query } => {
                let Some(aabb) = layout.aabbs.get(&SequencePairTile { target, query }) else {
                    continue;
                };

                let view = View {
                    x_min: aabb.mins.x,
                    y_min: aabb.mins.y,
                    x_max: aabb.mins.x,
                    y_max: aabb.mins.y,
                };

                view_events.send(ViewEvent { view });
            }
            GotoAlignmentEvent::Alignment { alignment_index } => {
                let AlignmentIndex {
                    query,
                    target,
                    pair_index,
                } = alignment_index;

                let tile_aabb = layout.aabbs.get(&SequencePairTile { target, query });

                let alignment = alignments.indices.get(&(target, query)).and_then(|ixs| {
                    let ix = ixs.get(pair_index)?;
                    alignments.alignments.get(*ix)
                });

                let Some((tile_aabb, alignment)) = tile_aabb.zip(alignment) else {
                    continue;
                };

                let loc = &alignment.location;

                let x_min = tile_aabb.mins.x + loc.target_range.start as f64;
                let y_min = tile_aabb.mins.y + loc.query_range.start as f64;
                let x_max = x_min + loc.aligned_target_len() as f64;
                let y_max = y_min + loc.aligned_query_len() as f64;

                let view = View {
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                };

                view_events.send(ViewEvent { view });
            }
        }
    }
}
