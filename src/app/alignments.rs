use crate::{sequences::SeqId, PafViewerApp};
use bevy::{prelude::*, utils::HashMap};

pub mod layout;

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentEntityIndex>()
            .init_resource::<SequencePairEntityIndex>()
            .add_plugins(layout::AlignmentLayoutPlugin);
        //
    }
}

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct AlignmentEntityIndex(pub HashMap<AlignmentIndex, Entity>);

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct SequencePairEntityIndex(pub HashMap<super::SequencePairTile, Entity>);

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct AlignmentIndex {
    pub query: SeqId,
    pub target: SeqId,

    pub pair_index: usize,
}
