use crate::{
    paf::AlignmentIndex,
    render::{color::PafColorSchemes, exact::CpuViewRasterizerEgui},
    sequences::SeqId,
    PafViewerApp,
};
use bevy::prelude::*;

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        //
    }
}

#[derive(Component)]
pub struct Alignment {
    pub query: SeqId,
    pub target: SeqId,
}
