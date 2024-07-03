use bevy::prelude::*;

use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
};
use clap::Parser;
use rustc_hash::FxHashMap;

use crate::{sequences::SeqId, PafViewerApp};

pub struct PafViewerPlugin;

impl Plugin for PafViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, prepare_alignments);
        todo!();
    }
}

#[derive(Resource)]
pub struct PafViewer {
    pub app: PafViewerApp,
}

#[derive(Component, Debug, Clone, Copy)]
struct SequencePairTile {
    target: SeqId,
    query: SeqId,
}

fn setup(mut commands: Commands) {
    // prepare camera etc.
}

fn prepare_alignments(
    mut commands: Commands,
    viewer: Res<PafViewer>,
    mut polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut polylines: ResMut<Assets<Polyline>>,
) {
    // create entities for the grid (tiles) & polylines for alignments

    // TODO pull in per-alignment color schemes
    // let op_colors = {
    //     use crate::cigar::CigarOp as Cg;
    //     [(Cg::M, Color::BLACK),
    //      (Cg::Eq, Color::BLACK),
    //      (Cg::X, Color::RED),
    //     ].into_iter().collect::<FxHashMap<_,_>>()
    //     };

    let mut op_mats = FxHashMap::default();
    use crate::cigar::CigarOp as Cg;
    for (op, color) in [
        (Cg::M, Color::BLACK),
        (Cg::Eq, Color::BLACK),
        (Cg::X, Color::RED),
    ] {
        let mat = polyline_materials.add(PolylineMaterial {
            width: 4.0,
            color,
            depth_bias: 0.0,
            perspective: true,
        });
        op_mats.insert(op, mat);
    }

    for (pair_id, alignments) in viewer.app.alignments.pairs.iter() {
        for (ix, alignment) in alignments.iter().enumerate() {
            // bevy_polyline only supports single color polylines,
            // so to draw cigar ops with different colors, use one
            // polyline per op

            let transform = todo!();

            for (&op, mat) in op_mats.iter() {
                //
                let mut vertices = Vec::new();

                commands.spawn(PolylineBundle {
                    polyline: polylines.add(Polyline { vertices }),
                    // material: polyline_materials.add(PolylineMaterial {
                    // })
                    transform,
                    ..default() // global_transform: todo!(),
                                // visibility: todo!(),
                                // inherited_visibility: todo!(),
                                // view_visibility: todo!(),
                });
            }
        }
        //
    }
}

pub fn run(app: PafViewerApp) -> anyhow::Result<()> {
    let args = crate::cli::Cli::parse();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
    }

    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(PafViewer { app })
        .add_plugins(PafViewerPlugin)
        .run();
    //

    Ok(())
}
