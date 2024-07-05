use bevy::{
    input::mouse::MouseMotion,
    prelude::*,
    render::camera::{CameraProjection, ScalingMode},
};

use bevy_polyline::{
    material::PolylineMaterial,
    polyline::{Polyline, PolylineBundle},
    PolylinePlugin,
};
use clap::Parser;
use rustc_hash::FxHashMap;

use crate::{paf::AlignmentIndex, sequences::SeqId, PafViewerApp};

pub struct PafViewerPlugin;

impl Plugin for PafViewerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup, prepare_alignments).chain())
            .add_systems(
                Update,
                update_camera.before(bevy::render::camera::camera_system::<OrthographicProjection>),
            );
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

fn update_camera(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mouse_motion: EventReader<MouseMotion>,

    mut camera_query: Query<(&mut Transform, &Projection), With<Camera>>,
) {
    for (mut transform, proj) in camera_query.iter_mut() {
        let Projection::Orthographic(proj) = proj else {
            continue;
        };

        let xv = proj.area.width() * 0.01;
        let yv = proj.area.height() * 0.01;

        let mut dx = 0.0;

        if keyboard.pressed(KeyCode::ArrowLeft) {
            dx -= xv;
        }
        if keyboard.pressed(KeyCode::ArrowRight) {
            dx += xv;
        }

        let mut dy = 0.0;

        if keyboard.pressed(KeyCode::ArrowUp) {
            dy += yv;
        }
        if keyboard.pressed(KeyCode::ArrowDown) {
            dy -= yv;
        }

        transform.translation.x += dx;
        transform.translation.y += dy;
        // println!("{:?}", transform.translation);
    }

    //
}

fn setup(mut commands: Commands, viewer: Res<PafViewer>) {
    let grid = &viewer.app.alignment_grid;

    let x_len = grid.x_axis.total_len as f32;
    let y_len = grid.y_axis.total_len as f32;

    // prepare camera etc.

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        projection: OrthographicProjection {
            // near: todo!(),
            // far: todo!(),
            // viewport_origin: todo!(),
            // scaling_mode: ScalingMode::AutoMin { min_width: (), min_height: () }
            scale: 10_000.0,
            // area: todo!(),
            ..default()
        }
        .into(),
        // camera: Camera {
        //     // hdr: true,
        //     ..default()
        // },
        ..default()
    });

    // commands.spawn(Camera2dBundle
}

pub fn run(app: PafViewerApp) -> anyhow::Result<()> {
    // let args = crate::cli::Cli::parse();

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(PolylinePlugin)
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(PafViewer { app })
        .add_plugins(PafViewerPlugin)
        .run();
    //

    Ok(())
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

    let grid = &viewer.app.alignment_grid;

    for (_pair_id @ &(tgt_id, qry_id), alignments) in viewer.app.alignments.pairs.iter() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        let transform =
            Transform::from_translation(Vec3::new(x_offset as f32, y_offset as f32, 0.0));

        for (_ix, alignment) in alignments.iter().enumerate() {
            // bevy_polyline only supports single color polylines,
            // so to draw cigar ops with different colors, use one
            // polyline per op

            let location = &alignment.location;

            // let align_ix = AlignmentIndex {
            //     pair: *pair_id,
            //     index: ix,
            // };

            let mut m_verts = Vec::new();
            let mut eq_verts = Vec::new();
            let mut x_verts = Vec::new();

            let mut tgt_cg = 0;
            let mut qry_cg = 0;

            for (op, count) in alignment.cigar.whole_cigar() {
                let tgt_start = tgt_cg;
                let qry_start = qry_cg;

                let (tgt_end, qry_end) = match op {
                    Cg::Eq | Cg::X | Cg::M => {
                        tgt_cg += count as u64;
                        qry_cg += count as u64;
                        //
                        (tgt_start + count as u64, qry_start + count as u64)
                    }
                    Cg::I => {
                        qry_cg += count as u64;
                        //
                        (tgt_start, qry_start + count as u64)
                    }
                    Cg::D => {
                        tgt_cg += count as u64;
                        //
                        (tgt_start + count as u64, qry_start)
                    }
                };

                let tgt_range = location.map_from_aligned_target_range(tgt_start..tgt_end);
                let qry_range = location.map_from_aligned_query_range(qry_start..qry_end);

                let mut from =
                    ultraviolet::DVec2::new(tgt_range.start as f64, qry_range.start as f64);
                let mut to = ultraviolet::DVec2::new(tgt_range.end as f64, qry_range.end as f64);

                if location.query_strand.is_rev() {
                    std::mem::swap(&mut from.y, &mut to.y);
                }

                let p0 = Vec3::new(from.x as f32, from.y as f32, 0.0);
                let p1 = Vec3::new(to.x as f32, to.y as f32, 0.0);

                let buf = match op {
                    Cg::M => &mut m_verts,
                    Cg::Eq => &mut eq_verts,
                    Cg::X => &mut x_verts,
                    _ => continue,
                };

                buf.push(p0);
                buf.push(p1);
            }

            for (op, vertices) in [(Cg::M, m_verts), (Cg::Eq, eq_verts), (Cg::X, x_verts)] {
                let material = op_mats.get(&op).unwrap().clone();
                let polyline = polylines.add(Polyline { vertices });
                commands.spawn(PolylineBundle {
                    polyline,
                    material,
                    transform: transform.clone(),
                    ..default()
                });
            }
        }
    }
}
