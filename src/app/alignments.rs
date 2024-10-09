use crate::{sequences::SeqId, PafViewerApp};
use bevy::{prelude::*, utils::HashMap};

use super::AlignmentColorSchemes;

pub mod layout;

/*

Plugin for placing loaded alignments in the world and interacting with them

*/

pub struct AlignmentsPlugin;

impl Plugin for AlignmentsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AlignmentEntityIndex>()
            .init_resource::<SequencePairEntityIndex>()
            .add_plugins(layout::AlignmentLayoutPlugin)
            .add_systems(
                Startup,
                prepare_alignments.after(super::setup_screenspace_camera),
            );
        //
    }
}

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct AlignmentEntityIndex(pub HashMap<AlignmentIndex, Entity>);

#[derive(Debug, Default, Resource, Deref, DerefMut)]
pub struct SequencePairEntityIndex(pub HashMap<SequencePairTile, Entity>);

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct SequencePairTile {
    pub target: SeqId,
    pub query: SeqId,
}

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub struct AlignmentIndex {
    pub query: SeqId,
    pub target: SeqId,

    pub pair_index: usize,
}

fn update_seq_pair_transforms(
    default_layout: Res<layout::DefaultLayout>,

    mut seq_pair_tiles: Query<(&mut Transform, &SequencePairTile)>,
) {
    for (mut transform, tile_key) in seq_pair_tiles.iter_mut() {
        let Some(aabb) = default_layout.layout.aabbs.get(tile_key) else {
            continue;
        };

        let center = aabb.center();
        transform.translation = Vec3::new(center.x as f32, center.y as f32, 0.0);
    }
}

pub(super) fn prepare_alignments(
    mut commands: Commands,
    alignments: Res<crate::Alignments>,
    sequences: Res<crate::Sequences>,
    alignment_grid: Res<crate::AlignmentGrid>,
    color_schemes: Res<AlignmentColorSchemes>,
    cli_args: Res<crate::cli::Cli>,

    mut seq_pair_entity_index: ResMut<SequencePairEntityIndex>,
    mut alignment_entity_index: ResMut<AlignmentEntityIndex>,
    mut vertex_index: ResMut<super::render::AlignmentVerticesIndex>,

    mut meshes: ResMut<Assets<Mesh>>,
    mut border_rect_materials: ResMut<Assets<super::render::bordered_rect::BorderedRectMaterial>>,
    mut alignment_materials: ResMut<Assets<super::render::AlignmentPolylineMaterial>>,
    mut alignment_vertices: ResMut<Assets<super::render::AlignmentVertices>>,

    mut infobar_writer: EventWriter<super::infobar::InfobarAlignmentEvent>,
) {
    let grid = &alignment_grid;

    let low_mem = cli_args.low_mem;

    use bevy_mod_picking::prelude::*;

    let border_rect_mat =
        border_rect_materials.add(crate::app::render::bordered_rect::BorderedRectMaterial {
            fill_color: LinearRgba::new(0.0, 0.0, 0.0, 0.0),
            border_color: LinearRgba::new(0.4, 0.4, 0.4, 1.0),
            border_opacities: 0xFFFFFFFF,
            border_width_px: 1.0,
            alpha_mode: AlphaMode::Blend,
        });

    let mut digit_counts = [0usize; 10];

    for ((tgt_id, qry_id), alignments) in alignments.pairs() {
        let x_offset = grid.x_axis.sequence_offset(tgt_id).unwrap();
        let y_offset = grid.y_axis.sequence_offset(qry_id).unwrap();

        let seq_pair = SequencePairTile {
            target: tgt_id,
            query: qry_id,
        };

        let tgt_len = sequences
            .get(tgt_id)
            .map(|s| s.len() as f64)
            .unwrap_or_default();
        let qry_len = sequences
            .get(qry_id)
            .map(|s| s.len() as f64)
            .unwrap_or_default();

        let x_pos = x_offset as f64 + tgt_len * 0.5;
        let y_pos = y_offset as f64 + qry_len * 0.5;
        let transform = Transform::from_translation(Vec3::new(x_pos as f32, y_pos as f32, 0.0));

        let mesh = Rectangle::from_size([tgt_len as f32, qry_len as f32].into());

        let parent = commands
            .spawn((
                meshes.add(mesh),
                border_rect_mat.clone(),
                seq_pair,
                SpatialBundle {
                    transform,
                    ..default()
                },
                Pickable {
                    should_block_lower: false,
                    is_hoverable: true,
                },
                // On::<Pointer<Over>>::run(|input: Res<ListenerInput<Pointer<Over>>>| {
                //     println!("hovering seq pair: {:?}", input.listener());
                // }),
            ))
            .with_children(|parent| {
                for (ix, alignment) in alignments.enumerate() {
                    let len = alignment.location.target_total_len as usize;
                    let pow10 = len.ilog10() as usize;
                    digit_counts[pow10.min(digit_counts.len() - 1)] += 1;

                    let al_comp = AlignmentIndex {
                        target: alignment.target_id,
                        query: alignment.query_id,
                        pair_index: ix,
                    };
                    let color_scheme = color_schemes.colors.get(&al_comp);

                    // let vertices = if cli_args.low_mem {
                    //     render::AlignmentVertices::from_alignment_ignore_cigar(alignment)
                    // } else {
                    //     render::AlignmentVertices::from_alignment(alignment)
                    // };

                    let mut al_entity = parent.spawn((
                        al_comp,
                        Pickable {
                            should_block_lower: false,
                            is_hoverable: true,
                        },
                        On::<Pointer<Out>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                        On::<Pointer<Over>>::send_event::<super::infobar::InfobarAlignmentEvent>(),
                    ));

                    if !cli_args.low_mem {
                        let material = super::render::AlignmentPolylineMaterial::from_alignment(
                            grid,
                            alignment,
                            color_scheme.clone(),
                        );
                        let vertices = super::render::AlignmentVertices::from_alignment(alignment);

                        let vx_handle = alignment_vertices.add(vertices);

                        vertex_index.vertices.insert(al_comp, vx_handle.clone());

                        al_entity.insert((alignment_materials.add(material), vx_handle));
                    }
                    // .insert(
                    //     On::<Pointer<Over>>::run(|input: Res<ListenerInput<Pointer<Over>>>, alignments: Query<&alignments::Alignment>| {
                    //         println!("hovering alignment: {:?}", input.listener());
                    //     })
                    // )

                    let al_entity = al_entity.id();
                    alignment_entity_index.insert(al_comp, al_entity);
                }
            })
            .id();

        seq_pair_entity_index.insert(seq_pair, parent);
    }

    for (exp10, count) in digit_counts.iter().enumerate() {
        println!("{exp10:10} - {count}");
    }
}
