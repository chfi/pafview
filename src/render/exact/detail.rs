// we want a texture with all the possible nucleotides & pairs (for mismatches),
// and with different background colors

// pairs only need the one BG color, other nucleotides need all, one per cigar op

//  G  T  C  A
// GTCA   GG GT GC GA
// TCAG   TG TT TC TA
// CAGT   CG CT CC CA
// AGTC   AG AT AC AA
//

use rustc_hash::FxHashMap;
use ultraviolet::{DVec2, UVec2, Vec2};

use crate::{sequences::SeqId, CigarOp};

use crate::PixelBuffer;

const TILE_BUFFER_SIZE: usize = 32;
const TILE_BUFFER_SIZE_F: f32 = TILE_BUFFER_SIZE as f32;

pub type TileBuffers = FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>;

pub(crate) fn build_op_pixel_buffers() -> FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer> {
    // can be used as an atlas of 16x32 sprites, index into by subtracting ' ' from a
    // printable/"normal" ascii byte
    let font_bitmap =
        lodepng::decode32(include_bytes!("../../../assets/spleen_font/16x32.png")).unwrap();

    let png_font_pixels = PixelBuffer {
        width: font_bitmap.width as u32,
        height: font_bitmap.height as u32,
        pixels: font_bitmap
            .buffer
            .into_iter()
            .map(|rgba| {
                let [r, g, b, a]: [u8; 4] = rgba.into();
                let a = a.min(r).min(g).min(b);
                egui::Color32::from_rgba_premultiplied(r, g, b, a)
            })
            .collect::<Vec<_>>(),
    };

    let tile_size = TILE_BUFFER_SIZE as u32;

    let masked = |bg: egui::Color32, fg: egui::Color32| {
        move |_below: egui::Color32, above: egui::Color32| -> egui::Color32 {
            if above == egui::Color32::TRANSPARENT {
                bg
            } else {
                fg
            }
        }
    };

    let draw_char = |dst: &mut PixelBuffer,
                     dst_offset: [f32; 2],
                     dst_size: [f32; 2],
                     ch: char,
                     bg: egui::Color32,
                     fg: egui::Color32| {
        let ix = (ch as u8 - b' ') as u32;

        let y = if dst_offset[1] < 0.0 {
            dst_offset[1].abs().round() as u32
        } else {
            0
        };

        let src_offset = [ix * 16, y];
        let src_size = [16, 32];
        png_font_pixels.sample_subimage_nn_into_with(
            dst,
            dst_offset,
            dst_size,
            src_offset,
            src_size,
            masked(bg, fg),
        );
    };

    use CigarOp as Cg;
    let mut tiles = FxHashMap::default();

    // add individual target/query bps for I & D
    // add both bp pairs for M/=/X

    for (op, bg_color) in [
        (Cg::M, egui::Color32::BLACK),
        (Cg::Eq, egui::Color32::BLACK),
        (Cg::X, egui::Color32::RED),
        (Cg::I, egui::Color32::WHITE),
        (Cg::D, egui::Color32::WHITE),
    ] {
        let buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);
        tiles.insert((op, [None, None]), buffer);
    }

    let nucleotides = ['G', 'T', 'C', 'A', 'U', 'N'];

    for (op, bg_color) in [
        (Cg::I, egui::Color32::WHITE),
        (Cg::D, egui::Color32::WHITE), // testing
    ] {
        for &nucl in nucleotides.iter() {
            let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);

            let fg_color = egui::Color32::BLACK;

            let x = TILE_BUFFER_SIZE_F * 0.25;
            draw_char(
                &mut buffer,
                [x, 0.0],
                [16.0, 32.0],
                nucl,
                bg_color,
                fg_color,
            );

            if op == Cg::I {
                tiles.insert((Cg::I, [None, Some(nucl)]), buffer);
            } else {
                tiles.insert((Cg::D, [Some(nucl), None]), buffer);
            }
        }
    }

    // TODO Eq should show just one nucleotide
    for (op, bg_color) in [
        (Cg::M, egui::Color32::BLACK),
        (Cg::Eq, egui::Color32::BLACK),
        (Cg::X, egui::Color32::RED),
    ] {
        for &query in nucleotides.iter() {
            for &target in nucleotides.iter() {
                let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);

                let fg_color = egui::Color32::WHITE;
                // let fg_color = if op == Cg::X {
                //     egui::Color32::BLACK
                // } else {
                //     egui::Color32::WHITE
                // };

                let x0 = 0.0;
                let y0 = -4.0;
                let x1 = TILE_BUFFER_SIZE_F / 2.0;
                let y1 = TILE_BUFFER_SIZE_F / 8.0;

                if op == Cg::Eq {
                    draw_char(
                        &mut buffer,
                        [TILE_BUFFER_SIZE_F * 0.25, 0.0],
                        [16.0, 32.0],
                        query,
                        bg_color,
                        fg_color,
                    );
                } else {
                    draw_char(
                        &mut buffer,
                        [x0, y0],
                        [16.0, 32.0],
                        query,
                        bg_color,
                        fg_color,
                    );
                    draw_char(
                        &mut buffer,
                        [x1, y1],
                        [16.0, 32.0],
                        target,
                        bg_color,
                        fg_color,
                    );
                }

                tiles.insert((op, [Some(target), Some(query)]), buffer);
            }
        }
    }

    tiles
}

pub fn draw_alignments(
    tile_buffers: &FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>,
    sequences: &crate::sequences::Sequences,
    grid: &crate::AlignmentGrid,
    alignments: &crate::paf::Alignments,
    view: &crate::view::View,
    canvas_size: impl Into<UVec2>,
    // canvas: &mut PixelBuffer,
) -> Option<PixelBuffer> {
    let canvas_size = canvas_size.into();
    let screen_dims = [canvas_size.x as f32, canvas_size.y as f32];

    // using the AlignmentGrid, find the alignments that overlap the view
    // (in 99% of cases this will only be one, but that 1% still can matter)
    let x_tiles = grid
        .x_axis
        .tiles_covered_by_range(view.x_range())?
        .collect::<Vec<_>>();
    let y_tiles = grid
        .y_axis
        .tiles_covered_by_range(view.y_range())?
        .collect::<Vec<_>>();

    // for each overlapping alignment,
    //   find the local target & query ranges
    //   compute the corresponding pixel rectangle ("subcanvas") in the output canvas
    //   step through (using AlignmentIter) the local target range,
    //     for each op, find the corresponding pixel rectangle in the subcanvas,
    //       & the source tile using the op type & sequence nucleotides,
    //       then copy into final canvas using `sample_subimage_into`

    fn clamped_range(
        axis: &crate::grid::GridAxis,
        seq_id: SeqId,
        view_range: std::ops::RangeInclusive<f64>,
    ) -> Option<std::ops::Range<u64>> {
        // let clamped_range = |axis: &crate::grid::GridAxis,
        //                      seq_id: SeqId,
        //                      view_range: std::ops::RangeInclusive<f64>| {
        let range = axis.sequence_axis_range(seq_id)?;
        let start = range.start.max(*view_range.start() as u64);
        let end = range.end.min(*view_range.end() as u64).max(start);

        let start = start - range.start;
        let end = end - range.start;
        Some(start..end)
    }

    let map_to_point = |target_id: SeqId, query_id: SeqId, target: u64, query: u64| {
        let x = grid
            .x_axis
            .axis_local_to_global_exact(target_id, target)
            .unwrap();
        let y = grid
            .y_axis
            .axis_local_to_global_exact(query_id, query)
            .unwrap();

        DVec2::new(x as f64, y as f64)
    };

    let sequence_getter = |t_id: SeqId, q_id: SeqId| {
        let target_seq = sequences.get_bytes(t_id);
        let query_seq = sequences.get_bytes(q_id);
        move |op: CigarOp, target: usize, query: usize| {
            let t_seq = op.consumes_target().then_some(()).and(
                target_seq
                    .and_then(|seq| seq.get(target).copied())
                    .map(|c| c as char),
            );
            let q_seq = op.consumes_query().then_some(()).and(
                query_seq
                    .and_then(|seq| seq.get(query).copied())
                    .map(|c| c as char),
            );
            [t_seq, q_seq]
        }
    };

    // let color_buffers = [
    //     egui::Color32::RED,
    //     egui::Color32::GREEN,
    //     egui::Color32::BLUE,
    //     egui::Color32::GOLD,
    // ]
    // .into_iter()
    // .map(|color| PixelBuffer::new_color(32, 32, color))
    // .collect::<Vec<_>>();
    // println!("drawing alignments!");
    let mut dst_pixels = PixelBuffer::new_color(canvas_size.x, canvas_size.y, egui::Color32::WHITE);

    let mut tile_i = 0;

    // let tile = super::create_test_pattern_buffer(64, 64);

    for &target_id in &x_tiles {
        for &query_id in &y_tiles {
            let Some(alignment) = alignments.pairs.get(&(target_id, query_id)) else {
                continue;
            };

            let this_tile = tile_i;
            tile_i += 1;

            // clamped ranges + pixel ranges
            let clamped_target = clamped_range(&grid.x_axis, target_id, view.x_range()).unwrap();
            let clamped_query = clamped_range(&grid.y_axis, query_id, view.y_range()).unwrap();

            // world coordinates of the visible screen rectangle corresponding to this alignment
            let world_min = map_to_point(
                target_id,
                query_id,
                clamped_target.start,
                clamped_query.start,
            );
            let world_max =
                map_to_point(target_id, query_id, clamped_target.end, clamped_query.end);

            let s0 = view.map_world_to_screen(screen_dims, world_min);
            let s1 = view.map_world_to_screen(screen_dims, world_max);

            let screen_max = s0.max_by_component(s1);
            let screen_min = s0.min_by_component(s1);

            // println!(
            //     "drawing ({}, {}) to rect ({screen_min:?}, {screen_max:?})",
            //     target_id.0, query_id.0
            // );

            let screen_size = screen_max - screen_min;
            let px_per_bp = screen_size.x / (clamped_target.end - clamped_target.start) as f32;

            let dst_size = Vec2::new(px_per_bp, px_per_bp);

            let seqs = sequence_getter(target_id, query_id);

            // let (_, local_t_start) = grid
            //     .x_axis
            //     .global_to_axis_local(clamped_target.start as f64)
            //     .unwrap();
            // // let local_t_end = grid.x_axis.global_to_axis_local(clamped_target.end as f64);

            // let (_, local_q_start) = grid
            //     .y_axis
            //     .global_to_axis_local(clamped_query.start as f64)
            //     .unwrap();

            let x_global_start = grid.x_axis.sequence_offset(target_id).unwrap();
            let y_global_start = grid.y_axis.sequence_offset(query_id).unwrap();
            let seq_global_offset = DVec2::new(x_global_start as f64, y_global_start as f64);

            // println!("seq global offset: {seq_global_offset:?}");

            /*
            // testing/debug bits

            let dst_offset = view.map_world_to_screen(screen_dims, seq_global_offset);
            // view.map_world_to_screen(screen_dims, seq_global_offset + [300.0, 300.0].into());

            // let src_tile = &color_buffers[this_tile % color_buffers.len()];
            // src_tile.sample_subimage_into(
            //     // dst, dst_offset, dst_size, src_offset, src_size)
            //     &mut dst_pixels,
            //     dst_offset.into(),
            //     Vec2::broadcast(32.0).into(),
            //     // dst_offset.into(),
            //     // dst_size.into(),
            //     [0, 0],
            //     [32, 32],
            //     // src_offset,
            //     // src_size,
            // );

            for (src_tile, offset) in std::iter::zip(
                color_buffers.iter().cycle(),
                [
                    [0.0, 0.0],
                    [300.0, 0.0],
                    [0.0, 300.0],
                    [500.0, 500.],
                    [0.0, -400.0],
                ],
            ) {
                let dst_offset =
                    view.map_world_to_screen(screen_dims, seq_global_offset + offset.into());
                src_tile.sample_subimage_into(
                    // dst, dst_offset, dst_size, src_offset, src_size)
                    &mut dst_pixels,
                    dst_offset.into(),
                    Vec2::broadcast(32.0).into(),
                    // dst_offset.into(),
                    // dst_size.into(),
                    [0, 0],
                    [32, 32],
                    // src_offset,
                    // src_size,
                );
            }
            */

            // end testing/debug bits

            let mut count = 0;
            for item in alignment.iter_target_range(clamped_target.clone()) {
                count += 1;
                let op = item.op;
                // let count = item.op_count;

                let mut aabb_min = Vec2::broadcast(std::f32::INFINITY);
                let mut aabb_max = Vec2::broadcast(std::f32::NEG_INFINITY);

                for (_i, [tgt, qry]) in item.enumerate() {
                    let nucls = seqs(op, tgt, qry);

                    // TODO the clamped ranges must be in local sequence space;
                    // clamped_target and clamped_query are global u64s
                    // let tgt_offset = tgt - clamped_target.start as usize;
                    // let qry_offset = qry - clamped_query.start as usize;

                    let world_offset = seq_global_offset + [tgt as f64, qry as f64].into();

                    let dst_offset = view.map_world_to_screen(
                        screen_dims,
                        world_offset, // seq_global_offset + [tgt as f64, qry as f64].into(),
                    );

                    // let tile = tile_buffers
                    //     .get(&(CigarOp::X, [Some('G'), Some('T')]))
                    //     .unwrap();

                    let Some(tile) = tile_buffers.get(&(op, nucls)) else {
                        log::error!("Did not find tile for ({op:?}, {nucls:?}");
                        continue;
                    };

                    tile.sample_subimage_nn_into(
                        &mut dst_pixels,
                        dst_offset.into(),
                        dst_size.into(),
                        [0, 0],
                        [TILE_BUFFER_SIZE as u32, TILE_BUFFER_SIZE as u32],
                        // [10, 10],
                    );
                }
            }
        }
    }

    Some(dst_pixels)
}
