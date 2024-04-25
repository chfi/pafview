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

use crate::{sequences::SeqId, CigarIndex, CigarIter, CigarOp};

use super::PixelBuffer;

const TILE_BUFFER_SIZE: usize = 32;
const TILE_BUFFER_SIZE_F: f32 = TILE_BUFFER_SIZE as f32;

pub type TileBuffers = FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>;

pub(super) fn build_op_pixel_buffers() -> FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer> {
    let fonts = egui::text::Fonts::new(1.0, 1024, egui::FontDefinitions::default());

    let tile_size = TILE_BUFFER_SIZE as u32;

    let gtca_galley = fonts.layout(
        "GTCA".into(),
        egui::FontId::monospace(16.0),
        egui::Color32::BLACK,
        512.0,
    );
    let gtca_small_galley = fonts.layout(
        "GTCA".into(),
        egui::FontId::monospace(10.0),
        egui::Color32::BLACK,
        512.0,
    );

    let gtca_glyphs = gtca_galley.rows[0]
        .glyphs
        .iter()
        .take(4)
        .copied()
        .collect::<Vec<_>>();
    let gtca_small_glyphs = gtca_small_galley.rows[0]
        .glyphs
        .iter()
        .take(4)
        .copied()
        .collect::<Vec<_>>();

    fonts.begin_frame(1.0, 1024);
    let font_img = fonts.image();
    // let gtca_small_img

    let font_buffer = PixelBuffer {
        width: font_img.width() as u32,
        height: font_img.height() as u32,
        pixels: font_img.srgba_pixels(None).collect::<Vec<_>>(),
    };

    let get_nucl_i = |ix: usize| {
        let g = gtca_glyphs[ix].uv_rect;
        let src_offset = [g.min[0] as u32, g.min[1] as u32];
        let src_size = [
            g.max[0] as u32 - src_offset[0],
            g.max[1] as u32 - src_offset[1],
        ];
        (src_offset, src_size)
    };

    let get_small_nucl_i = |ix: usize| {
        let g = gtca_small_glyphs[ix].uv_rect;
        let src_offset = [g.min[0] as u32, g.min[1] as u32];
        let src_size = [
            g.max[0] as u32 - src_offset[0],
            g.max[1] as u32 - src_offset[1],
        ];
        (src_offset, src_size)
    };

    use CigarOp as Cg;
    let mut tiles = FxHashMap::default();

    // add individual target/query bps for I & D
    // add both bp pairs for M/=/X

    for (op, bg_color) in [
        (Cg::M, egui::Color32::BLACK),
        (Cg::Eq, egui::Color32::BLUE), // testing
        (Cg::X, egui::Color32::RED),
        (Cg::I, egui::Color32::GREEN),  // testing
        (Cg::D, egui::Color32::YELLOW), // testing
    ] {
        let buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);
        tiles.insert((op, [None, None]), buffer);
    }

    let nucleotides = ['G', 'T', 'C', 'A'];

    for (op, bg_color) in [
        (Cg::I, egui::Color32::GREEN),
        (Cg::D, egui::Color32::YELLOW), // testing
    ] {
        for (nucl_i, &nucl) in nucleotides.iter().enumerate() {
            let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);
            let (offset, size) = get_nucl_i(nucl_i);
            font_buffer.sample_subimage_into(&mut buffer, [0.0, 0.0], [1.0, 1.0], offset, size);

            if op == Cg::I {
                tiles.insert((Cg::I, [None, Some(nucl)]), buffer);
            } else {
                tiles.insert((Cg::D, [Some(nucl), None]), buffer);
            }
        }
    }
    for (op, bg_color) in [
        (Cg::M, egui::Color32::BLACK),
        (Cg::Eq, egui::Color32::BLUE), // testing
        (Cg::X, egui::Color32::RED),
    ] {
        for (qi, &query) in nucleotides.iter().enumerate() {
            for (ti, &target) in nucleotides.iter().enumerate() {
                let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);

                let (q_offset, q_size) = get_small_nucl_i(qi);
                let (t_offset, t_size) = get_small_nucl_i(ti);

                font_buffer.sample_subimage_into(
                    &mut buffer,
                    [0.0, 0.0],
                    [1.0, 1.0],
                    q_offset,
                    q_size,
                );

                font_buffer.sample_subimage_into(
                    &mut buffer,
                    [tile_size as f32 * 0.5, tile_size as f32 * 0.5],
                    [1.0, 1.0],
                    t_offset,
                    t_size,
                );

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

    let color_buffers = [
        egui::Color32::RED,
        egui::Color32::GREEN,
        egui::Color32::BLUE,
        egui::Color32::GOLD,
    ]
    .into_iter()
    .map(|color| PixelBuffer::new_color(32, 32, color))
    .collect::<Vec<_>>();
    // println!("drawing alignments!");
    let mut dst_pixels = PixelBuffer::new_color(canvas_size.x, canvas_size.y, egui::Color32::WHITE);

    let mut tile_i = 0;

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

            println!(
                "drawing ({}, {}) to rect ({screen_min:?}, {screen_max:?})",
                target_id.0, query_id.0
            );

            let screen_size = screen_max - screen_min;
            let px_per_bp = screen_size.x / (clamped_target.end - clamped_target.start) as f32;

            let dst_size = Vec2::new(px_per_bp, px_per_bp);

            let seqs = sequence_getter(target_id, query_id);

            let (_, local_t_start) = grid
                .x_axis
                .global_to_axis_local(clamped_target.start as f64)
                .unwrap();
            // let local_t_end = grid.x_axis.global_to_axis_local(clamped_target.end as f64);

            let (_, local_q_start) = grid
                .y_axis
                .global_to_axis_local(clamped_query.start as f64)
                .unwrap();
            // let local_q_end = grid.y_axis.global_to_axis_local(clamped_query.end as f64);

            // let local_t_start = local_t_start * alignment.location.target_total_len as f64;
            // let local_q_start = local_q_start * alignment.location.query_total_len as f64;

            // println!("px_per_bp: {px_per_bp}");

            let x_global_start = grid.x_axis.sequence_offset(target_id).unwrap();
            let y_global_start = grid.y_axis.sequence_offset(query_id).unwrap();
            let seq_global_offset = DVec2::new(x_global_start as f64, y_global_start as f64);

            println!("seq global offset: {seq_global_offset:?}");

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

            println!("clamped_target: {clamped_target:?}");
            let mut count = 0;
            for item in alignment.iter_target_range(clamped_target.clone()) {
                count += 1;
                let op = item.op;
                // let count = item.op_count;

                let mut aabb_min = Vec2::broadcast(std::f32::INFINITY);
                let mut aabb_max = Vec2::broadcast(std::f32::NEG_INFINITY);

                for (i, [tgt, qry]) in item.enumerate() {
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
                    // if i == 0 {
                    //     println!("{seq_global_offset:?} + [{tgt}, {qry}] =>\t{dst_offset:?}");
                    //     println!("{world_offset:?}");
                    // }

                    aabb_min = aabb_min.min_by_component(dst_offset);
                    aabb_max = aabb_max.max_by_component(dst_offset);

                    // let x_start = screen_min.x + px_per_bp * (tgt as f32 - local_t_start as f32);
                    // let y_start = screen_min.y + px_per_bp * (qry as f32 - local_q_start as f32);
                    // let x_start = screen_min.x;
                    // let y_start = screen_min.y;

                    // let y_start = qry - clamped_range.start;
                    // let dst_offset =
                    // let dst_offset = Vec2::new(x_start, y_start);

                    let Some(tile) = tile_buffers.get(&(op, nucls)) else {
                        log::error!("Did not find tile for ({op:?}, {nucls:?}");
                        continue;
                    };

                    // println!("sampling to {dst_offset:?} with size {dst_size:?}");

                    // let tile = tile_buffers
                    //     .get(&(CigarOp::X, [Some('A'), Some('G')]))
                    //     .unwrap();
                    // let dst_offset = Vec2::new(100.0, 100.0);
                    let dst_size = Vec2::new(50.0, 50.0);

                    // println!("tgt & qry offsets: {tgt_offset}, {qry_offset}");
                    // println!("  [{tgt}, {qry}] - dst_offset: {dst_offset:?}");
                    // let dst_size = Vec2::new(px_per_bp, px_per_bp);
                    // let dst_offset = screen_min;
                    // let dst_size = screen_max - screen_min;
                    // let src_offset = todo!();
                    // let src_size = todo!();
                    tile.sample_subimage_into(
                        &mut dst_pixels,
                        dst_offset.into(),
                        dst_size.into(),
                        [0, 0],
                        [TILE_BUFFER_SIZE as u32, TILE_BUFFER_SIZE as u32],
                        // src_offset,
                        // src_size,
                    );
                }

                // println!("AABB: {aabb_min:?}\t{aabb_max:?}");
            }
            if count > 0 {
                // println!("local_t_start: {local_t_start}\tlocal_q_start: {local_q_start}");
            }

            println!("drew {count} items");
        }
    }

    Some(dst_pixels)
}
