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
use ultraviolet::{DVec2, UVec2};

use crate::{sequences::SeqId, CigarIndex, CigarIter, CigarOp};

use super::PixelBuffer;

const TILE_BUFFER_SIZE: usize = 32;
const TILE_BUFFER_SIZE_F: f32 = TILE_BUFFER_SIZE as f32;

fn build_op_pixel_buffers() -> FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer> {
    let fonts = egui::text::Fonts::new(2.0, 512, egui::FontDefinitions::default());

    let tile_width = TILE_BUFFER_SIZE;
    let tile_height = tile_width;

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

    fonts.begin_frame(1.0, 512);
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

    let tile_size = 32;

    // add individual target/query bps for I & D

    // add both bp pairs for M/=/X

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

/*
fn build_detail_texture() -> Option<Vec<egui::Color32>> {
    let fonts = egui::text::Fonts::new(2.0, 512, egui::FontDefinitions::default());

    let tile_width = 16.0;
    let tile_height = tile_width;

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
    // let g_small = &gtca_galley.rows[0]

    // let row = &gtca_galley.rows[0];
    // let g = row.glyphs[0];
    // let t = row.glyphs[1];
    // let c = row.glyphs[2];
    // let a = row.glyphs[3];

    let width = 256;
    let height = 256;

    let mut pixels = vec![egui::Color32::TRANSPARENT; width * height];

    let mut row = 0;

    use egui::Color32 as Color;

    let bg_colors = vec![
        Color::BLACK,
        Color::RED,
        Color::BLACK,
        Color::TRANSPARENT,
        Color::TRANSPARENT,
    ];

    let fg_colors = vec![
        Color::WHITE,
        Color::BLACK,
        Color::WHITE,
        Color::BLACK,
        Color::BLACK,
    ];

    let nucls = vec!['G', 'T', 'C', 'A'];
    let nucl_pairs = nucls
        .iter()
        .flat_map(|a| nucls.iter().map(|b| (*a, *b)))
        .collect::<Vec<_>>();

    let small_glyph_pairs = gtca_small_glyphs
        .iter()
        .flat_map(|a| gtca_small_glyphs.iter().map(|b| (*a, *b)))
        .collect::<Vec<_>>();

    // for (col, (fst, snd)) in std::iter::zip(&gtca_small_glyphs

    // row 0 - (mis)match, white GTCA on black, both seqs
    // for (col, &(fst, snd)) in small_glyph_pairs.iter().enumerate() {
    //     let dst_x0 = col * tile_width;
    //     let dst_x1 = dst_x0 + tile_width;

    //     let dst_y0 = row * tile_height;
    //     let dst_y1 = dst_y0 + tile_height;

    //     //
    // }

    /*
    row += 1;
    // row 1 - mismatch, black GTCA on red, both seqs
    for (col, &(fst, snd)) in nucl_pairs.iter().enumerate() {
        //
    }

    row += 1;

    // row 2 - same as row 0 (but may change)
    for (col, &(fst, snd)) in nucl_pairs.iter().enumerate() {
        //
    }

    // row 3 - black GTCA on transparent, target only
    row += 1;
    for (col, &bp) in nucls.iter().enumerate() {
        //
    }

    // row 4 - black GTCA on transparent, query only
    row += 1;
    for (col, &bp) in nucls.iter().enumerate() {
        //
    }
    */

    // use crate::cigar::CigarOp as Cg;
    // let cigar_ops = [Cg::M, Cg::X, Cg::Eq, Cg::D, Cg::I];

    // for op in cigar_ops {
    //     let bg_color = cigar_color_def(op);

    // for nucl in ['G', 'T', 'C', 'A'] {
    //     //
    //     // TODO use a CPU font rasterizer for the letters
    // }
    // }

    Some(pixels)
}
*/

fn cigar_color_def(op: CigarOp) -> egui::Color32 {
    match op {
        CigarOp::M => egui::Color32::BLACK,
        CigarOp::X => egui::Color32::RED,
        CigarOp::Eq => egui::Color32::GREEN,
        CigarOp::D => egui::Color32::BLUE,
        CigarOp::I => egui::Color32::BLUE,
        _ => egui::Color32::TRANSPARENT,
    }
}

pub fn draw_alignments(
    alignments: &crate::paf::Alignments,
    sequences: &crate::sequences::Sequences,
    grid: &crate::AlignmentGrid,
    view: &crate::view::View,
    tile_buffers: &FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>,
    canvas_size: impl Into<UVec2>,
    // canvas: &mut PixelBuffer,
) -> PixelBuffer {
    let canvas_size = canvas_size.into();
    let screen_dims = [canvas_size.x as f32, canvas_size.y as f32];

    // using the AlignmentGrid, find the alignments that overlap the view
    // (in 99% of cases this will only be one, but that 1% still can matter)
    let x_tiles = grid
        .x_axis
        .tiles_covered_by_range(view.x_range())
        .unwrap()
        .collect::<Vec<_>>();
    let y_tiles = grid
        .y_axis
        .tiles_covered_by_range(view.y_range())
        .unwrap()
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
    };

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

    let mut dst_pixels = PixelBuffer::new(canvas_size.x, canvas_size.y);

    for &target_id in &x_tiles {
        for &query_id in &y_tiles {
            let Some(alignment) = alignments.pairs.get(&(target_id, query_id)) else {
                continue;
            };

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

            let screen_min = view.map_world_to_screen(screen_dims, world_min);
            let screen_max = view.map_world_to_screen(screen_dims, world_max);

            let seqs = sequence_getter(target_id, query_id);

            for item in alignment.iter_target_range(clamped_target) {
                let op = item.op;
                let count = item.op_count;

                for [tgt, qry] in item {
                    let nucls = seqs(op, tgt, qry);

                    let Some(tile) = tile_buffers.get(&(op, nucls)) else {
                        continue;
                    };

                    let dst_offset = screen_min;
                    let dst_size = screen_max - screen_min;
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
            }
        }
    }

    dst_pixels
}

fn draw_cigar_section(
    alignment: &crate::paf::Alignment,
    target_seq: Option<&[u8]>,
    query_seq: Option<&[u8]>,
    view: &crate::view::View,
    // canvas corresponding to the whole view
    canvas: &mut PixelBuffer,
) {
    // derive target & query range from view & alignment
    // let cg_iter = cigar.iter_target_range(target_range.clone());

    todo!();
}

/*
pub fn draw_subsection(
    target_seq: &[u8],
    query_seq: &[u8],
    match_data: &crate::ProcessedCigar,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas_size: UVec2,
    canvas_data: &mut Vec<egui::Color32>,
) {
    let size = (canvas_size.x * canvas_size.y) as usize;
    canvas_data.clear();
    canvas_data.resize(size, egui::Color32::WHITE);

    // TODO doesn't take strand into account yet
    let match_iter = MatchOpIter::from_range(
        &match_data.match_offsets,
        &match_data.match_cigar_index,
        &match_data.cigar,
        target_range.clone(),
    );

    let tgt_len = target_range.end - target_range.start;
    let bp_width = canvas_size.x as f64 / tgt_len as f64;

    let qry_len = query_range.end - query_range.start;
    let bp_height = canvas_size.y as f64 / qry_len as f64;

    for ([target_pos, query_pos], cg_ix) in match_iter {
        let cg_op = match_data.cigar[cg_ix].0;
        let is_match = cg_op.is_match();

        let color = if is_match {
            egui::Color32::BLACK
        } else {
            egui::Color32::RED
        };

        let Some(target_offset) = target_pos.checked_sub(target_range.start) else {
            continue;
        };
        let Some(query_offset) = query_pos.checked_sub(query_range.start) else {
            continue;
        };

        let x0 = target_offset as f64 * bp_width;
        let x1 = (1 + target_offset) as f64 * bp_width;

        let y0 = query_offset as f64 * bp_height;
        let y1 = (1 + query_offset) as f64 * bp_height;

        let x = 0.5 * (x0 + x1);
        let y = 0.5 * (y0 + y1);

        for x in (x0.floor() as usize)..(x1.floor() as usize) {
            for y in (y0.floor() as usize)..(y1.floor() as usize) {
                let y = (canvas_size.y as usize)
                    .checked_sub(y + 1)
                    .unwrap_or_default();
                let ix = x + y * canvas_size.x as usize;
                if x < canvas_size.x as usize && y < canvas_size.y as usize {
                    canvas_data.get_mut(ix).map(|px| *px = color);
                }
            }
        }
    }
}
*/
