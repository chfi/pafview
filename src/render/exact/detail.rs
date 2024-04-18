// we want a texture with all the possible nucleotides & pairs (for mismatches),
// and with different background colors

// pairs only need the one BG color, other nucleotides need all, one per cigar op

//  G  T  C  A
// GTCA   GG GT GC GA
// TCAG   TG TT TC TA
// CAGT   CG CT CC CA
// AGTC   AG AT AC AA
//

use ultraviolet::UVec2;

use crate::{CigarIndex, CigarIter, CigarOp};

use super::PixelBuffer;

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

pub fn draw_cigar_section(
    target_seq: &[u8],
    query_seq: &[u8],
    cigar: &CigarIndex,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas: &mut PixelBuffer,
    // subcanvas_offset: UVec2
) {
    let cg_iter = cigar.iter_target_range(target_range.clone());

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
