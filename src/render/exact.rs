use std::sync::Arc;

use rustc_hash::FxHashMap;
use ultraviolet::{DVec2, UVec2, Vec2};

use crate::app::alignments::layout::SeqPairLayout;
use crate::app::alignments::AlignmentIndex;
use crate::render::color::AlignmentColorScheme;
use crate::{sequences::SeqId, CigarOp};

use crate::PixelBuffer;

use super::color::PafColorSchemes;

pub(crate) struct CpuViewRasterizerEgui {
    pub tile_cache: TileBufferCache,
}

impl std::default::Default for CpuViewRasterizerEgui {
    fn default() -> Self {
        Self::initialize()
    }
}

impl CpuViewRasterizerEgui {
    pub fn initialize() -> Self {
        let tile_cache = TileBufferCache::init();
        Self { tile_cache }
    }
}

// we want a texture with all the possible nucleotides & pairs (for mismatches),
// and with different background colors

// pairs only need the one BG color, other nucleotides need all, one per cigar op

//  G  T  C  A
// GTCA   GG GT GC GA
// TCAG   TG TT TC TA
// CAGT   CG CT CC CA
// AGTC   AG AT AC AA
//

const TILE_BUFFER_SIZE: usize = 32;
const TILE_BUFFER_SIZE_F: f32 = TILE_BUFFER_SIZE as f32;

pub type TileBuffers = FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>;

pub(crate) struct TileBufferCache {
    cache: FxHashMap<AlignmentColorScheme, Arc<TileBuffers>>,
    // font_bitmap: lodepng::Bitmap<lodepng::RGBA>,
    png_font_pixels: PixelBuffer,
}

#[allow(dead_code)]
impl TileBufferCache {
    pub fn init() -> Self {
        let font_bitmap =
            lodepng::decode32(include_bytes!("../../assets/spleen_font/16x32.png")).unwrap();

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

        Self {
            cache: FxHashMap::default(),
            // font_bitmap,
            png_font_pixels,
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn get_cached_tile_buffers(
        &self,
        color_scheme: &AlignmentColorScheme,
    ) -> Option<&Arc<TileBuffers>> {
        self.cache.get(color_scheme)
    }

    pub fn cache_tile_buffers_for<'a>(
        &mut self,
        color_scheme: &'a AlignmentColorScheme,
    ) -> &TileBuffers {
        if !self.cache.contains_key(color_scheme) {
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
                self.png_font_pixels.sample_subimage_nn_into_with(
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

            let ops = [Cg::M, Cg::Eq, Cg::X, Cg::I, Cg::D];

            for &op in &ops {
                let bg_color = color_scheme.get_bg(op);
                let buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);
                tiles.insert((op, [None, None]), buffer);
            }

            let nucleotides = ['G', 'T', 'C', 'A', 'U', 'N'];

            for op in [Cg::I, Cg::D] {
                //
                let bg_color = color_scheme.get_bg(op);
                for &nucl in nucleotides.iter() {
                    let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);

                    let fg_color = color_scheme.get_fg(op);

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

            for op in [Cg::M, Cg::Eq, Cg::X] {
                let bg_color = color_scheme.get_bg(op);
                let fg_color = color_scheme.get_fg(op);
                for &query in nucleotides.iter() {
                    for &target in nucleotides.iter() {
                        let mut buffer = PixelBuffer::new_color(tile_size, tile_size, bg_color);

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

            let key = color_scheme.clone();
            self.cache.insert(key, tiles.into());
        }

        let Some(tiles) = self.cache.get(color_scheme) else {
            unreachable!();
        };

        tiles
    }
}

fn draw_alignments_at_offset<'a>(
    tile_cache: &TileBufferCache,
    alignment_colors: &PafColorSchemes,
    sequences: &crate::sequences::Sequences,
    view: &crate::view::View,
    canvas_size: impl Into<[u32; 2]>,
    seq_pair_offset: impl Into<[f64; 2]>,
    pixel_buffer: &mut PixelBuffer,
    alignments: impl IntoIterator<Item = (AlignmentIndex, &'a crate::Alignment)>,
) {
    let canvas_size = canvas_size.into();
    let canvas_size: bevy::math::UVec2 = canvas_size.into();
    let seq_pair_offset = seq_pair_offset.into();
    let seq_pair_offset: bevy::math::DVec2 = seq_pair_offset.into();

    // this may not be correct
    fn clamped_range(
        offset: u64,
        seq_range: &std::ops::Range<u64>,
        view_range: std::ops::RangeInclusive<f64>,
    ) -> std::ops::Range<u64> {
        let v_start = (*view_range.start() as u64)
            .checked_sub(offset)
            .unwrap_or(0);
        let v_end = (*view_range.end() as u64).checked_sub(offset).unwrap_or(0);

        let start = seq_range.start.max(v_start);
        let end = seq_range.end.min(v_end);
        // might fix a rare crash...
        let s = start.min(end);
        let e = start.max(end);
        s..e
    }

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

    let px_per_bp = canvas_size.x as f64 / view.width();

    for (align_ix, alignment) in alignments {
        // map clamp alignment bounds to `view` given the `seq_pair_offset`
        let loc = &alignment.location;
        let al_x0 = loc.target_range.start as f64 + seq_pair_offset.x;
        let al_x1 = al_x0 + loc.aligned_target_len() as f64;
        let al_y0 = loc.query_range.start as f64 + seq_pair_offset.y;
        let al_y1 = al_y0 + loc.aligned_query_len() as f64;

        // skip if alignment doesn't cover `view`
        let cl_x0 = al_x0.clamp(view.x_min, view.x_max);
        let cl_x1 = al_x1.clamp(view.x_min, view.x_max);

        let cl_y0 = al_y0.clamp(view.y_min, view.y_max);
        let cl_y1 = al_y1.clamp(view.y_min, view.y_max);

        if cl_x0 == cl_x1 || cl_y0 == cl_y1 {
            continue;
        }

        let color_scheme = alignment_colors.get(&align_ix);
        let Some(tile_buffers) = tile_cache.cache.get(&color_scheme) else {
            log::error!("Did not find tile buffer for alignment");
            continue;
        };

        let clamped_target = clamped_range(
            seq_pair_offset.x.round() as u64,
            &loc.target_range,
            view.x_range(),
        );

        let dst_size = Vec2::new(px_per_bp as f32, px_per_bp as f32);

        let seqs = sequence_getter(alignment.target_id, alignment.query_id);

        for item in alignment.iter_target_range(clamped_target) {
            let op = item.op;

            for [tgt, qry] in item {
                let nucls = seqs(op, tgt, qry);

                let world_offset = seq_pair_offset + bevy::math::DVec2::new(tgt as f64, qry as f64);

                if world_offset.x < view.x_min
                    || world_offset.x > view.x_max
                    || world_offset.y < view.y_min
                    || world_offset.y > view.y_max
                {
                    continue;
                }

                let world_offset: [f64; 2] = world_offset.into();
                let dst_offset = view.map_world_to_screen(canvas_size.as_vec2(), world_offset);

                let Some(tile) = tile_buffers.get(&(op, nucls)) else {
                    log::error!("Did not find tile for ({op:?}, {nucls:?}");
                    continue;
                };

                tile.sample_subimage_nn_into(
                    pixel_buffer,
                    dst_offset.into(),
                    dst_size.into(),
                    [0, 0],
                    [TILE_BUFFER_SIZE as u32, TILE_BUFFER_SIZE as u32],
                );
            }
        }
    }
}

pub(crate) fn draw_seq_pair_layouts_with_color_schemes<'a>(
    // tile_buffers: &FxHashMap<(CigarOp, [Option<char>; 2]), PixelBuffer>,
    tile_cache: &TileBufferCache,
    alignment_colors: &PafColorSchemes,
    sequences: &crate::sequences::Sequences,
    // grid: &crate::AlignmentGrid,
    alignments: &crate::paf::Alignments,
    view: &crate::view::View,
    canvas_size: impl Into<UVec2>,
    layouts: impl IntoIterator<Item = &'a SeqPairLayout>,
) -> PixelBuffer {
    let canvas_size = canvas_size.into();
    let screen_dims = [canvas_size.x as f32, canvas_size.y as f32];

    let mut dst_pixels =
        PixelBuffer::new_color(canvas_size.x, canvas_size.y, egui::Color32::TRANSPARENT);

    for layout in layouts {
        for (tile, aabb) in layout.aabbs.iter() {
            let offset: [f64; 2] = aabb.mins.into();

            let Some(tile_alignments) = alignments.pair_alignments((tile.target, tile.query))
            else {
                continue;
            };

            let alignments_iter = tile_alignments.enumerate().map(|(ix, al)| {
                let align_ix = AlignmentIndex {
                    query: al.query_id,
                    target: al.target_id,
                    pair_index: ix,
                };

                (align_ix, al)
            });

            draw_alignments_at_offset(
                tile_cache,
                alignment_colors,
                sequences,
                view,
                canvas_size,
                offset,
                &mut dst_pixels,
                alignments_iter,
            );
        }
    }

    dst_pixels
}
