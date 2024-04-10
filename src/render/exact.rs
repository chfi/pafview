use egui::{load::SizedTexture, ColorImage, ImageData, TextureOptions};
use rustc_hash::FxHashMap;
use ultraviolet::UVec2;

use crate::{
    cigar::{CigarOp, ProcessedCigar},
    grid::GridAxis,
};

#[derive(Default)]
pub struct CpuViewRasterizerEgui {
    last_texture: Option<(SizedTexture, egui::Rect)>,
    last_view: Option<crate::view::View>,

    last_wgpu_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl CpuViewRasterizerEgui {
    pub fn draw_into_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ctx: &egui::Context,
        app: &crate::PafViewerApp,
        view: &crate::view::View,
    ) {
        let size = ctx.screen_rect().size();

        let need_realloc = if let Some((texture, _)) = self.last_wgpu_texture.as_ref() {
            let extent = texture.size();
            size.x as u32 != extent.width || size.y as u32 != extent.height
        } else {
            true
        };

        if need_realloc {
            let size = wgpu::Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("BpRasterizer Texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("BpRasterizer Texture View")),
                ..Default::default()
            });

            self.last_wgpu_texture = Some((texture, texture_view));
        }

        let Some((texture, _texture_view)) = &self.last_wgpu_texture else {
            unreachable!();
        };

        if let Some((pixels, px_size, _rect)) =
            draw_exact_to_cpu_buffer(app, [size.x as u32, size.y as u32], view)
        {
            let size = wgpu::Extent3d {
                width: px_size.x as u32,
                height: px_size.y as u32,
                depth_or_array_layers: 1,
            };
            //
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level: 1,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&pixels),
                wgpu::ImageDataLayout::default(),
                size,
            )
        }
    }

    pub fn draw_and_display_view_layer(
        &mut self,
        ctx: &egui::Context,
        app: &crate::PafViewerApp,
        view: &crate::view::View,
    ) {
        if self.last_view.as_ref() != Some(view) {
            let size = ctx.screen_rect().size();

            if let Some((pixels, px_size, rect)) =
                draw_exact_to_cpu_buffer(app, [size.x as u32, size.y as u32], view)
            {
                //
                let tex_mgr = ctx.tex_manager();
                let mut tex_mgr = tex_mgr.write();
                let tex_id = tex_mgr.alloc(
                    "ExactRenderTexture".into(),
                    ImageData::Color(
                        ColorImage {
                            size: [px_size.x as usize, px_size.y as usize],
                            pixels,
                        }
                        .into(),
                    ),
                    TextureOptions::LINEAR,
                );
                self.last_texture = Some((
                    SizedTexture::new(tex_id, [px_size.x as f32, px_size.y as f32]),
                    rect,
                ));
                self.last_view = Some(view.clone());
            } else {
                self.last_texture = None;
                self.last_view = None;
            }
        }

        if let Some((texture, _rect)) = &self.last_texture {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Background,
                "cpu-rasterizer-layer".into(),
            ));

            let uv = egui::Rect::from_min_max([0.0, 0.0].into(), [1.0, 1.0].into());
            // painter.image(texture.id, *rect, uv, egui::Color32::WHITE);
            let tint = egui::Color32::WHITE;
            // let tint = tint.linear_multiply(0.5);
            let rect = ctx.screen_rect();
            painter.image(texture.id, rect, uv, tint);
        }
    }
}

pub fn draw_exact_to_cpu_buffer(
    app: &crate::PafViewerApp,
    canvas_size: impl Into<UVec2>,
    view: &crate::view::View,
) -> Option<(Vec<egui::Color32>, UVec2, egui::Rect)> {
    let canvas_size = canvas_size.into();

    let bp_per_pixel = view.width() / canvas_size.x as f64;

    // log::info!("bp_per_pixel: {bp_per_pixel}");

    if bp_per_pixel > super::PafRenderer::SCALE_LIMIT_BP_PER_PX {
        return None;
    }

    // find the "tiles" that are covered by the view; given the view
    // scale threshold this will probably never be more than 4 but no
    // reason not to do it properly

    let x_axis = &app.alignment_grid.x_axis;
    let y_axis = &app.alignment_grid.y_axis;

    let clamped_range = |axis: &GridAxis,
                         seq_id: usize,
                         // range: std::ops::Range<u64>,
                         view_range: std::ops::RangeInclusive<f64>| {
        let range = axis.sequence_axis_range(seq_id)?;
        let start = range.start.max(*view_range.start() as u64);
        let end = range.end.min(*view_range.end() as u64).max(start);

        let start = start - range.start;
        let end = end - range.start;
        Some(start..end)
    };

    let x_tiles = x_axis
        .tiles_covered_by_range(view.x_range())?
        .collect::<Vec<_>>();
    let y_tiles = y_axis
        .tiles_covered_by_range(view.y_range())?
        .collect::<Vec<_>>();

    // log::info!("x_tiles covered by {:?}: {}", view.x_range(), x_tiles.len());
    // log::info!("y_tiles covered by {:?}: {}", view.y_range(), y_tiles.len());

    let mut tile_bufs: FxHashMap<(usize, usize), (Vec<egui::Color32>, UVec2, egui::Rect)> =
        FxHashMap::default();

    for target_id in x_tiles {
        for &query_id in &y_tiles {
            let key = (target_id, query_id);
            let Some(line_id) = app.paf_input.pair_line_ix.get(&key) else {
                continue;
            };

            let tgt_name = app.seq_names.get_by_right(&target_id).unwrap();
            let qry_name = app.seq_names.get_by_right(&query_id).unwrap();

            // log::info!("drawing [{tgt_name}, {qry_name}]");

            // get the "local range" for the tile, intersecting with the view
            let target_range = clamped_range(&x_axis, target_id, view.x_range()).unwrap();
            let query_range = clamped_range(&y_axis, query_id, view.y_range()).unwrap();

            if query_range.is_empty() || target_range.is_empty() {
                continue;
            }

            // log::info!("  > target_range: {target_range:?}");
            // log::info!("  > query_range: {query_range:?}");

            // compute from canvas_size & the proportions of target/query range
            // vs the view
            let bp_width = target_range.end - target_range.start;
            let bp_height = query_range.end - query_range.start;

            let px_width = (bp_width as f64 / view.width()) * canvas_size.x as f64;
            let px_height = (bp_height as f64 / view.height()) * canvas_size.y as f64;

            let subcanvas_size = UVec2::new(px_width as u32, px_height as u32);

            let mut buf = Vec::new();

            let top_left = [target_range.start as f64, query_range.start as f64];
            let btm_right = [target_range.end as f64, query_range.end as f64];

            let size = ultraviolet::Vec2::new(canvas_size.x as f32, canvas_size.y as f32);
            let p0: [f32; 2] = view.map_world_to_screen(size, top_left).into();
            let p1: [f32; 2] = view.map_world_to_screen(size, btm_right).into();

            let screen_rect = egui::Rect::from_two_pos(p0.into(), p1.into());

            draw_subsection(
                &app.paf_input.processed_lines[*line_id],
                target_range,
                query_range,
                subcanvas_size,
                &mut buf,
            );

            tile_bufs.insert(key, (buf, subcanvas_size, screen_rect));
        }
    }

    // log::info!("using 1 out of {} tiles", tile_bufs.len());

    return tile_bufs.into_iter().next().map(|(_, a)| a);

    /*
    // then draw each tile to its own buffer
    let img_size = (canvas_size.x * canvas_size.y) as usize;
    let mut pixel_buf = vec![egui::Color32::TRANSPARENT; img_size];
    log::info!("canvas size: {canvas_size:?}");

    for ((_tgt_id, _qry_id), (tile_pixels, tile_rect)) in tile_bufs {
        // find the rows & columns in `pixel_buf` that corresponds
        // to the current pair
        // log::info!("tile_rect: {tile_rect:?}");

        let src_width = tile_rect.width() as usize;

        let left = tile_rect.left() as usize;
        let right = tile_rect.right() as usize;
        let top = tile_rect.top() as usize;
        let bottom = tile_rect.bottom() as usize;

        for y in top..bottom {
            let i = y * canvas_size.x as usize;
            let range = (i + left)..(i + right - 1);
            let src_y_offset = y - top;
            let src_i = src_y_offset * src_width;
            let mut src_range = src_i..(src_i + src_width);

            if src_range.len() > range.len() {
                src_range.end = src_range.start + range.len();
            }

            // dbg!(&src_range);
            // dbg!(&range);

            let src_slice = &tile_pixels[src_range];
            let dst_slice = &mut pixel_buf[range];
            dst_slice.clone_from_slice(src_slice);
            // pixel_buf[range].clone_from_slice(src_slice);
        }
    }

    Some(pixel_buf)
    */
}

/// pixel/bp-perfect CPU rasterization

#[derive(Default)]
pub struct ExactRenderViewDebug {
    textures: FxHashMap<[usize; 2], (egui::TextureId, UVec2)>,
    last_texture: Option<SizedTexture>,
}

impl ExactRenderViewDebug {
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        app: &crate::PafViewerApp,
        window_dims: impl Into<UVec2>,
        view: &crate::view::View,
    ) {
        egui::Window::new("exact render view test").show(ctx, |ui| {
            let mut clicked = false;

            egui::Grid::new("exact-debug-renderer-grid")
                .num_columns(2)
                .show(ui, |ui| {
                    if ui.button("Render").clicked() {
                        clicked = true;
                    }
                    // });

                    if let Some(texture) = self.last_texture {
                        ui.image(egui::ImageSource::Texture(texture));
                    }
                });

            // let x_axis = &app.alignment_grid.x_axis;
            // let y_axis = &app.alignment_grid.y_axis;

            let window_dims = window_dims.into();

            if clicked {
                if let Some((pixels, size, rect)) = draw_exact_to_cpu_buffer(app, window_dims, view)
                {
                    let tex_mgr = ctx.tex_manager();
                    let mut tex_mgr = tex_mgr.write();
                    let tex_id = tex_mgr.alloc(
                        "ExactRenderTexture".into(),
                        ImageData::Color(
                            ColorImage {
                                // size: [window_dims.x as usize, window_dims.y as usize],
                                // size: [rect.width() as usize, rect.height() as usize],
                                size: [size.x as usize, size.y as usize],
                                pixels,
                            }
                            .into(),
                        ),
                        TextureOptions::LINEAR,
                    );
                    let size = window_dims / 2;

                    self.last_texture =
                        Some(SizedTexture::new(tex_id, [size.x as f32, size.y as f32]));
                    // Some(SizedTexture::new(tex_id, [size.x as f32, size.y as f32]));
                }
            }
        });
    }
}

#[derive(Default)]
pub struct ExactRenderDebug {
    // last_params: Option<RenderParams>,
    last_texture: Option<SizedTexture>,

    textures: FxHashMap<[usize; 2], (egui::TextureId, UVec2)>,
    // egui_id_target: egui::Id,
    // egui_id_query: egui::Id,
}

impl ExactRenderDebug {
    /*
     */

    pub const TARGET_DATA_ID: &'static str = "exact-render-test-target";
    pub const QUERY_DATA_ID: &'static str = "exact-render-test-query";

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        app: &crate::PafViewerApp,
        // seq_names: &bimap::BiMap<String, usize>,
        // paf_input: &crate::PafInput,
    ) {
        let parse_range =
            |axis: &crate::grid::GridAxis, txt: &str| -> Option<(usize, std::ops::Range<u64>)> {
                let mut split = txt.split(':');
                let name = split.next()?;
                let id = *app.seq_names.get_by_left(name)?;

                let offset = axis.sequence_offset(id)?;
                // let offset = seqs[id].offset;

                let mut range = split
                    .next()?
                    .split('-')
                    .filter_map(|s| s.parse::<u64>().ok());
                let start = range.next()? + offset;
                let end = range.next()? + offset;

                Some((id, start..end))
            };

        egui::Window::new("exact render test").show(ctx, |ui| {
            let target_id = egui::Id::new(Self::TARGET_DATA_ID);
            let query_id = egui::Id::new(Self::QUERY_DATA_ID);
            // let target_id = ui.id().with("target-range");
            // let query_id = ui.id().with("query-range");

            let (mut target_buf, mut query_buf) = ui
                .data(|data| {
                    let t = data.get_temp::<String>(target_id)?;
                    let q = data.get_temp::<String>(query_id)?;
                    Some((t, q))
                })
                .unwrap_or_default();

            let mut clicked = false;

            egui::Grid::new("exact-debug-renderer-grid")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Target");
                            ui.text_edit_singleline(&mut target_buf);
                        });

                        ui.horizontal(|ui| {
                            ui.label("Query");
                            ui.text_edit_singleline(&mut query_buf);
                        });

                        if ui.button("Render").clicked() {
                            clicked = true;
                        }
                    });

                    if let Some(texture) = self.last_texture {
                        ui.image(egui::ImageSource::Texture(texture));
                    }
                });

            let x_axis = &app.alignment_grid.x_axis;
            let y_axis = &app.alignment_grid.y_axis;

            if clicked {
                // let (tgt_id, tgt_range) = parse_range(x_axis, &target_buf);
                // let (qry_id, qry_range) = parse_range(y_axis, &query_buf);
                let target = parse_range(x_axis, &target_buf);
                let query = parse_range(y_axis, &query_buf);

                if let (Some((tgt_id, tgt_range)), Some((qry_id, qry_range))) = (target, query) {
                    let line = *app.paf_input.pair_line_ix.get(&(tgt_id, qry_id)).unwrap();
                    let match_data = &app.paf_input.processed_lines[line];

                    let mut pixels = Vec::new();
                    draw_subsection(
                        match_data,
                        tgt_range,
                        qry_range,
                        [500, 500].into(),
                        &mut pixels,
                    );

                    let tex_mgr = ctx.tex_manager();
                    let mut tex_mgr = tex_mgr.write();
                    let tex_id = tex_mgr.alloc(
                        "ExactRenderTexture".into(),
                        ImageData::Color(
                            ColorImage {
                                size: [500, 500],
                                pixels,
                            }
                            .into(),
                        ),
                        TextureOptions::LINEAR,
                    );

                    self.last_texture = Some(SizedTexture::new(tex_id, [500.0, 500.0]));
                }
            }

            ui.data_mut(|data| {
                data.insert_temp(target_id, target_buf);
                data.insert_temp(query_id, query_buf);
            });
        });
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct RenderParams {
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas_size: UVec2,
}

pub fn draw_subsection(
    match_data: &crate::ProcessedCigar,
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas_size: UVec2,
    canvas_data: &mut Vec<egui::Color32>,
) {
    let size = (canvas_size.x * canvas_size.y) as usize;
    canvas_data.clear();
    canvas_data.resize(size, egui::Color32::TRANSPARENT);

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

    // for ([target_pos, query_pos], is_match) in match_iter {
    for ([target_pos, query_pos], cg_ix) in match_iter {
        // want to map target_pos to an x_range, query_pos to a y_range

        let cg_op = match_data.cigar[cg_ix].0;
        let is_match = cg_op.is_match();

        let color = if is_match {
            egui::Color32::BLACK
        } else {
            egui::Color32::RED
        };

        // let target_offset = target_pos;
        // let query_offset = query_pos;
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

struct MatchOpIter<'a> {
    match_offsets: &'a [[u64; 2]],
    match_cg_ix: &'a [usize],
    cigar: &'a [(CigarOp, u64)],

    target_range: std::ops::Range<u64>,

    index: usize,
    // current_match: Option<(usize, [u64; 2], bool)>,
    current_match: Option<(CigarOp, usize, std::ops::Range<u64>, [u64; 2])>,
    // current_match: Option<(std::ops::Range<u64>, [u64; 2], bool, bool)>,
    done: bool,
}

impl<'a> MatchOpIter<'a> {
    fn from_range(
        match_offsets: &'a [[u64; 2]],
        match_cg_ix: &'a [usize],
        cigar: &'a [(CigarOp, u64)],
        target_range: std::ops::Range<u64>,
    ) -> Self {
        let t_start = match_offsets.partition_point(|[t, _]| *t < target_range.start);

        let t_start = t_start.checked_sub(1).unwrap_or_default();

        Self {
            // data: match_data,
            match_offsets,
            match_cg_ix,
            cigar,

            target_range,

            index: t_start,
            current_match: None,

            done: false,
        }
    }
}

impl<'a> Iterator for MatchOpIter<'a> {
    // outputs each individual match/mismatch op's position along the
    // target and query
    type Item = ([u64; 2], usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.current_match.is_none() {
            if self.index >= self.match_offsets.len() {
                return None;
            }
            let ix = self.index;
            self.index += 1;
            let cg_ix = self.match_cg_ix[ix];
            let (op, count) = self.cigar[cg_ix];
            let range = 0..count;
            let origin = self.match_offsets[ix];
            self.current_match = Some((op, cg_ix, range, origin));
        }

        if let Some((op, cg_ix, mut range, origin @ [tgt, qry])) = self.current_match.take() {
            let next_offset = range.next()?;

            if tgt + next_offset > self.target_range.end {
                self.done = true;
            }

            if !range.is_empty() {
                self.current_match = Some((op, cg_ix, range, origin));
            }

            let pos = [tgt + next_offset, qry + next_offset];
            let out = (pos, cg_ix);

            return Some(out);
        }

        None
    }
}

#[cfg(test)]
mod tests {

    use ultraviolet::DVec2;

    use super::*;
    use crate::ProcessedCigar;

    fn test_cigar() -> Vec<(CigarOp, u64)> {
        use crate::CigarOp as C;

        vec![
            (C::M, 10),
            (C::X, 1),
            (C::M, 13),
            (C::D, 7),
            (C::M, 13),
            (C::X, 1),
            (C::M, 21),
            (C::I, 13),
            (C::M, 18),
            (C::X, 5),
            (C::M, 39),
            (C::X, 1),
            (C::M, 3),
        ]
    }

    fn cigar_offsets(cg: &[(CigarOp, u64)]) -> (Vec<[u64; 2]>, Vec<usize>) {
        let mut offsets = Vec::new();
        let mut indices = Vec::new();

        let mut origin = [0u64, 0];

        for (cg_ix, &(op, count)) in cg.iter().enumerate() {
            if op.is_match_or_mismatch() {
                offsets.push(origin);
                indices.push(cg_ix);
            }
            origin = op.apply_to_offsets(count, origin);
        }

        (offsets, indices)
    }

    #[test]
    fn test_match_op_iter() {
        let cigar = test_cigar();
        let (offsets, cg_ix) = cigar_offsets(&cigar);

        let len = cigar.iter().map(|(_, c)| *c).sum::<u64>();

        let iter = MatchOpIter::from_range(&offsets, &cg_ix, &cigar, 0..len);

        for ([tgt, qry], is_match) in iter {
            println!("[{tgt:3}, {qry:3}] - {is_match}");
        }

        println!();

        let iter = MatchOpIter::from_range(&offsets, &cg_ix, &cigar, 15..40);

        for ([tgt, qry], is_match) in iter {
            println!("[{tgt:3}, {qry:3}] - {is_match}");
        }
    }
}
