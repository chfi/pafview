use egui::{load::SizedTexture, ColorImage, ImageData, TextureOptions};
use rustc_hash::FxHashMap;
use ultraviolet::UVec2;

use crate::{
    cigar::{CigarOp, ProcessedCigar},
    grid::{AxisRange, GridAxis},
};

#[derive(Default)]
pub struct CpuViewRasterizerEgui {
    pub(super) last_wgpu_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

pub(super) struct CpuRasterizerBindGroups {
    pub(super) image_bind_groups: [super::ImageRendererBindGroups; 2],
    // pub(super) texture_view_id: wgpu::Id<wgpu::TextureView>,
}

impl CpuRasterizerBindGroups {
    pub(super) fn new(device: &wgpu::Device, image_renderer: &super::ImageRenderer) -> Self {
        let bind_group_0_layout = image_renderer.pipeline.get_bind_group_layout(0);

        // NB: double buffering is probably not strictly necessary right now, but
        // might as well have it so it's easier to thread later
        let bind_groups_front = super::ImageRendererBindGroups::new(device, &bind_group_0_layout);
        let bind_groups_back = super::ImageRendererBindGroups::new(device, &bind_group_0_layout);

        Self {
            image_bind_groups: [bind_groups_front, bind_groups_back],
        }
    }

    pub(super) fn create_bind_groups(
        &mut self,
        device: &wgpu::Device,
        image_renderer: &super::ImageRenderer,
        texture_view: &wgpu::TextureView,
    ) {
        image_renderer.create_bind_groups(device, &mut self.image_bind_groups[0], texture_view);
        image_renderer.create_bind_groups(device, &mut self.image_bind_groups[1], texture_view);
    }
}

impl CpuViewRasterizerEgui {
    pub fn draw_into_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window_dims: [u32; 2],
        app: &crate::PafViewerApp,
        view: &crate::view::View,
    ) {
        let [width, height] = window_dims;
        let need_realloc = if let Some((texture, _)) = self.last_wgpu_texture.as_ref() {
            let extent = texture.size();
            width != extent.width || height != extent.height
        } else {
            true
        };

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        if need_realloc {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("BpRasterizer Texture"),
                size: extent,
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

        if let Some(pixels) = draw_exact_to_cpu_buffer(app, window_dims, view) {
            let extent = wgpu::Extent3d {
                width: pixels.width,
                height: pixels.height,
                depth_or_array_layers: 1,
            };

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                    // origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&pixels.pixels),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * extent.width),
                    rows_per_image: None,
                },
                extent,
            )
        }
    }
}

fn draw_exact_to_cpu_buffer(
    app: &crate::PafViewerApp,
    canvas_size: impl Into<UVec2>,
    view: &crate::view::View,
    // ) -> Option<(Vec<egui::Color32>, UVec2, egui::Rect)> {
) -> Option<PixelBuffer> {
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

            // compute from canvas_size & the proportions of target/query range
            // vs the view
            let bp_width = target_range.end - target_range.start;
            let bp_height = query_range.end - query_range.start;

            let px_width = (bp_width as f64 / view.width()) * canvas_size.x as f64;
            let px_height = (bp_height as f64 / view.height()) * canvas_size.y as f64;

            let subcanvas_size = UVec2::new(px_width as u32, px_height as u32);

            let mut buf = Vec::new();

            let x_range =
                x_axis.axis_range_into_global(&AxisRange::seq(target_id, target_range.clone()))?;
            let y_range =
                y_axis.axis_range_into_global(&AxisRange::seq(query_id, query_range.clone()))?;

            let top_left = [*x_range.start(), *y_range.start()];
            let btm_right = [*x_range.end(), *y_range.end()];

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

    let mut px_buffer = PixelBuffer::new_color(canvas_size.x, canvas_size.y, egui::Color32::WHITE);

    for (tile, (tile_pixels, tile_size, tile_rect)) in tile_bufs {
        let dst_o = tile_rect.left_top();
        let dst_offset = [dst_o.x as i32, dst_o.y as i32];

        px_buffer.blit_from_slice(dst_offset, tile_size, &tile_pixels);
    }

    Some(px_buffer)
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

struct PixelBuffer {
    width: u32,
    height: u32,
    pixels: Vec<egui::Color32>,
}

impl PixelBuffer {
    fn new_color(width: u32, height: u32, color: egui::Color32) -> Self {
        Self {
            width,
            height,
            pixels: vec![color; (width * height) as usize],
        }
    }

    fn new(width: u32, height: u32) -> Self {
        Self::new_color(width, height, egui::Color32::TRANSPARENT)
    }

    fn blit_from_buffer(&mut self, dst_offset: impl Into<[i32; 2]>, src: &Self) {
        let src_size = [src.width, src.height];
        self.blit_from_slice(dst_offset, src_size, &src.pixels);
    }

    fn blit_from_slice(
        &mut self,
        dst_offset: impl Into<[i32; 2]>,
        src_size: impl Into<[u32; 2]>,
        src: &[egui::Color32],
    ) {
        let [x0, y0] = dst_offset.into();
        let [src_width, src_height] = src_size.into();
        debug_assert!(src.len() == (src_width as usize * src_height as usize));

        let (dst_vis_cols, src_vis_cols) = {
            let dst_min = x0.max(0) as u32;
            let dst_max = ((x0 + src_width as i32) as u32).min(self.width);

            let src_min = (-y0).max(0) as u32;
            let src_max = src_min + (dst_max - dst_min);

            (dst_min..dst_max, src_min..src_max)
        };

        let (dst_vis_rows, src_vis_rows) = {
            let dst_min = y0.max(0) as u32;
            let dst_max = ((y0 + src_height as i32) as u32).min(self.height);

            let src_min = (-y0).max(0) as u32;
            let src_max = src_min + (dst_max - dst_min);

            (dst_min..dst_max, src_min..src_max)
        };

        for (dst_row, src_row) in std::iter::zip(dst_vis_rows, src_vis_rows) {
            // for dst_row in dst_vis_rows {
            let col_start = dst_vis_cols.start;
            let col_len = dst_vis_cols.end - col_start;

            let dst_start = dst_row * self.width + col_start;
            let dst_end = dst_start + col_len;
            let dst_range = (dst_start as usize)..(dst_end as usize);

            let dst_slice = &mut self.pixels[dst_range];

            let col_start = src_vis_cols.start;
            let col_end = src_vis_cols.end;
            debug_assert_eq!(col_end - col_start, col_len);

            let src_start = src_row * src_width + col_start;
            let src_end = src_start + col_len;
            let src_range = (src_start as usize)..(src_end as usize);

            let src_slice = &src[src_range];

            dst_slice.copy_from_slice(src_slice);
        }
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

    #[test]
    fn test_pixel_buffer_blit() {
        let mut buf = PixelBuffer::new(48, 48);

        let red = PixelBuffer::new_color(8, 8, egui::Color32::RED);
        let blue = PixelBuffer::new_color(16, 8, egui::Color32::BLUE);
        let green = PixelBuffer::new_color(8, 16, egui::Color32::GREEN);

        buf.blit_from_buffer([-4, 16], &red);
        buf.blit_from_buffer([16, 40], &green);
        buf.blit_from_buffer([40, -3], &blue);

        let mut reds = 0;
        let mut greens = 0;
        let mut blues = 0;

        for &px in buf.pixels.iter() {
            if px == egui::Color32::RED {
                reds += 1;
            } else if px == egui::Color32::GREEN {
                greens += 1;
            } else if px == egui::Color32::BLUE {
                blues += 1;
            }
        }

        // debug_print_pixel_buffer(&buf);
        assert_eq!(reds, 4 * 8);
        assert_eq!(greens, 8 * 8);
        assert_eq!(blues, 8 * 5);
    }

    #[allow(dead_code)]
    fn debug_print_pixel_buffer(buf: &PixelBuffer) {
        let h_border = (0..buf.width + 2).map(|_| '-').collect::<String>();
        println!("{h_border}");
        for row in 0..buf.height {
            print!("|");
            for col in 0..buf.width {
                let ix = (col + row * buf.width) as usize;

                let px = &buf.pixels[ix];

                if !px.is_opaque() {
                    print!(" ");
                } else if *px == egui::Color32::RED {
                    print!("R");
                } else if *px == egui::Color32::BLUE {
                    print!("B");
                } else if *px == egui::Color32::GREEN {
                    print!("G");
                } else {
                    print!(" ");
                }
            }
            println!("|");
        }

        println!("{h_border}");
    }
}
