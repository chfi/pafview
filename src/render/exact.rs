use egui::{load::SizedTexture, ColorImage, ImageData, TextureOptions};
use rustc_hash::FxHashMap;
use ultraviolet::UVec2;

use crate::{
    cigar::{CigarOp, ProcessedCigar},
    grid::{AxisRange, GridAxis},
    sequences::SeqId,
};

pub mod detail;

use crate::PixelBuffer;

#[derive(Default)]
pub struct CpuViewRasterizerEgui {
    tile_buffers: detail::TileBuffers,

    pub(super) last_wgpu_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl CpuViewRasterizerEgui {
    pub fn initialize() -> Self {
        let tile_buffers = detail::build_op_pixel_buffers();
        Self {
            tile_buffers,
            last_wgpu_texture: None,
        }
    }
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

        let sequences = &app.sequences;
        let grid = &app.alignment_grid;
        let alignments = &app.alignments;

        // if let Some(pixels) = draw_exact_to_cpu_buffer(app, window_dims, view) {
        if let Some(pixels) = detail::draw_alignments(
            &self.tile_buffers,
            sequences,
            grid,
            alignments,
            view,
            window_dims,
        ) {
            let extent = wgpu::Extent3d {
                width: pixels.width as u32,
                height: pixels.height as u32,
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

/*
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
                         seq_id: SeqId,
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

    let mut tile_bufs: FxHashMap<(SeqId, SeqId), (Vec<egui::Color32>, UVec2, egui::Rect)> =
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
*/

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
struct RenderParams {
    target_range: std::ops::Range<u64>,
    query_range: std::ops::Range<u64>,
    canvas_size: UVec2,
}

#[cfg(test)]
mod tests {

    use std::hash::Hash;

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
}
