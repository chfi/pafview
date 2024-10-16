use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{paf::AlignmentIndex, sequences::SeqId, IndexedCigar};

use super::{
    color::{AlignmentColorScheme, GPUColorScheme},
    lines::LineColorSchemePipeline,
};

pub struct MatchDrawBatchData {
    // TODO: the cigars can't be combined if i want to color them separately
    // alignment_pair_index: FxHashMap<(SeqId, SeqId), usize>,
    alignment_buffer_index: FxHashMap<crate::paf::AlignmentIndex, usize>,

    buffers: DrawBatchBuffers,
}

#[derive(Default)]
pub struct ColorSchemeBuffers {
    indices: FxHashMap<AlignmentColorScheme, usize>,

    buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl ColorSchemeBuffers {
    pub fn from_color_schemes(
        device: &wgpu::Device,
        pipeline: &LineColorSchemePipeline,
        colors: &super::color::PafColorSchemes,
    ) -> Self {
        let mut result = Self::default();

        result.append_color_scheme(device, pipeline, &colors.default);

        for scheme in colors.overrides.values() {
            result.append_color_scheme(device, pipeline, scheme);
        }

        result
    }

    fn append_color_scheme(
        &mut self,
        device: &wgpu::Device,
        pipeline: &LineColorSchemePipeline,
        color_scheme: &AlignmentColorScheme,
    ) {
        if self.indices.contains_key(color_scheme) {
            return;
        };

        let ix = self.buffers.len();

        let color_data = GPUColorScheme::from_color_scheme(color_scheme);

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("Alignment color scheme #{ix} buffer")),
            contents: bytemuck::cast_slice(&[color_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Alignment color scheme #{ix} bind group")),
            layout: &pipeline.bind_group_layout_1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        self.indices.insert(color_scheme.clone(), ix);
        self.buffers.push(buffer);
        self.bind_groups.push(bind_group);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: Vec2,
    p1: Vec2,
}

impl super::PafRenderer {
    pub(super) fn draw_frame_tiled_color_schemes(
        line_pipeline: &super::LinePipeline,
        color_scheme_pipeline: &LineColorSchemePipeline,
        color_buffers: &ColorSchemeBuffers,
        alignment_colors: &super::color::PafColorSchemes,
        batch_data: &MatchDrawBatchData,
        params: &super::PafDrawSet,
        uniforms: &super::PafUniforms,
        identity_uniform: &wgpu::BindGroup,
        grid_data: &Option<(wgpu::Buffer, wgpu::Buffer, std::ops::Range<u32>)>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let attch = if let Some(msaa_view) = &params.framebuffers.msaa_view {
            wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(&params.framebuffers.color_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: &params.framebuffers.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(attch)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&line_pipeline.pipeline);

        if let Some((grid_vertices, grid_colors, grid_instances)) = grid_data {
            rpass.set_bind_group(0, &uniforms.grid_bind_group, &[]);
            rpass.set_bind_group(1, identity_uniform, &[]);
            rpass.set_vertex_buffer(0, grid_vertices.slice(..));
            rpass.set_vertex_buffer(1, grid_colors.slice(..));
            rpass.draw(0..6, grid_instances.clone());
        }

        rpass.set_pipeline(&color_scheme_pipeline.pipeline);
        rpass.set_bind_group(0, &uniforms.line_bind_group, &[]);

        for (&align_ix, &buf_ix) in &batch_data.alignment_buffer_index {
            let color_scheme = alignment_colors.get(align_ix);

            let Some(color_ix) = color_buffers.indices.get(color_scheme) else {
                continue;
            };
            let color_bind_group = &color_buffers.bind_groups[*color_ix];

            rpass.set_bind_group(1, color_bind_group, &[]);
            rpass.set_bind_group(2, &batch_data.buffers.tile_pos_bind_groups[buf_ix], &[]);
            rpass.set_vertex_buffer(0, batch_data.buffers.vertex_pos_buffers[buf_ix].slice(..));
            rpass.set_vertex_buffer(1, batch_data.buffers.vertex_color_buffers[buf_ix].slice(..));
            rpass.draw(0..6, batch_data.buffers.vertex_instances[buf_ix].clone());

            // if let Some(color_scheme) =
            // rpass.set_bind_group(1, &col
            // rpass.set_bind_group(1, &batch_data.buffers.tile_pos_bind_groups[buf_ix], &[]);
            // rpass.set_vertex_buffer(0, batch_data.buffers.vertex_pos_buffers[buf_ix].slice(..));
            // rpass.set_vertex_buffer(1, batch_data.buffers.vertex_color_buffers[buf_ix].slice(..));
            // rpass.draw(0..6, batch_data.buffers.vertex_instances[buf_ix].clone());
        }
    }

    pub(super) fn draw_frame_tiled(
        batch_data: &MatchDrawBatchData,
        line_pipeline: &super::LinePipeline,
        params: &super::PafDrawSet,
        uniforms: &super::PafUniforms,
        identity_uniform: &wgpu::BindGroup,
        grid_data: &Option<(wgpu::Buffer, wgpu::Buffer, std::ops::Range<u32>)>,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let attch = if let Some(msaa_view) = &params.framebuffers.msaa_view {
            wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(&params.framebuffers.color_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: &params.framebuffers.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(attch)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_pipeline(&line_pipeline.pipeline);

        if let Some((grid_vertices, grid_colors, grid_instances)) = grid_data {
            rpass.set_bind_group(0, &uniforms.grid_bind_group, &[]);
            rpass.set_bind_group(1, identity_uniform, &[]);
            rpass.set_vertex_buffer(0, grid_vertices.slice(..));
            rpass.set_vertex_buffer(1, grid_colors.slice(..));
            rpass.draw(0..6, grid_instances.clone());
        }

        rpass.set_bind_group(0, &uniforms.line_bind_group, &[]);

        for (_align_ix, &buf_ix) in &batch_data.alignment_buffer_index {
            rpass.set_bind_group(1, &batch_data.buffers.tile_pos_bind_groups[buf_ix], &[]);
            rpass.set_vertex_buffer(0, batch_data.buffers.vertex_pos_buffers[buf_ix].slice(..));
            rpass.set_vertex_buffer(1, batch_data.buffers.vertex_color_buffers[buf_ix].slice(..));
            rpass.draw(0..6, batch_data.buffers.vertex_instances[buf_ix].clone());
        }
    }
}

impl MatchDrawBatchData {
    pub(super) fn draw_matches<'a: 'b, 'b>(
        // &'a self,
        index: &'a FxHashMap<(usize, usize), usize>,
        buffers: &'a DrawBatchBuffers,
        line_pipeline: &'a super::LinePipeline,
        params: &'a super::PafDrawSet,
        uniforms: &'b super::PafUniforms,
        rpass: &'b mut wgpu::RenderPass<'b>,
    ) {
        rpass.set_pipeline(&line_pipeline.pipeline);
        rpass.set_bind_group(0, &uniforms.line_bind_group, &[]);

        for (&(_tgt, _qry), &buf_ix) in index {
            rpass.set_bind_group(1, &buffers.tile_pos_bind_groups[buf_ix], &[]);
            rpass.set_vertex_buffer(0, buffers.vertex_pos_buffers[buf_ix].slice(..));
            rpass.set_vertex_buffer(1, buffers.vertex_color_buffers[buf_ix].slice(..));
            rpass.draw(0..6, buffers.vertex_instances[buf_ix].clone());
        }
    }

    // pub fn from_paf_input(
    pub fn from_alignments(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        alignment_grid: &crate::AlignmentGrid,
        alignments: &crate::Alignments,
    ) -> Self {
        let mut buffers = DrawBatchBuffers::default();

        let mut vertex_position_tmp: Vec<LineVertex> = Vec::new();
        // let mut vertex_color_tmp: Vec<egui::Color32> = Vec::new();
        let mut vertex_color_tmp: Vec<u32> = Vec::new();

        let mut alignment_buffer_index = FxHashMap::default();

        for (pair @ &(target_id, query_id), alignments) in alignments.pairs.iter() {
            // for (line_ix, input_line) in input.processed_lines.iter().enumerate() {

            for (cg_ix, alignment) in alignments.iter().enumerate() {
                let buf_ix = buffers.vertex_pos_buffers.len();

                let align_ix = AlignmentIndex {
                    pair: *pair,
                    index: cg_ix,
                };
                alignment_buffer_index.insert(align_ix, buf_ix);

                vertex_position_tmp.clear();
                vertex_color_tmp.clear();

                let op_line_vertices =
                    line_vertices_from_cigar(&alignment.location, alignment.cigar.whole_cigar());

                for (&[from, to], (op, _count)) in
                    op_line_vertices.iter().zip(alignment.cigar.whole_cigar())
                {
                    use crate::CigarOp::{Eq, D, I, M, X};
                    if matches!(op, I | D) {
                        continue;
                    }

                    let color_ix = match op {
                        M => 0u32,
                        Eq => 1,
                        X => 2,
                        I => 3,
                        D => 4,
                    };

                    let p0 = Vec2::new(from.x as f32, from.y as f32);
                    let p1 = Vec2::new(to.x as f32, to.y as f32);

                    vertex_position_tmp.push(LineVertex { p0, p1 });
                    vertex_color_tmp.push(color_ix);
                }

                let match_count = vertex_position_tmp.len();

                let pos_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertex_position_tmp),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

                let color_buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertex_color_tmp),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

                buffers.vertex_pos_buffers.push(pos_buffer);
                buffers.vertex_color_buffers.push(color_buffer);
                buffers.vertex_instances.push(0..match_count as u32);

                let x_offset = alignment_grid
                    .x_axis
                    .sequence_offset(target_id)
                    .unwrap_or_default() as f32;
                let y_offset = alignment_grid
                    .y_axis
                    .sequence_offset(query_id)
                    .unwrap_or_default() as f32;

                let pos_mat = Mat4::from_translation(Vec3::new(x_offset, y_offset, 0.0));
                let pos_uniform = device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[pos_mat]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pos_uniform.as_entire_binding(),
                    }],
                });

                buffers.tile_pos_uniforms.push(pos_uniform);
                buffers.tile_pos_bind_groups.push(bind_group);
            }
        }

        Self {
            alignment_buffer_index,
            buffers,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct DrawBatchBuffers {
    vertex_pos_buffers: Vec<wgpu::Buffer>,
    vertex_color_buffers: Vec<wgpu::Buffer>,
    vertex_instances: Vec<std::ops::Range<u32>>,

    tile_pos_uniforms: Vec<wgpu::Buffer>,
    tile_pos_bind_groups: Vec<wgpu::BindGroup>,
}

pub fn line_vertices_from_cigar(
    location: &crate::paf::AlignmentLocation,
    cigar_ops: impl Iterator<Item = (crate::cigar::CigarOp, u32)>,
) -> Vec<[ultraviolet::DVec2; 2]> {
    use crate::cigar::CigarOp;
    use ultraviolet::DVec2;

    let mut vertices = Vec::new();

    let mut tgt_cg = 0;
    let mut qry_cg = 0;

    for (op, count) in cigar_ops {
        // tgt_cg and qry_cg are offsets from the start of the cigar
        let tgt_start = tgt_cg;
        let qry_start = qry_cg;

        let (tgt_end, qry_end) = match op {
            CigarOp::Eq | CigarOp::X | CigarOp::M => {
                tgt_cg += count as u64;
                qry_cg += count as u64;
                //
                (tgt_start + count as u64, qry_start + count as u64)
            }
            CigarOp::I => {
                qry_cg += count as u64;
                //
                (tgt_start, qry_start + count as u64)
            }
            CigarOp::D => {
                tgt_cg += count as u64;
                //
                (tgt_start + count as u64, qry_start)
            }
        };

        let tgt_range = location.map_from_aligned_target_range(tgt_start..tgt_end);
        let qry_range = location.map_from_aligned_query_range(qry_start..qry_end);

        let mut from = DVec2::new(tgt_range.start as f64, qry_range.start as f64);
        let mut to = DVec2::new(tgt_range.end as f64, qry_range.end as f64);

        if location.query_strand.is_rev() {
            std::mem::swap(&mut from.y, &mut to.y);
        }

        vertices.push([from, to]);
    }

    vertices
}
