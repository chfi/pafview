use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::{sequences::SeqId, IndexedCigar};

pub struct MatchDrawBatchData {
    alignment_pair_index: FxHashMap<(SeqId, SeqId), usize>,

    buffers: DrawBatchBuffers,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: Vec2,
    p1: Vec2,
}

impl super::PafRenderer {
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

        for (&(_tgt, _qry), &buf_ix) in &batch_data.alignment_pair_index {
            rpass.set_bind_group(1, &batch_data.buffers.grid_pos_bind_groups[buf_ix], &[]);
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
            rpass.set_bind_group(1, &buffers.grid_pos_bind_groups[buf_ix], &[]);
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
        let mut vertex_color_tmp: Vec<egui::Color32> = Vec::new();

        let mut alignment_pair_index = FxHashMap::default();

        for (&(target_id, query_id), alignments) in alignments.pairs.iter() {
            // for (line_ix, input_line) in input.processed_lines.iter().enumerate() {
            let buf_ix = buffers.vertex_pos_buffers.len();

            alignment_pair_index.insert((target_id, query_id), buf_ix);

            vertex_position_tmp.clear();
            vertex_color_tmp.clear();

            for alignment in alignments {
                let op_line_vertices =
                    line_vertices_from_cigar(&alignment.location, alignment.cigar.whole_cigar());

                // for (&[from, to], &is_match) in alignment.cigar.op_line_vertices.iter()
                for (&[from, to], (op, _count)) in
                    op_line_vertices.iter().zip(alignment.cigar.whole_cigar())
                // .zip(alignment.cigar.cigar.iter())
                {
                    use crate::CigarOp::{D, I};
                    if matches!(op, I | D) {
                        continue;
                    }

                    let is_match = op.is_match();
                    let color = if is_match {
                        egui::Color32::BLACK
                    } else {
                        egui::Color32::RED
                    };

                    let p0 = Vec2::new(from.x as f32, from.y as f32);
                    let p1 = Vec2::new(to.x as f32, to.y as f32);

                    vertex_position_tmp.push(LineVertex { p0, p1 });
                    vertex_color_tmp.push(color);
                }
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

            buffers.grid_pos_uniforms.push(pos_uniform);
            buffers.grid_pos_bind_groups.push(bind_group);
        }

        Self {
            alignment_pair_index,
            buffers,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct DrawBatchBuffers {
    vertex_pos_buffers: Vec<wgpu::Buffer>,
    vertex_color_buffers: Vec<wgpu::Buffer>,
    vertex_instances: Vec<std::ops::Range<u32>>,

    grid_pos_uniforms: Vec<wgpu::Buffer>,
    grid_pos_bind_groups: Vec<wgpu::BindGroup>,
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

        if tgt_end < tgt_start || qry_end < qry_start {
            dbg!(tgt_start..tgt_end);
            dbg!(qry_start..qry_end);
        }
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
