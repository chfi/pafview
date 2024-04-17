use bimap::BiMap;
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::sequences::SeqId;

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

    pub fn from_paf_input(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        alignment_grid: &crate::AlignmentGrid,
        input: &crate::PafInput,
    ) -> Self {
        let mut buffers = DrawBatchBuffers::default();

        let mut vertex_position_tmp: Vec<LineVertex> = Vec::new();
        let mut vertex_color_tmp: Vec<egui::Color32> = Vec::new();

        let mut alignment_pair_index = FxHashMap::default();

        for (line_ix, input_line) in input.processed_lines.iter().enumerate() {
            let buf_ix = buffers.vertex_pos_buffers.len();

            let target_id = input_line.target_id;
            let query_id = input_line.query_id;

            alignment_pair_index.insert((target_id, query_id), buf_ix);

            vertex_position_tmp.clear();
            vertex_color_tmp.clear();

            let match_count = input_line.match_edges.len() as u32;

            for (&[from, to], &is_match) in input_line
                .match_edges
                .iter()
                .zip(&input_line.match_is_match)
            {
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
            buffers.vertex_instances.push(0..match_count);

            let x_offset = alignment_grid
                .x_axis
                .sequence_offset(input_line.target_id)
                .unwrap_or_default() as f32;
            let y_offset = alignment_grid
                .y_axis
                .sequence_offset(input_line.query_id)
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
