use bimap::BiMap;
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

pub(super) struct MatchDrawBatchData {
    alignment_pair_index: FxHashMap<(usize, usize), usize>,

    buffers: DrawBatchBuffers,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    p0: Vec2,
    p1: Vec2,
}

impl super::PafRenderer {
    fn submit_batch_draw_matches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &crate::view::View,
        window_dims: [u32; 2],
    ) {
        todo!();
    }
}

impl MatchDrawBatchData {
    fn draw_matches<'a: 'b, 'b>(
        &'a self,
        line_pipeline: &'b super::LinePipeline,
        params: &'b super::PafDrawSet,
        uniforms: &'b super::PafUniforms,
        rpass: &'b mut wgpu::RenderPass<'b>,
    ) {
        rpass.set_pipeline(&line_pipeline.pipeline);

        todo!();
        //
    }

    fn from_paf_input(
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
struct DrawBatchBuffers {
    vertex_pos_buffers: Vec<wgpu::Buffer>,
    vertex_color_buffers: Vec<wgpu::Buffer>,
    vertex_instances: Vec<std::ops::Range<u32>>,

    grid_pos_uniforms: Vec<wgpu::Buffer>,
    grid_pos_bind_groups: Vec<wgpu::BindGroup>,
}
