use bevy::{
    ecs::system::lifetimeless::SRes,
    prelude::*,
    render::{
        render_asset::{PrepareAssetError, RenderAsset},
        render_resource::{BindGroup, BindGroupLayout, Buffer, ShaderType, UniformBuffer},
        renderer::{RenderDevice, RenderQueue},
    },
};
use wgpu::util::BufferInitDescriptor;

use crate::{render::color::AlignmentColorScheme, CigarOp};

/*



*/

pub struct AlignmentRendererPlugin;

impl Plugin for AlignmentRendererPlugin {
    fn build(&self, app: &mut App) {
        todo!();
    }
}

#[derive(Clone, Resource)]
pub struct AlignmentPolylinePipeline {
    pub proj_config_layout: BindGroupLayout,
    pub color_scheme_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

struct AlignmentPolylineBindGroups {
    group_0: BindGroup,
    group_1: BindGroup,
    group_2: BindGroup,
}

struct GpuAlignmentPolylineMaterial {
    proj_buffer: UniformBuffer<Mat4>,
    config_buffer: UniformBuffer<GpuAlignmentRenderConfig>,
    color_scheme_buffer: UniformBuffer<GpuAlignmentColorScheme>,
    model_buffer: UniformBuffer<Mat4>,

    bind_groups: AlignmentPolylineBindGroups,
}

/*
impl RenderAsset for GpuAlignmentPolylineMaterial {
    type SourceAsset = AlignmentPolylineMaterial;

    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SRes<AlignmentPolylinePipeline>,
    );

    fn prepare_asset(
        source_asset: Self::SourceAsset,
        param: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self, bevy::render::render_asset::PrepareAssetError<Self::SourceAsset>> {


        todo!()
    }
}
*/

#[derive(Component, Clone, Copy, PartialEq, Debug)]
pub struct AlignmentRenderConfig {
    pub line_width: f32,
    pub brightness: f32,
}

#[derive(ShaderType, Component, Clone)]
struct GpuAlignmentRenderConfig {
    line_width: f32,
    brightness: f32,
    _pad0: f32,
    _pad1: f32,
}

// #[derive(ShaderType, Component, Clone)]
#[derive(Asset, Debug, PartialEq, Clone, TypePath)]
struct AlignmentPolylineMaterial {
    projection: Mat4,

    config: AlignmentRenderConfig,
    color_scheme: AlignmentColorScheme,
    // model: Mat4,
    // projection: UniformBuffer<Mat4>,
    // config: UniformBuffer<GpuAlignmentConfig>,
}

#[derive(ShaderType, Component, Clone)]
struct GpuAlignmentColorScheme {
    m_bg: Vec4,
    eq_bg: Vec4,
    x_bg: Vec4,
    i_bg: Vec4,
    d_bg: Vec4,
}

#[derive(ShaderType, Component, Clone)]
struct AlignmentModel {
    model: Mat4,
}

// struct GpuAlignmentModel {
//     model:
// }

#[derive(Debug, Default, Asset, Clone, TypePath)]
pub struct AlignmentVertices {
    data: Vec<(Vec2, Vec2, CigarOp)>,
    // data: Vec<u8>,
}

struct GpuAlignmentVertices {
    vertex_buffer: Buffer,
    segment_count: u32,
}

impl RenderAsset for GpuAlignmentVertices {
    type SourceAsset = AlignmentVertices;

    type Param = SRes<RenderDevice>;

    fn prepare_asset(
        vertices: Self::SourceAsset,
        render_device: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let segment_count = vertices.data.len() as u32;
        let data = vertices
            .data
            .into_iter()
            .flat_map(|(from, to, op)| {
                let mut out = [0u8; 4 * 5];

                let op_ix = match op {
                    CigarOp::M => 0u32,
                    CigarOp::Eq => 1,
                    CigarOp::X => 2,
                    CigarOp::I => 3,
                    CigarOp::D => 4,
                };

                out[0..16].clone_from_slice(bytemuck::cast_slice(&[from, to]));
                out[16..20].clone_from_slice(bytemuck::cast_slice(&[op_ix]));

                out
            })
            .collect::<Vec<_>>();

        let vertex_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            usage: wgpu::BufferUsages::VERTEX,
            label: Some("Alignment Vertex Buffer"),
            contents: &data,
        });

        Ok(GpuAlignmentVertices {
            vertex_buffer,
            segment_count,
        })
    }
}
