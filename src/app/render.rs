use std::sync::Arc;

use bevy::{
    ecs::system::{lifetimeless::SRes, EntityCommands},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            CachedRenderPipelineId, PipelineCache, RenderPipelineDescriptor, ShaderType,
            UniformBuffer,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
        view::RenderLayers,
        Extract, Render, RenderApp, RenderSet,
    },
    utils::HashMap,
};
use wgpu::{util::BufferInitDescriptor, ColorWrites, ShaderStages, VertexStepMode};

use crate::{
    math_conv::{ConvertFloat32, ConvertVec2},
    render::color::AlignmentColorScheme,
    CigarOp,
};

use super::view::AlignmentViewport;

// pub mod async_render;
pub mod base_level;
pub mod bordered_rect;
pub mod cigar_sampling;
pub mod gpu_lines;

/*

rendering is done by creating a sprite with the
`AlignmentDisplayImage` component, which then can be given the
`AlignmentRenderTarget` component, with a given alignment view, to
trigger a render (the GPU line renderer or CPU base-level rasterizer
will be used depending on the view scale)

the sprite can also be given a map of alignment position overrides;
if present, only the alignments with overrides will be rendered to
the texture used by the sprite

the plugin setup creates an alignment display sprite that is rendered
to the screenspace camera (`RenderLayer` 1) and updated based on the
`AlignmentViewport` resource



*/

// marker for alignment images that are linked to the main viewport
#[derive(Debug, Component)]
pub struct MainAlignmentView;

#[derive(Component)]
pub(super) struct Rendering;

#[derive(Clone, Copy, PartialEq)]
struct RenderParams {
    view: crate::view::View,
    canvas_size: UVec2,
}

#[derive(Component)]
struct ForceRender;
