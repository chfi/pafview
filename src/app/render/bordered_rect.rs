use bevy::{
    asset::load_internal_asset,
    prelude::*,
    render::render_resource::AsBindGroup,
    sprite::{Material2d, Material2dPlugin},
};

pub struct BorderedRectRenderPlugin;

const BORDERED_RECT_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(66283402603114632559453785259996878691);

const BORDERED_RECT_2D_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(320771424793966079501694784629340793754);

const BORDERED_RECT_UTIL_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(62328999086576302936194172130037625200);

impl Plugin for BorderedRectRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            BORDERED_RECT_UTIL_SHADER_HANDLE,
            "bordered_rect_util.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            BORDERED_RECT_SHADER_HANDLE,
            "bordered_rect.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            BORDERED_RECT_2D_SHADER_HANDLE,
            "bordered_rect_2d.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(MaterialPlugin::<BorderedRectMaterial>::default())
            .add_plugins(Material2dPlugin::<BorderedRectMaterial2d>::default());
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct BorderedRectMaterial {
    #[uniform(0)]
    pub fill_color: LinearRgba,

    #[uniform(1)]
    pub border_color: LinearRgba,

    // 8 bits/0-255 for each of the four sides: N/E/S/W
    #[uniform(2)]
    pub border_opacities: u32,

    #[uniform(3)]
    pub border_width_px: f32,

    pub alpha_mode: AlphaMode,
}

impl Default for BorderedRectMaterial {
    fn default() -> Self {
        Self {
            fill_color: Color::NONE.into(),
            border_color: Color::BLACK.into(),
            border_opacities: 0xFFFFFFFF,
            border_width_px: 1.0,
            alpha_mode: AlphaMode::Blend,
        }
    }
}

impl Material for BorderedRectMaterial {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        // bevy::render::render_resource::ShaderRef::Path(
        //     "../src/app/render/bordered_rect.wgsl".into(),
        // )
        BORDERED_RECT_SHADER_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline<Self>,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.label = Some("BorderedRect Pipeline".into());
        Ok(())
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct BorderedRectMaterial2d {
    #[uniform(0)]
    pub fill_color: LinearRgba,

    #[uniform(1)]
    pub border_color: LinearRgba,

    // 8 bits/0-255 for each of the four sides: N/E/S/W
    #[uniform(2)]
    pub border_opacities: u32,

    #[uniform(3)]
    pub border_width_px: f32,

    pub alpha_mode: AlphaMode,
}

impl Material2d for BorderedRectMaterial2d {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        BORDERED_RECT_2D_SHADER_HANDLE.into()
    }

    fn specialize(
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::sprite::Material2dKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.label = Some("BorderedRect Pipeline 2D".into());
        Ok(())
    }

    // fn alpha_mode(&self) -> AlphaMode {
    //     self.alpha_mode
    // }
}
