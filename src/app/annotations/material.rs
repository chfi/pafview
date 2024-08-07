use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::sprite::Material2d;

pub(super) struct AnnotationMaterialsPlugin;

impl Plugin for AnnotationMaterialsPlugin {
    fn build(&self, app: &mut App) {
        todo!()
    }
}

#[derive(ShaderType, Reflect, Debug, Clone)]
pub(super) struct ColoredRegionUniform {
    fill_alpha: f32,
    border_alpha: f32,
    border_width: f32,
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub(super) struct ColoredRegionMaterial {
    #[uniform(100)]
    config: ColoredRegionUniform,
}

impl Material2d for ColoredRegionMaterial {
    // TODO finish & use `shaders/colored_region.wgsl`
}
