use bevy::{asset::load_internal_asset, prelude::*, render::render_resource::AsBindGroup};

pub struct BorderedRectRenderPlugin;

pub const BORDERED_RECT_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(66283402603114632559453785259996878691);

impl Plugin for BorderedRectRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            BORDERED_RECT_SHADER_HANDLE,
            "bordered_rect.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(MaterialPlugin::<BorderedRectMaterial>::default())
            .add_systems(Startup, debug_print_shader);
    }
}

fn debug_print_shader(shaders: Res<Assets<Shader>>, mut is_done: Local<bool>) {
    if *is_done {
        return;
    }

    if let Some(shader) = shaders.get(&BORDERED_RECT_SHADER_HANDLE) {
        *is_done = true;
        //

        match &shader.source {
            bevy::render::render_resource::Source::Wgsl(wgsl) => {
                println!("loaded wgsl");
                println!("{wgsl}");
            }
            bevy::render::render_resource::Source::Glsl(_glsl, _stage) => {
                println!("loaded glsl");
            }
            bevy::render::render_resource::Source::SpirV(_spirv) => {
                println!("loaded spirv");
            }
        }
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

impl Material for BorderedRectMaterial {
    fn fragment_shader() -> bevy::render::render_resource::ShaderRef {
        BORDERED_RECT_SHADER_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }
}
