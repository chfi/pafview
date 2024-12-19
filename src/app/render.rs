use bevy::prelude::*;

pub mod base_level;
pub mod bordered_rect;
pub mod sampled_lines;

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

#[derive(Clone, Copy, PartialEq, Reflect)]
struct RenderParams {
    view: crate::view::View,
    canvas_size: UVec2,
}

impl RenderParams {
    fn scale(&self) -> f64 {
        self.view.width() / self.canvas_size.x as f64
    }
}
