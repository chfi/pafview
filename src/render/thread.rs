use std::sync::Arc;

use crossbeam::channel::{self, Receiver, RecvError, Sender};
use wgpu::{Device, Queue};

/*
pub struct PafRendererHandle {
    //
    send: Sender<()>,
    recv: Receiver<()>,
}

#[derive(Clone)]
enum RenderCommand {
    Draw { viewport: crate::view::Viewport },
}

enum RenderMessage {}

fn start_renderer_thread(
    device: Arc<Device>,
    queue: Arc<Queue>,
    app: &crate::PafViewerApp,
    color_format: wgpu::TextureFormat,
    sample_count: u32,
) -> PafRendererHandle {
    let alignment_grid = app.alignment_grid.clone();
    let alignments = app.alignments.clone();

    let thread = std::thread::spawn(move || {
        let mut paf_renderer = super::PafRenderer::new(
            &device,
            color_format,
            sample_count,
            // match_buffer,
            // match_color_buffer,
            // match_instances,
        );

        let match_draw_data = super::batch::MatchDrawBatchData::from_alignments(
            &device,
            &paf_renderer.line_pipeline.bind_group_layout_1,
            &alignment_grid,
            &alignments,
        );
    });
    // need the grid too, somehow,

    //
    todo!();
}

*/
