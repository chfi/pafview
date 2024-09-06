// pub mod batch;
pub mod color;
pub mod exact;

pub fn create_multisampled_framebuffer(
    device: &wgpu::Device,
    dims: [u32; 2],
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::TextureView {
    let [width, height] = dims;
    let multisampled_texture_extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        size: multisampled_texture_extent,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };

    device
        .create_texture(multisampled_frame_descriptor)
        .create_view(&wgpu::TextureViewDescriptor::default())
}
