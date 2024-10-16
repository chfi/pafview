pub struct PixelBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<egui::Color32>,
}

impl PixelBuffer {
    pub fn sample_into(&self, dst: &mut PixelBuffer, dst_offset: [f32; 2], dst_scale: [f32; 2]) {
        // Calculate the bounds in the dst buffer that correspond to the src bounds
        let start_x = ((0.0 - dst_offset[0]) * dst_scale[0].max(0.0).ceil()) as usize;
        let start_y = ((0.0 - dst_offset[1]) * dst_scale[1].max(0.0).ceil()) as usize;
        let end_x =
            (((self.width as f32 - dst_offset[0]) * dst_scale[0]).min(dst.width as f32)) as usize;
        let end_y =
            (((self.height as f32 - dst_offset[1]) * dst_scale[1]).min(dst.height as f32)) as usize;

        for y in start_y..end_y {
            for x in start_x..end_x {
                let src_x = (x as f32 - dst_offset[0]) / dst_scale[0];
                let src_y = (y as f32 - dst_offset[1]) / dst_scale[1];

                if src_x >= 0.0
                    && src_x < self.width as f32
                    && src_y >= 0.0
                    && src_y < self.height as f32
                {
                    dst.pixels[y * dst.width as usize + x] =
                        bilinear_interpolate(self, src_x, src_y).into();
                    // } else {
                    //     dst.pixels[y * dst.width as usize + x] = egui::Color32::TRANSPARENT;
                }
            }
        }
    }
}
impl PixelBuffer {
    // nearest neighbor

    pub fn sample_subimage_nn_into_with(
        &self,
        dst: &mut PixelBuffer,
        dst_offset: [f32; 2],
        dst_size: [f32; 2],
        src_offset: [u32; 2],
        src_size: [u32; 2],
        color_fn: impl Fn(egui::Color32, egui::Color32) -> egui::Color32,
    ) {
        let [x0, y0] = dst_offset;
        let [dw, dh] = dst_size;

        let dst_x0 = x0.floor();
        let dst_y0 = y0.floor();

        let dst_x1 = (x0 + dw).ceil();
        let dst_y1 = (y0 + dh).ceil();

        let dst_x_range = (dst_x0 as usize)..(dst_x1 as usize);
        let dst_y_range = (dst_y0 as usize)..(dst_y1 as usize);

        let src_w = src_size[0] as f32;
        let src_h = src_size[1] as f32;

        for (y_i, dst_y) in dst_y_range.enumerate() {
            let y_t = (y_i as f32) / (dst_y1 - dst_y0);
            let src_y = src_offset[1] as usize + (y_t * src_h).round() as usize;

            for (x_i, dst_x) in dst_x_range.clone().enumerate() {
                let x_t = (x_i as f32) / (dst_x1 - dst_x0);
                let src_x = src_offset[0] as usize + (x_t * src_w).round() as usize;

                let src_i = self.width as usize * src_y + src_x;
                let dst_i = dst.width as usize * dst_y + dst_x;

                let src_px = if let Some(src_px) = self.pixels.get(src_i) {
                    *src_px
                } else {
                    egui::Color32::TRANSPARENT
                };

                if let Some(dst_px) = dst.pixels.get_mut(dst_i) {
                    let above = src_px;
                    let below = *dst_px;
                    *dst_px = color_fn(below, above);
                }
            }
        }
    }

    pub fn sample_subimage_nn_into(
        &self,
        dst: &mut PixelBuffer,
        dst_offset: [f32; 2],
        dst_size: [f32; 2],
        src_offset: [u32; 2],
        src_size: [u32; 2],
    ) {
        self.sample_subimage_nn_into_with(dst, dst_offset, dst_size, src_offset, src_size, |_, c| c)
    }

    pub fn sample_subimage_into_bilerp(
        &self,
        dst: &mut PixelBuffer,
        dst_offset: [f32; 2],
        dst_size: [f32; 2],
        src_offset: [u32; 2],
        src_size: [u32; 2],
    ) {
        // Calculate scale factors based on the size ratios
        let scale_x = dst_size[0] / src_size[0] as f32;
        let scale_y = dst_size[1] / src_size[1] as f32;

        // Calculate the area in the destination buffer affected by the operation
        let start_x = dst_offset[0].ceil() as usize;
        let start_y = dst_offset[1].ceil() as usize;
        let end_x = ((dst_offset[0] + dst_size[0]).min(dst.width as f32)).floor() as usize;
        let end_y = ((dst_offset[1] + dst_size[1]).min(dst.height as f32)).floor() as usize;

        println!("dst: ({dst_offset:?}, {dst_size:?}), src: ({src_offset:?}, {src_size:?}) => \t x: {:?}, y: {:?}",
                 start_y..end_y,
                 start_x..end_x,
                 );

        // Iterate over the calculated destination bounds
        for y in start_y..end_y {
            for x in start_x..end_x {
                // Calculate corresponding source coordinates
                let src_x = (x as f32 - dst_offset[0]) / scale_x + src_offset[0] as f32;
                let src_y = (y as f32 - dst_offset[1]) / scale_y + src_offset[1] as f32;

                // Only proceed if within bounds of the source subimage
                if src_x >= src_offset[0] as f32
                    && src_x < (src_offset[0] + src_size[0]) as f32
                    && src_y >= src_offset[1] as f32
                    && src_y < (src_offset[1] + src_size[1]) as f32
                {
                    dst.pixels[y * dst.width as usize + x] =
                        bilinear_interpolate_offset(self, src_x, src_y, src_offset, src_size)
                            .into();
                    // } else {
                    //     dst.pixels[y * dst.width as usize + x] = egui::Color32::TRANSPARENT;
                } else {
                    if let Some(v) = dst.pixels.get_mut(y * dst.width as usize + x) {
                        *v = egui::Color32::RED;
                        // .into();
                    }
                }
                /*
                {

                    dst.pixels[y * dst.width + x] =
                        bilinear_interpolate(self, src_x, src_y, src_offset, src_size);
                } else {
                    dst.pixels[y * dst.width + x] = [0.0, 0.0, 0.0, 0.0]; // Use a background color or transparency
                }
                    */
            }
        }
    }
}

impl PixelBuffer {
    pub fn new_color(width: u32, height: u32, color: egui::Color32) -> Self {
        Self {
            width,
            height,
            pixels: vec![color; (width * height) as usize],
        }
    }

    pub fn new(width: u32, height: u32) -> Self {
        Self::new_color(width, height, egui::Color32::TRANSPARENT)
    }

    pub fn get(&self, x: usize, y: usize) -> Option<egui::Color32> {
        if x >= self.width as usize || y >= self.height as usize {
            None
        } else {
            self.pixels.get(x + y * self.width as usize).copied()
        }
    }
}

impl PixelBuffer {
    pub fn write_png_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let pixels: &[u8] = bytemuck::cast_slice(&self.pixels);

        lodepng::encode32_file(path, pixels, self.width as usize, self.height as usize)?;

        Ok(())
    }
}

fn bilinear_interpolate(src: &PixelBuffer, x: f32, y: f32) -> egui::Rgba {
    bilinear_interpolate_offset(src, x, y, [0, 0], [src.width, src.height])
}

fn bilinear_interpolate_offset(
    src: &PixelBuffer,
    x: f32,
    y: f32,
    src_offset: [u32; 2],
    src_size: [u32; 2],
) -> egui::Rgba {
    let x_floor = x.floor() as usize;
    let y_floor = y.floor() as usize;
    let x_ceil = x.ceil() as usize;
    let y_ceil = y.ceil() as usize;

    // Adjust coordinates for src_offset
    let x_floor = x_floor + src_offset[0] as usize;
    let x_ceil = x_ceil + src_offset[0] as usize;
    let y_floor = y_floor + src_offset[1] as usize;
    let y_ceil = y_ceil + src_offset[1] as usize;

    // Clamp coordinates to src_size
    let x_floor = x_floor.min(src_offset[0] as usize + src_size[0] as usize - 1);
    let x_ceil = x_ceil.min(src_offset[0] as usize + src_size[0] as usize - 1);
    let y_floor = y_floor.min(src_offset[1] as usize + src_size[1] as usize - 1);
    let y_ceil = y_ceil.min(src_offset[1] as usize + src_size[1] as usize - 1);

    // Bilinear interpolation logic (unchanged)
    // ...
    let s_width = src.width as usize;
    let s_height = src.height as usize;

    let t = x - x.floor();
    let u = y - y.floor();
    // let t = x - x_floor as f32;
    // let u = y - y_floor as f32;

    if x_floor >= s_width || y_floor >= s_height || x_ceil >= s_width || y_ceil >= s_height {
        return egui::Color32::RED.into();
    }

    let p00: egui::Rgba = src.pixels[y_floor * s_width + x_floor].into();
    let p10: egui::Rgba = src.pixels[y_floor * s_width + x_ceil].into();
    let p01: egui::Rgba = src.pixels[y_ceil * s_width + x_floor].into();
    let p11: egui::Rgba = src.pixels[y_ceil * s_width + x_ceil].into();

    // Interpolate rows
    let p0 = [
        p00[0] + t * (p10[0] - p00[0]),
        p00[1] + t * (p10[1] - p00[1]),
        p00[2] + t * (p10[2] - p00[2]),
        p00[3] + t * (p10[3] - p00[3]),
    ];
    let p1 = [
        p01[0] + t * (p11[0] - p01[0]),
        p01[1] + t * (p11[1] - p01[1]),
        p01[2] + t * (p11[2] - p01[2]),
        p01[3] + t * (p11[3] - p01[3]),
    ];

    // let r =

    // Interpolate columns
    egui::Rgba::from_rgba_premultiplied(
        p0[0] + u * (p1[0] - p0[0]),
        p0[1] + u * (p1[1] - p0[1]),
        p0[2] + u * (p1[2] - p0[2]),
        p0[3] + u * (p1[3] - p0[3]),
    )
}

pub(crate) fn create_test_pattern_buffer(width: u32, height: u32) -> PixelBuffer {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x % 10) as f32 / 10.0;
            let g = (y % 10) as f32 / 10.0;

            let color = egui::Rgba::from_rgba_premultiplied(r, g, 0.5, 1.0);
            pixels.push(color.into());
        }
    }
    PixelBuffer {
        width,
        height,
        pixels,
    }
}

// Helper function to create a test image buffer with identifiable patterns
pub(crate) fn create_test_buffer(width: u32, height: u32) -> PixelBuffer {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            // Create a simple gradient pattern
            let color = egui::Rgba::from_rgba_premultiplied(
                x as f32 / width as f32,
                y as f32 / height as f32,
                0.5,
                1.0,
            );
            pixels.push(color.into());
        }
    }
    PixelBuffer {
        width,
        height,
        pixels,
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::hash::Hash;

    use ultraviolet::DVec2;

    /*
    #[test]
    fn test_exact_copy() {
        let src = create_test_buffer(10, 10);
        let mut dst = create_test_buffer(10, 10); // Destination buffer of the same size

        src.sample_subimage_into(&mut dst, [0.0, 0.0], [10.0, 10.0], [0, 0], [10, 10]);

        for (src_pixel, dst_pixel) in src.pixels.iter().zip(dst.pixels.iter()) {
            assert_eq!(
                src_pixel, dst_pixel,
                "Pixels must be identical for exact copy"
            );
        }
    }

    #[test]
    fn test_scaling_down() {
        let src = create_test_buffer(10, 10);
        let mut dst = create_test_buffer(5, 5); // Smaller destination buffer

        src.sample_subimage_into(&mut dst, [0.0, 0.0], [5.0, 5.0], [0, 0], [10, 10]);

        // Simple check: corners of dst should match corners of src
        assert_eq!(dst.pixels[0], src.pixels[0], "Top-left corner should match");
        assert_eq!(
            dst.pixels[4], src.pixels[9],
            "Top-right corner should match"
        );
        assert_eq!(
            dst.pixels[20], src.pixels[90],
            "Bottom-left corner should match"
        );
        assert_eq!(
            dst.pixels[24], src.pixels[99],
            "Bottom-right corner should match"
        );
    }

    #[test]
    fn test_subimage_extraction() {
        let src = create_test_buffer(10, 10);
        let mut dst = create_test_buffer(5, 5); // Destination buffer for subimage

        // Extract a central 5x5 block from src
        src.sample_subimage_into(&mut dst, [0.0, 0.0], [5.0, 5.0], [3, 3], [5, 5]);

        // Check if the center of src is now the entirety of dst
        for (y, dst_pixel) in dst.pixels.iter().enumerate() {
            let src_pixel = src.pixels[(y + 3) * 10 + 3]; // Corresponding src pixel
            assert_eq!(
                dst_pixel, &src_pixel,
                "Pixels should match extracted subimage"
            );
        }
    }

    #[test]
    fn test_edge_clipping() {
        let src = create_test_buffer(10, 10);
        let mut dst = create_test_buffer(5, 5); // Smaller buffer for testing edge clipping

        // Attempt to sample beyond the bounds of src
        src.sample_subimage_into(&mut dst, [0.0, 0.0], [5.0, 5.0], [8, 8], [10, 10]);

        // Check if the out-of-bounds areas are set to a default clear color
        for dst_pixel in dst.pixels.iter() {
            assert_eq!(
                dst_pixel.to_array(),
                [0, 0, 0, 0],
                "Out-of-bounds pixels should be clear"
            );
        }
    }

    #[allow(dead_code)]
    fn debug_print_pixel_buffer(buf: &PixelBuffer) {
        let h_border = (0..buf.width + 2).map(|_| '-').collect::<String>();
        println!("{h_border}");
        for row in 0..buf.height {
            print!("|");
            for col in 0..buf.width {
                let ix = (col + row * buf.width) as usize;

                let px = &buf.pixels[ix];

                if !px.is_opaque() {
                    print!(" ");
                } else if *px == egui::Color32::RED {
                    print!("R");
                } else if *px == egui::Color32::BLUE {
                    print!("B");
                } else if *px == egui::Color32::GREEN {
                    print!("G");
                } else {
                    print!(" ");
                }
            }
            println!("|");
        }

        println!("{h_border}");
    }
    */
}
