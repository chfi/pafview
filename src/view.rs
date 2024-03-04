use ultraviolet::{DMat4, DVec2, DVec4, Mat4, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct View {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl View {
    pub fn map_screen_to_view(&self, screen_dims: [u32; 2], screen_pt: impl Into<Vec2>) -> DVec2 {
        let [sw, sh] = screen_dims;
        let sw = sw as f64;
        let sh = sh as f64;

        let sp = screen_pt.into();

        let np = DVec2::new(sp.x as f64 / sw, sp.y as f64 / sh);

        let tleft = DVec2::new(self.x_min, self.y_min);

        tleft + np * self.size()
    }

    pub fn map_view_to_screen(&self, screen_dims: [u32; 2], view_pt: impl Into<DVec2>) -> Vec2 {
        let [sw, sh] = screen_dims;
        let sdims = DVec2::new(sw as f64, sh as f64);

        let vp = view_pt.into();

        let nx = (vp.x - self.x_min) / self.width();
        let ny = (vp.y - self.y_min) / self.height();

        let np = DVec2::new(nx, ny);

        let sp = np * sdims;
        Vec2::new(sp.x as f32, sp.y as f32)
    }

    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    pub fn size(&self) -> DVec2 {
        DVec2::new(self.width(), self.height())
    }

    pub fn center(&self) -> DVec2 {
        DVec2::new(
            self.x_min + self.width() * 0.5,
            self.y_min + self.height() * 0.5,
        )
    }

    pub fn to_dmat4(&self) -> DMat4 {
        let right = self.x_max;
        let left = self.x_min;
        let top = self.y_min;
        let bottom = self.y_max;
        let near = 0.1;
        let far = 10.0;
        let rml = right - left;
        let rpl = right + left;
        let tmb = top - bottom;
        let tpb = top + bottom;
        let fmn = far - near;
        DMat4::new(
            DVec4::new(2.0 / rml, 0.0, 0.0, 0.0),
            DVec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
            DVec4::new(0.0, 0.0, -1.0 / fmn, 0.0),
            DVec4::new(-(rpl / rml), -(tpb / tmb), -(near / fmn), 1.0),
        )
    }

    pub fn to_mat4(&self) -> Mat4 {
        ultraviolet::projection::orthographic_wgpu_dx(
            self.x_min as f32,
            self.x_max as f32,
            self.y_min as f32,
            self.y_max as f32,
            0.1,
            10.0,
        )
    }

    pub fn translate(&mut self, dx: f64, dy: f64) {
        self.x_min += dx;
        self.x_max += dx;
        self.y_min += dy;
        self.y_max += dy;
    }

    pub fn scale_around_center(&mut self, s: f64) {
        let x_len = self.x_max - self.x_min;
        let y_len = self.y_max - self.y_min;

        let x0 = self.x_min + x_len * 0.5;
        let y0 = self.y_min + y_len * 0.5;

        let x_hlen = (s * x_len) * 0.5;
        let y_hlen = (s * y_len) * 0.5;

        self.x_min = x0 - x_hlen;
        self.x_max = x0 + x_hlen;
        self.y_min = y0 - y_hlen;
        self.y_max = y0 + y_hlen;
    }

    /// Returns a `View` that contains all of the given `view` while adhering to
    /// the aspect ratio given by `viewport_dims`
    pub fn fit_view_into_viewport(view: View, viewport_dims: [u32; 2]) -> Self {
        let [sw, sh] = viewport_dims;

        let screen_aspect = (sw as f64) / (sh as f64);
        let view_aspect = view.width() / view.height();

        todo!();
    }
}
