use ultraviolet::{DVec2, Mat4, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct View {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl View {
    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    pub fn center(&self) -> DVec2 {
        DVec2::new(
            self.x_min + self.width() * 0.5,
            self.y_min + self.height() * 0.5,
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
