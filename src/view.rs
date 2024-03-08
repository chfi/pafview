use ultraviolet::{DMat4, DVec2, DVec4, Mat4, Vec2, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct View {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl View {
    pub fn x_range(&self) -> std::ops::RangeInclusive<f64> {
        self.x_min..=self.x_max
    }

    pub fn y_range(&self) -> std::ops::RangeInclusive<f64> {
        self.y_min..=self.y_max
    }

    pub fn map_world_to_view(&self, world_pt: impl Into<DVec2>) -> DVec2 {
        let wp = world_pt.into();
        let tleft = DVec2::new(self.x_min, self.y_min);

        let vp = wp - tleft;
        vp
    }

    pub fn map_view_to_world(&self, view_pt: impl Into<DVec2>) -> DVec2 {
        let vp = view_pt.into();
        let tleft = DVec2::new(self.x_min, self.y_min);

        let wp = vp + tleft;
        wp
    }

    pub fn map_screen_to_view(
        &self,
        screen_dims: impl Into<[u32; 2]>,
        screen_pt: impl Into<Vec2>,
    ) -> DVec2 {
        let [sw, sh] = screen_dims.into();
        let sw = sw as f64;
        let sh = sh as f64;

        let sp = screen_pt.into();

        let np = DVec2::new(sp.x as f64 / sw, sp.y as f64 / sh);

        let tleft = DVec2::new(self.x_min, self.y_min);

        tleft + np * self.size()
    }

    pub fn map_view_to_screen(
        &self,
        screen_dims: impl Into<[u32; 2]>,
        view_pt: impl Into<DVec2>,
    ) -> Vec2 {
        let [sw, sh] = screen_dims.into();
        let sdims = DVec2::new(sw as f64, sh as f64);

        let vp = view_pt.into();

        let nx = (vp.x - self.x_min) / self.width();
        let ny = (vp.y - self.y_min) / self.height();

        let np = DVec2::new(nx, ny);

        let sp = np * sdims;
        Vec2::new(sp.x as f32, sp.y as f32)
    }

    pub fn map_world_to_screen(
        &self,
        screen_dims: impl Into<[u32; 2]>,
        world_pt: impl Into<DVec2>,
    ) -> Vec2 {
        let view_pt = self.map_world_to_view(world_pt);
        self.map_view_to_screen(screen_dims, view_pt)
    }

    pub fn map_screen_to_world(
        &self,
        screen_dims: impl Into<[u32; 2]>,
        screen_pt: impl Into<Vec2>,
    ) -> DVec2 {
        let view_pt = self.map_screen_to_view(screen_dims, screen_pt);
        self.map_view_to_world(view_pt)
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

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use proptest::{prelude::*, test_runner::TestRng};

    use super::*;

    // TODO generate/proptest limits
    const AXIS_LIMIT: f64 = 100_000_000_000.0;

    prop_compose! {
        // generate views within some set max
        fn view_limit_strategy
            (x_limit: f64, y_limit: f64)
            (x_min_ in 0.0..x_limit,
             y_min_ in 0.0..y_limit)
            (
                x_min in Just(x_min_),
                y_min in Just(y_min_),
                x_max in (x_min_ + 1.0)..x_limit,
                y_max in (y_min_ + 1.0)..y_limit)
             -> View {
                View { x_min, y_min, x_max, y_max }
            }
    }

    prop_compose! {
        fn view_strategy()(view in view_limit_strategy(AXIS_LIMIT, AXIS_LIMIT)
        ) -> View {
            view
        }
    }

    prop_compose! {
        fn prop_view_point(view: View)(vx in 0f64..view.width(), vy in 0f64..view.height()) -> DVec2 {
            DVec2::new(vx, vy)
        }
    }

    fn point_in_view(view: View, mut rng: TestRng) -> (View, DVec2) {
        let vx = rng.gen_range(view.x_min..view.x_max);
        let vy = rng.gen_range(view.y_min..view.y_max);
        (view, DVec2::new(vx, vy))
    }

    proptest! {
        #[test]
        fn screen_view_isomorphic(view in view_strategy(),
                                  s_x in 0f32..1920.0,
                                  s_y in 0f32..1080.0) {
            let screen_dims = [1920, 1080u32];
            let s_pt = Vec2::new(s_x, s_y);
            let v_pt = view.map_screen_to_view(screen_dims, s_pt);

            let s_pt_ = view.map_view_to_screen(screen_dims, v_pt);
            let diff = s_pt_ - s_pt.into();
            prop_assert_eq!(diff, Vec2::new(0.0, 0.0));
        }

        #[test]
        fn view_screen_isomorphic((view, v_pt) in view_strategy()
                                  .prop_perturb(point_in_view)
        ) {
            let screen_dims = [1920, 1080u32];

            let s_pt = view.map_view_to_screen(screen_dims, v_pt);
            let v_pt_ = view.map_screen_to_view(screen_dims, s_pt);
            let diff = v_pt - v_pt_;

            let eps =  std::f32::EPSILON as f64;
            let x_eq = approx_eq!(f64, diff.x / view.width(), 0.0, epsilon = eps);
            let y_eq = approx_eq!(f64, diff.y / view.height(), 0.0, epsilon = eps);
            prop_assert!(x_eq && y_eq);
        }
    }

    proptest! {
        #[test]
        fn world_view_isomorphic(view in view_strategy(),
                                 wx in 0f64..AXIS_LIMIT,
                                 wy in 0f64..AXIS_LIMIT,
        ) {
            let w_pt = DVec2::new(wx, wy);
            let v_pt = view.map_world_to_view(w_pt);
            let w_pt_ = view.map_view_to_world(v_pt);
            // .. close enough
            let x_eps = 1_000_000f64.min(view.width()) / view.width();
            let y_eps = 1_000_000f64.min(view.height()) / view.height();
            prop_assert!(approx_eq!(f64, w_pt.x, w_pt_.x, epsilon = x_eps)
                         && approx_eq!(f64, w_pt.y, w_pt_.y, epsilon = y_eps),
                         "({}, {}) != ({}, {})",
                         w_pt.x, w_pt.y,
                         w_pt_.x, w_pt_.y,
            )
        }

        #[test]
        fn view_world_isomorphic((view, v_pt) in view_strategy()
                                 .prop_perturb(point_in_view)
        ) {
            let w_pt = view.map_view_to_world(v_pt);
            let v_pt_ = view.map_world_to_view(w_pt);
            prop_assert!(approx_eq!(f64, v_pt.x, v_pt_.x)
                         && approx_eq!(f64, v_pt.y, v_pt_.y),
                         "({}, {}) != ({}, {})",
                         v_pt.x, v_pt.y,
                         v_pt_.x, v_pt_.y,
            )
        }
    }
}
