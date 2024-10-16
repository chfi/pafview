use std::ops::{Range, RangeInclusive};

use ultraviolet::{DMat3, DMat4, DVec2, DVec4, Mat3, Mat4, Vec2};

use crate::math_conv::*;

#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub view_center: DVec2, // view center in world coordinates
    pub view_size: DVec2,   // world coordinates

    pub canvas_offset: Vec2, // top left corner of canvas in screen points
    pub canvas_size: Vec2,   // pixels
}

impl Viewport {
    pub fn new(
        view_center: impl Into<[f64; 2]>,
        view_size: impl Into<[f64; 2]>,
        canvas_offset: impl Into<[f32; 2]>,
        canvas_size: impl Into<[f32; 2]>,
    ) -> Self {
        let view_center = view_center.as_duv();
        let view_size = view_size.as_duv();
        let canvas_size = canvas_size.as_uv();
        let canvas_offset = canvas_offset.as_uv();

        Self {
            view_center,
            view_size,
            canvas_size,
            canvas_offset,
        }
    }

    pub fn x_range(&self) -> std::ops::RangeInclusive<f64> {
        let halfwidth = self.view_size.x * 0.5;
        let min = self.view_center.x - halfwidth;
        let max = self.view_center.x + halfwidth;
        min..=max
    }

    pub fn y_range(&self) -> std::ops::RangeInclusive<f64> {
        let halfheight = self.view_size.y * 0.5;
        let min = self.view_center.y - halfheight;
        let max = self.view_center.y + halfheight;
        min..=max
    }

    pub fn translate_view(&mut self, world_delta: impl Into<[f64; 2]>) {
        self.view_center += world_delta.as_duv();
    }

    pub fn scale_view(&mut self, scale_mult: f64) {
        self.view_size *= scale_mult;
    }

    pub fn set_view_center(&mut self, new_center_world: impl Into<[f64; 2]>) {
        self.view_center = new_center_world.as_duv();
    }

    pub fn set_canvas_bounds(
        &mut self,
        canvas_offset: impl Into<[f32; 2]>,
        canvas_size: impl Into<[f32; 2]>,
    ) {
        let offset = canvas_offset.as_uv();
        let size = canvas_size.as_uv();

        if size.x <= 0.0 || size.y <= 0.0 {
            return;
        }

        let view_scale = self.view_size.x / self.canvas_size.x as f64;
        let canvas_aspect = size.y as f64 / size.x as f64;

        let new_width = view_scale * size.x as f64;
        let new_height = new_width * canvas_aspect;

        self.canvas_offset = offset;
        self.canvas_size = size;
        self.view_size = DVec2::new(new_width, new_height);
    }
}

impl Viewport {
    /// Transforms world coordinates to screen coordinates, mapping the top left
    /// of the `self.view_{..}` world rectangle to the top left of the screen canvas.
    pub fn world_screen_mat3(&self) -> Mat3 {
        let view_size = self.view_size.to_f32();
        let scale = Mat3::from_nonuniform_scale_homogeneous(Vec2::new(
            self.canvas_size.x / view_size.x,
            self.canvas_size.y / view_size.y,
        ));
        let translate = Mat3::from_translation(self.canvas_offset);
        let mut offset = -self.view_center.to_f32();

        offset.y -= view_size.y * 0.5;
        offset.x += view_size.x * 0.5;

        let center_translate = Mat3::from_translation(offset);
        let y_flip = Mat3::from_nonuniform_scale_homogeneous(Vec2::new(1.0, -1.0));

        y_flip * translate * scale * center_translate
    }

    /// Transforms world coordinates to screen coordinates, mapping the top left
    /// of the `self.view_{..}` world rectangle to the top left of the screen canvas.
    pub fn world_screen_dmat3(&self) -> DMat3 {
        let canvas_size = self.canvas_size.to_f64();
        let scale = DMat3::from_nonuniform_scale_homogeneous(DVec2::new(
            canvas_size.x / self.view_size.x,
            canvas_size.y / self.view_size.y,
        ));
        let translate = DMat3::from_translation(self.canvas_offset.to_f64());
        // let center_translate = DMat3::from_translation(-self.view_center + self.view_size * 0.5);

        // translate * scale * center_translate
        let mut offset = -self.view_center;

        offset.y -= self.view_size.y * 0.5;
        offset.x += self.view_size.x * 0.5;

        let center_translate = DMat3::from_translation(offset);
        let y_flip = DMat3::from_nonuniform_scale_homogeneous(DVec2::new(1.0, -1.0));

        y_flip * translate * scale * center_translate
    }

    /// Transforms screen coordinates to world coordinates, mapping the top left
    /// pixel of the screen canvas to the top left world point of the `self.view_{..}`
    /// rectangle.
    pub fn screen_world_mat3(&self) -> Mat3 {
        let view_size = self.view_size.to_f32();
        let scale_inv = Mat3::from_nonuniform_scale_homogeneous(Vec2::new(
            view_size.x / self.canvas_size.x,
            view_size.y / self.canvas_size.y,
        ));
        let translate_inv = Mat3::from_translation(-self.canvas_offset);

        let mut offset = self.view_center.to_f32();
        offset.y += view_size.y * 0.5;
        offset.x -= view_size.x * 0.5;

        let center_translate_inv = Mat3::from_translation(offset);
        let y_flip = Mat3::from_nonuniform_scale_homogeneous(Vec2::new(1.0, -1.0));

        center_translate_inv * scale_inv * translate_inv * y_flip
    }

    /// Transforms screen coordinates to world coordinates, mapping the top left
    /// pixel of the screen canvas to the top left world point of the `self.view_{..}`
    /// rectangle.
    pub fn screen_world_dmat3(&self) -> DMat3 {
        let canvas_size = self.canvas_size.to_f64();
        let scale_inv = DMat3::from_nonuniform_scale_homogeneous(DVec2::new(
            self.view_size.x / canvas_size.x,
            self.view_size.y / canvas_size.y,
        ));
        let translate_inv = DMat3::from_translation(-self.canvas_offset.to_f64());

        let mut offset = self.view_center;
        offset.y += self.view_size.y * 0.5;
        offset.x -= self.view_size.x * 0.5;

        let center_translate_inv = DMat3::from_translation(offset);
        let y_flip = DMat3::from_nonuniform_scale_homogeneous(DVec2::new(1.0, -1.0));

        center_translate_inv * scale_inv * translate_inv * y_flip
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
// #[deprecated]
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

    pub fn x_range_usize(&self) -> std::ops::Range<usize> {
        let l = self.x_min.floor() as usize;
        let r = self.x_max.ceil() as usize;
        l..r
    }
    pub fn y_range_usize(&self) -> std::ops::Range<usize> {
        let u = self.y_min.floor() as usize;
        let d = self.y_max.ceil() as usize;
        u..d
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
        screen_dims: impl Into<[f32; 2]>,
        screen_pt: impl Into<Vec2>,
    ) -> DVec2 {
        let [sw, sh] = screen_dims.into();
        let sw = sw as f64;
        let sh = sh as f64;

        let sp = screen_pt.into();

        let np = DVec2::new(sp.x as f64 / sw, 1.0 - (sp.y as f64 / sh));

        np * self.size()
    }

    pub fn map_view_to_screen(
        &self,
        screen_dims: impl Into<[f32; 2]>,
        view_pt: impl Into<DVec2>,
    ) -> Vec2 {
        let [sw, sh] = screen_dims.into();
        let sdims = DVec2::new(sw as f64, sh as f64);

        let vp = view_pt.into();

        let nx = vp.x / self.width();
        let ny = vp.y / self.height();

        let np = DVec2::new(nx, ny);

        let sp = np * sdims;
        Vec2::new(sp.x as f32, sh as f32 - sp.y as f32)
    }

    pub fn map_world_to_screen(
        &self,
        screen_dims: impl Into<[f32; 2]>,
        world_pt: impl Into<DVec2>,
    ) -> Vec2 {
        let view_pt = self.map_world_to_view(world_pt);
        self.map_view_to_screen(screen_dims, view_pt)
    }

    pub fn map_screen_to_world(
        &self,
        screen_dims: impl Into<[f32; 2]>,
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

    pub fn bp_per_pixel(&self, canvas_width: u32) -> f64 {
        self.width() / canvas_width as f64
    }

    pub fn pixels_per_bp(&self, canvas_width: u32) -> f64 {
        canvas_width as f64 / self.width()
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

    /// Expands/contracts the view by a factor of `s`, keeping the
    /// point corresponding to `t` fixed in the view.
    ///
    /// Both `t.x` and `t.y` should be in `[0, 1]`, if `s` > 1.0, the
    /// view is zoomed out, if `s` < 1.0, it is zoomed in.
    pub fn zoom_with_focus(&mut self, t: impl Into<DVec2>, s: f64) {
        // log::warn!("zoomin");
        let l = self.x_min;
        let r = self.x_max;
        let u = self.y_min;
        let d = self.y_max;

        let t = t.into();

        let (l_, r_) = expand_with_fixpoint(l, r, t.x, s);
        let (u_, d_) = expand_with_fixpoint(u, d, 1.0 - t.y, s);

        self.x_min = l_;
        self.x_max = r_;
        self.y_min = u_;
        self.y_max = d_;
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

    pub fn fit_ranges_in_view_f64(
        &self,
        x_range: Option<RangeInclusive<f64>>,
        y_range: Option<RangeInclusive<f64>>,
    ) -> Self {
        let aspect_ratio = self.width() / self.height();
        self.fit_ranges_in_view_with_aspect_f64(aspect_ratio, x_range, y_range)
    }

    pub fn fit_ranges_in_view_with_aspect_f64(
        &self,
        aspect_ratio: f64,
        x_range: Option<RangeInclusive<f64>>,
        y_range: Option<RangeInclusive<f64>>,
    ) -> Self {
        let (x_min, x_max) = x_range
            .map(|r| (*r.start(), *r.end()))
            .unwrap_or_else(|| (self.x_min, self.x_max));
        let (y_min, y_max) = y_range
            .map(|r| (*r.start(), *r.end()))
            .unwrap_or_else(|| (self.y_min, self.y_max));

        let size = DVec2::new(x_max - x_min, y_max - y_min);
        let center = DVec2::new(x_min, y_min) + size * 0.5;

        let (_center, new_size) =
            calculate_covering_rectangle(aspect_ratio, center.into(), size.into());
        let new_size = DVec2::from(new_size);

        let min = center - new_size * 0.5;
        let max = center + new_size * 0.5;

        Self {
            x_min: min.x,
            x_max: max.x,
            y_min: min.y,
            y_max: max.y,
        }
    }

    pub fn fit_ranges_in_view_with_aspect(
        &self,
        aspect_ratio: f64,
        x_range: Option<Range<u64>>,
        y_range: Option<Range<u64>>,
    ) -> Self {
        let map_range = |range: Range<u64>| (range.start as f64)..=(range.end as f64);
        let x_range = x_range.map(map_range);
        let y_range = y_range.map(map_range);

        self.fit_ranges_in_view_with_aspect_f64(aspect_ratio, x_range, y_range)
    }

    pub fn resize_for_window_size(
        &self,
        old_window_size: impl Into<[u32; 2]>,
        new_window_size: impl Into<[u32; 2]>,
    ) -> Self {
        let [_ow, oh] = old_window_size.into();
        let [nw, nh] = new_window_size.into();

        let new_aspect = nw as f64 / nh as f64;

        let old_h_scale = self.height() / oh as f64;
        let new_height = nh as f64 * old_h_scale;
        let new_width = new_height * new_aspect;

        let center = self.center();

        Self {
            x_min: center.x - new_width * 0.5,
            x_max: center.x + new_width * 0.5,
            y_min: center.y - new_height * 0.5,
            y_max: center.y + new_height * 0.5,
        }
    }

    pub fn apply_limits(&mut self, window_size: impl Into<[u32; 2]>) {
        let [w_width, w_height] = window_size.into();

        let pixels_per_bp = w_width as f64 / self.width();
        if pixels_per_bp < 32.0 {
            return;
        }

        let pixels_per_bp = 32.0;
        let bp_per_pixel = 1.0 / pixels_per_bp;

        let width = bp_per_pixel * w_width as f64;
        let height = bp_per_pixel * w_height as f64;
        let center = self.center();

        self.x_min = center.x - width * 0.5;
        self.y_min = center.y - height * 0.5;
        self.x_max = center.x + width * 0.5;
        self.y_max = center.y + height * 0.5;
    }
}

fn calculate_covering_rectangle(
    aspect_ratio: f64,
    center: [f64; 2],
    size: [f64; 2],
) -> ([f64; 2], [f64; 2]) {
    let (w0, h0) = (size[0], size[1]); // Width and height of the input rectangle

    // Calculate new width and height assuming we adjust the width to fit the aspect ratio first
    let mut w1 = aspect_ratio * h0;
    let mut h1 = h0;

    // Check if the new size covers the input rectangle completely, adjust if necessary
    if w1 < w0 {
        // If new width is smaller than original width, adjust height to maintain aspect ratio
        h1 = w0 / aspect_ratio;
        w1 = w0;
    }

    // If initial width adjustment was enough (or too much), check if we need to adjust height instead
    if h1 < h0 {
        // This case might not be necessary after the above adjustment, but it's here for completeness
        w1 = aspect_ratio * h0;
        h1 = h0;
    }

    // The center remains the same
    (center, [w1, h1])
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
            let screen_dims = [1920.0, 1080f32];
            let s_pt = Vec2::new(s_x, s_y);
            let v_pt = view.map_screen_to_view(screen_dims, s_pt);

            let s_pt_ = view.map_view_to_screen(screen_dims, v_pt);
            let diff = s_pt_ - s_pt.into();
            let x_eq = approx_eq!(f32, diff.x / 1920.0, 0.0);
            let y_eq = approx_eq!(f32, diff.y / 1080.0, 0.0);
            prop_assert!(x_eq && y_eq);
        }

        #[test]
        fn view_screen_isomorphic((view, v_pt) in view_strategy()
                                  .prop_perturb(point_in_view)
        ) {
            let screen_dims = [1920.0, 1080f32];

            let s_pt = view.map_view_to_screen(screen_dims, v_pt);
            let v_pt_ = view.map_screen_to_view(screen_dims, s_pt);
            let diff = v_pt - v_pt_;

            let eps =  0.01;
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
            prop_assert!(w_pt.x.round() == w_pt_.x.round()
                         && w_pt.y.round() == w_pt_.y.round(),
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

fn expand_with_fixpoint(a: f64, b: f64, t: f64, s: f64) -> (f64, f64) {
    let l = b - a;
    let x = a + t * l;

    let p_a = t;
    let p_b = 1.0 - t;

    let l_ = l * s;

    /* // NB: this should probably be handled elsewhere
        // just so things don't implode
        if l_ < 1.0 {
            l_ = 1.0;
        }
    */

    let x_a = p_a * l_;
    let x_b = p_b * l_;

    let a_ = x - x_a;
    let b_ = x + x_b;

    (a_, b_)
}
