pub trait ConvertVec2 {
    fn as_uv(self) -> ultraviolet::Vec2;
    fn as_epos2(self) -> egui::Pos2;
    fn as_evec2(self) -> egui::Vec2;
    fn as_na(self) -> nalgebra::Vector2<f32>;
}

impl<Via: Into<[f32; 2]>> ConvertVec2 for Via {
    fn as_uv(self) -> ultraviolet::Vec2 {
        self.into().into()
    }
    fn as_epos2(self) -> egui::Pos2 {
        self.into().into()
    }
    fn as_evec2(self) -> egui::Vec2 {
        self.into().into()
    }
    fn as_na(self) -> nalgebra::Vector2<f32> {
        self.into().into()
    }
}

// clunky but whatever
pub trait ConvertFloat32<S> {
    fn to_f32(self) -> S;
}

pub trait ConvertFloat64<D> {
    fn to_f64(self) -> D;
}

impl ConvertFloat32<ultraviolet::Vec2> for ultraviolet::DVec2 {
    fn to_f32(self) -> ultraviolet::Vec2 {
        ultraviolet::Vec2::new(self.x as f32, self.y as f32)
    }
}

impl ConvertFloat64<ultraviolet::DVec2> for ultraviolet::Vec2 {
    fn to_f64(self) -> ultraviolet::DVec2 {
        ultraviolet::DVec2::new(self.x as f64, self.y as f64)
    }
}

// impl<Via: Into<[f64; 2]>> ConvertVec2 for Via {
//     // impl ConvertVec2 for [f64; 2] {
//     fn as_uv(self) -> ultraviolet::Vec2 {
//         let [x, y] = self.into();
//         ultraviolet::Vec2::new(x as f32, y as f32)
//     }
//     fn as_epos2(self) -> egui::Pos2 {
//         let [x, y] = self.into();
//         egui::pos2(x as f32, y as f32)
//     }
//     fn as_evec2(self) -> egui::Vec2 {
//         let [x, y] = self.into();
//         egui::vec2(x as f32, y as f32)
//     }
//     fn as_na(self) -> nalgebra::Vector2<f32> {
//         let [x, y] = self.into();
//         nalgebra::Vector2::new(x as f32, y as f32)
//     }
// }

pub trait ConvertDVec2 {
    fn as_duv(self) -> ultraviolet::DVec2;
    // fn as_epos2(self) -> egui::Pos2;
    // fn as_evec2(self) -> egui::Vec2;
    fn as_dna(self) -> nalgebra::Vector2<f64>;
}

impl<Via: Into<[f64; 2]>> ConvertDVec2 for Via {
    fn as_duv(self) -> ultraviolet::DVec2 {
        self.into().into()
    }
    fn as_dna(self) -> nalgebra::Vector2<f64> {
        self.into().into()
    }
}

pub trait ConvertVec3 {
    fn as_uv(self) -> ultraviolet::Vec3;
    fn as_na(self) -> nalgebra::Vector3<f32>;
}

impl<Via: Into<[f32; 3]>> ConvertVec3 for Via {
    fn as_uv(self) -> ultraviolet::Vec3 {
        self.into().into()
    }
    fn as_na(self) -> nalgebra::Vector3<f32> {
        self.into().into()
    }
}

pub trait ConvertMat3 {
    fn as_uv(self) -> ultraviolet::Mat3;
    fn as_na(self) -> nalgebra::Matrix3<f32>;
}

impl<Via: Into<[[f32; 3]; 3]>> ConvertMat3 for Via {
    fn as_uv(self) -> ultraviolet::Mat3 {
        self.into().into()
    }

    fn as_na(self) -> nalgebra::Matrix3<f32> {
        nalgebra::Matrix3::from_data(nalgebra::ArrayStorage(self.into()))
    }
}
