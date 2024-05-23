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
