use bimap::BiMap;
use egui::{util::IdTypeMap, Color32, DragValue, FontId, Ui};
use rustc_hash::FxHashMap;
use ultraviolet::{Mat4, Vec2};

use crate::{annotations::AnnotationStore, view::View, AlignedSeq, PafInput, PafViewerApp};

pub fn line_width_control(ctx: &egui::Context, renderer: &mut crate::PafRenderer) {
    egui::Window::new("Renderer").show(ctx, |ui| {
        ui.label("Match line width");

        ui.add(egui::DragValue::new(&mut renderer.line_width).clamp_range(1f32..=20f32));
    });
}
