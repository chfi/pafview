pub fn line_width_control(ctx: &egui::Context, renderer: &mut crate::render::PafRenderer) {
    egui::Window::new("Renderer").show(ctx, |ui| {
        ui.label("Match line width");

        ui.add(egui::DragValue::new(&mut renderer.line_width).clamp_range(1f32..=20f32));
    });
}
