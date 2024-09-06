//

use crate::AppConfig;

pub fn application_settings_window(
    ctx: &egui::Context,
    open: &mut bool,
    config: &mut AppConfig,
    //
) {
    egui::Window::new("Settings").open(open).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                let mut line_width = config.alignment_line_width;
                let alignment_line_width_range = 0.5..=80.0;

                ui.label("Alignment line width");

                if ui
                    .add(egui::Slider::new(
                        &mut line_width,
                        alignment_line_width_range,
                    ))
                    .changed()
                {
                    config.alignment_line_width = line_width;
                }
            });

            ui.horizontal(|ui| {
                let mut line_width = config.grid_line_width;
                let grid_line_width_range = 0.0..=5.0;

                ui.label("Grid line width");

                if ui
                    .add(egui::Slider::new(&mut line_width, grid_line_width_range))
                    .changed()
                {
                    config.grid_line_width = line_width;
                }
            });

            ui.separator();

            ui.horizontal(|ui| {
                let mut opacity = config.annotation_draw_config.color_region_opacity;

                ui.label("Annotated region opacity");
                let drag_resp = ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0));

                if drag_resp.changed() {
                    config.annotation_draw_config.color_region_opacity = opacity
                }
            });

            ui.horizontal(|ui| {
                let val = &mut config.annotation_draw_config.color_region_border;
                let text = if *val {
                    "Disable annotated region borders"
                } else {
                    "Enable annotated region borders"
                };

                if ui.button(text).clicked() {
                    *val = !*val;
                }
            });

            ui.separator();

            if ui.button("Reset default settings").clicked() {
                *config = AppConfig::default();
            }
        });
    });

    //
}
