use egui::{util::IdTypeMap, DragValue, Ui};
use ultraviolet::{Mat4, Vec2};

use crate::PafInput;

struct ViewControlResult {
    new_projection: Mat4,
}

// fn view_controls(input: &PafInput, proj: Mat4, ui: &mut Ui) {
pub(crate) fn view_controls(input: &PafInput, proj: &mut Mat4, ctx: &egui::Context) {
    // let origin = Vec2::new(proj[0][3] * 2.0, proj[1][3] * 2.0);
    // let orig_x = proj[3][0] * 2.0;
    // let orig_y = proj[3][1] * 2.0;

    let orig_x = proj[3][0] * -2.0 * proj[0][0];
    let orig_y = proj[3][1] * -2.0 * proj[1][1];
    let view_w = 2.0 / proj[0][0];
    let view_h = 2.0 / proj[1][1];

    let mouse_pos = ctx.input(|i| i.pointer.hover_pos());

    // println!("{proj:#?}");
    // let vwidth = 2.0 / projection[0][0];
    // let vheight = 2.0 / projection[1][1];
    // let mut new_x: Option<[f64; 2]> = None;
    // let mut new_y: Option<[f64; 2]> = None;
    // let mut new_xl: Option<f64> = None;
    // let mut new_xr: Option<f64> = None;

    // let mut new_yu: Option<f32> = None;
    // let mut new_yd: Option<f64> = None;

    let mut xl = orig_x - view_w / 2.0;
    let mut xr = orig_x + view_w / 2.0;

    let mut yu = orig_y - view_h / 2.0;
    let mut yd = orig_y + view_h / 2.0;

    // let range_text_widget = |ui: &mut Ui, id: &str, default: &str| {
    //     ui.horizontal(|ui| {
    //         ctx.data_mut(|data| {
    //             let txt = data.get_temp_mut_or_insert_with(ui.id().with(id), || default.to_string());
    //         });
    //     });
    // };

    let range_text_widget = |ui: &mut Ui, id: &str, default: String| {
        ui.horizontal(|ui| {
            let id = ui.id().with(id);
            let mut txt =
                ui.data_mut(|data| data.get_temp_mut_or_insert_with(id, || default).clone());
            // ui.data_mut(|data| {
            //     let txt = data.get_temp_mut_or_insert_with(ui.id().with(id), || default);

            let text_entry = ui.text_edit_singleline(&mut txt);

            let apply = ui.button("Apply X range");

            if text_entry.changed() {
                ui.data_mut(|data| data.insert_temp(id, txt.clone()));
            }

            if text_entry.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
                || apply.clicked()
            {
                // this doesn't work properly since the min is negative currently (not translated properly)
                let mut split = txt.split('-');
                let parsed = split.next().zip(split.next()).and_then(|(min, max)| {
                    let min = min.parse::<usize>().ok()?;
                    let max = max.parse::<usize>().ok()?;
                    Some((min, max))
                });

                return parsed;
            }
            return None;
        })
    };

    egui::Window::new("View")
        // .resizable(true)
        // .vscroll(true)
        // .default_open(false)
        .show(&ctx, |mut ui| {
            ui.label(format!("Origin: ({orig_x}, {orig_y})"));
            ui.label(format!("Mouse (screen): {mouse_pos:?}"));
            ui.label(format!("View width: ({view_w}, {view_h})"));

            let x_range_w =
                range_text_widget(ui, "x-range", format!("{}-{}", xl.round(), xr.round()));

            let y_range_w =
                range_text_widget(ui, "y-range", format!("{}-{}", yu.round(), yd.round()));

            let mut update_proj = false;

            if let Some((new_xl, new_xr)) = x_range_w.inner {
                update_proj = true;
                xl = new_xl as f32;
                xr = new_xr as f32;
            }

            if let Some((new_yu, new_yd)) = y_range_w.inner {
                update_proj = true;
                yu = new_yu as f32;
                yd = new_yd as f32;
            }

            if update_proj {
                *proj = ultraviolet::projection::orthographic_wgpu_dx(xl, xr, yu, yd, 0.1, 10.0);
            }

            // let id = ui.id();
            // let (mut x_range_text, mut y_range_text) = ctx.data_mut(|data| {
            //     let x_id = ui.id().with("x-range");
            //     let xtxt = data
            //         .get_temp_mut_or_insert_with(x_id, || format!("{xl:2}-{xr:2}"))
            //         .clone();

            //     let y_id = ui.id().with("y-range");
            //     let ytxt = data
            //         .get_temp_mut_or_insert_with(y_id, || format!("{yu:2}-{yd:2}"))
            //         .clone();

            //     (xtxt, ytxt)
            // });

            // ui.horizontal(|ui| {
            //     let x_range_entry = ui.text_edit_singleline(&mut x_range_text);

            // });
            // let x_range_entry = ui.text_edit_singleline(&mut x_range_text)
            // let xl_range = DragValue::new(&mut xl).clamp_range(0f32..(xr - 1000.0).max(0.0));
            // let xr_range = DragValue::new(&mut xr)
            //     .clamp_range((xl + 1000.0)..(xr - 1000.0).max(0.0)));
        }); //
}
