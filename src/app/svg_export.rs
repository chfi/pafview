use bevy::prelude::*;
use bevy_egui::{EguiClipboard, EguiContexts};

use nalgebra::{OPoint, Point2};
use svg::node::element::{path, Path};
use time::OffsetDateTime;

use super::render::sampled_lines::{
    pipeline::PolylineVertices, AlignmentCollisionLines, SampledAlignmentViewer, VertexSamplingTask,
};

pub struct SvgExportPlugin;

impl Plugin for SvgExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, export_svg_screenshot);
    }
}

fn export_svg_screenshot(
    // mut commands: Commands,
    main_viewer: Query<
        (&SampledAlignmentViewer, &AlignmentCollisionLines),
        Without<VertexSamplingTask>,
    >,

    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if !keyboard.just_pressed(KeyCode::F12) {
        return;
    }

    println!("exporting view as SVG...");

    let Ok((viewer, lines)) = main_viewer.get_single() else {
        return;
    };

    let Some(params) = viewer.last_rendered else {
        return;
    };

    let w = params.canvas_size.x as f32;
    let h = params.canvas_size.y as f32;

    let mut document = svg::Document::new().set("viewBox", (0.0, 0.0, w, h));

    for (key, polyline) in lines.polylines.iter() {
        let mut points = polyline.vertices().iter().map(|p| (p.x as f32, p.y as f32));

        let Some(p0) = points.next() else {
            continue;
        };

        let mut path_data = path::Data::new().move_to(p0);

        for point in points {
            path_data = path_data.line_to(point);
        }

        // TODO set color from color schemes
        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 8)
            .set("d", path_data.close());

        document = document.add(path);
    }

    let Ok(time) = std::time::UNIX_EPOCH.elapsed().map(|t| t.as_secs()) else {
        return;
    };
    let Ok(time) = OffsetDateTime::from_unix_timestamp(time as i64) else {
        return;
    };

    let (y, mon, d) = time.to_calendar_date();
    let (h, min, s) = time.to_hms();

    let ymdhms = format!("{y}-{mon}-{d}-{h}:{min}:{s}");

    // TODO proper file name (include PAF name, region (configurable))
    let file_name = format!("pafview-{ymdhms}.svg");

    match svg::save(&file_name, &document) {
        Ok(_) => {
            println!("saved SVG: {file_name}");
        }
        Err(e) => {
            log::error!("Error saving SVG: {e}");
        }
    }
}
