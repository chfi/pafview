use bevy::{math::DVec2, prelude::*};
use bevy_egui::{EguiClipboard, EguiContexts};

use nalgebra::{OPoint, Point2};
use svg::node::element::{
    path::{self, Data},
    Path,
};
use time::OffsetDateTime;

use super::{
    alignments::{layout::SeqPairLayout, AlignmentLayoutQuery},
    render::sampled_lines::{
        pipeline::PolylineVertices, AlignmentCollisionLines, SampledAlignmentViewer,
        VertexSamplingTask,
    },
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

    layouts: AlignmentLayoutQuery,

    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if !keyboard.just_pressed(KeyCode::F12) {
        return;
    }

    let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
        return;
    };

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

    let grid_path = Path::new()
        .set("fill", "none")
        .set("stroke", "black")
        .set("stroke-width", 0.5)
        .set(
            "d",
            grid_paths_in_view(layout, &params.view, params.canvas_size.as_vec2()),
        );

    document = document.add(grid_path);

    for (key, polyline) in lines.polylines.iter() {
        let mut points = polyline.vertices().iter().map(|p| (p.x as f32, p.y as f32));
        // .map(|p| (p.x as f32, h - p.y as f32));

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
            .set("stroke-width", 5)
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
    let ymdhms = format!("{y}-{mon:02}-{d:02}-{h:02}-{min:02}-{s:02}");

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

fn grid_paths_in_view(
    // document: &mut Document,
    tile_layout: &SeqPairLayout,
    view: &crate::view::View,
    canvas_size: Vec2,
) -> path::Data {
    let mut data = Data::new();

    let vis_tiles = tile_layout
        .layout_qbvh
        .aabbs_in_rect(view.center(), view.size() * 0.5);

    for tile in vis_tiles.iter() {
        let Some(aabb) = tile_layout.aabbs.get(tile) else {
            continue;
        };

        let center = view.map_world_to_screen(canvas_size, aabb.center().coords.data.0[0]);
        let size = DVec2::from(aabb.extents().data.0[0]) * (canvas_size.x as f64 / view.width());

        let p = Vec2::new(center.x, center.y);
        let size = size.as_vec2();
        let halfsize = size * 0.5;
        let dx = Vec2::X * size;
        let dy = Vec2::Y * size;

        let p0 = p - halfsize;
        let p1 = p0 + dx;
        let p2 = p1 + dy;
        let p3 = p2 - dx;

        let f = |p: Vec2| (p.x, p.y);

        data = data
            .move_to(f(p0))
            .line_to(f(p1))
            .line_to(f(p2))
            .line_to(f(p3))
            .close();

        // .line_to(())
        // data.move_to()
        // data.move_to(center)
        // let c = DVec2::from(aabb.center().coords.
        // let screen_center = aabb.center()
    }

    data
}
