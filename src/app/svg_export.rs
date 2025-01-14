use avian2d::parry::partitioning::QbvhUpdateWorkspace;
use bevy::{ecs::query, math::DVec2, prelude::*};
use bevy_egui::{EguiClipboard, EguiContexts};

use nalgebra::{OPoint, Point2};
use svg::node::element::{
    path::{self, Data},
    Path,
};
use time::OffsetDateTime;

use crate::{app::alignments::layout::AabbQbvh, toast::ToastMessageEvent};

use super::{
    alignments::{layout::SeqPairLayout, AlignmentLayoutQuery},
    annotations::Annotations,
    render::sampled_lines::{
        pipeline::PolylineVertices, AlignmentCollisionLines, SampledAlignmentViewer,
        VertexSamplingTask,
    },
    view::AlignmentViewport,
};

pub struct SvgExportPlugin;

impl Plugin for SvgExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, export_svg_screenshot);
    }
}

fn export_svg_screenshot(
    // mut commands: Commands,
    annotations: Res<Annotations>,
    alignment_view: Res<AlignmentViewport>,

    main_viewer: Query<
        (&SampledAlignmentViewer, &AlignmentCollisionLines),
        Without<VertexSamplingTask>,
    >,

    layouts: AlignmentLayoutQuery,

    keyboard: Res<ButtonInput<KeyCode>>,

    mut toast_msgs: EventWriter<ToastMessageEvent>,
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

    let view = &alignment_view.view;

    let screen_dims = params.canvas_size;

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

    let annotation_regions = annotations.0.annotation_lists.iter().flat_map(|list| {
        list.records.iter().filter_map(|record| {
            let region = layout.map_local_region_to_screen(
                view,
                screen_dims.as_vec2(),
                (record.tgt_id, record.tgt_range.clone()),
                (record.qry_id, record.qry_range.clone()),
            )?;

            Some((region, record.color, record.label.as_str()))
        })
    });

    document = document.add(annotations_element(
        lines,
        screen_dims.as_vec2(),
        annotation_regions,
    ));

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
            toast_msgs.send(ToastMessageEvent {
                header: "SVG Export".into(),
                body: format!("Screenshot saved to {file_name}"),
            });
            // println!("saved SVG: {file_name}");
        }
        Err(e) => {
            toast_msgs.send(ToastMessageEvent {
                header: "SVG Export".into(),
                body: format!("Error saving SVG: {e}"),
            });
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
    }

    data
}

fn position_target_label(
    // qbvh: &mut avian2d::parry::partitioning::Qbvh<u32>,
    qbvh: &mut AabbQbvh<u32>,
    qbvh_workspace: &mut avian2d::parry::partitioning::QbvhUpdateWorkspace,
    screen_dims: Vec2,
    // annotated region associated with label, in screenspace
    label_region: [Vec2; 2],
    label_size: Vec2,
    label_text: &str,
) -> Option<Vec2> {
    // choose offset for label inside `label_region`, adding to `qbvh` if position is found
    //
    // check `qbvh` for aabbs inside `label_region`, extended by `label_size`'s width...
    //
    use avian2d::parry::{
        self,
        bounding_volume::{Aabb, BoundingVolume},
        partitioning::QbvhUpdateWorkspace,
    };

    let [mins, maxs] = label_region;

    let mut mid = (mins + maxs).as_dvec2() * 0.5;
    let mut half_extents = (maxs - mins).as_dvec2() * 0.5;

    if label_size.x as f64 > 2.0 * half_extents.x {
        let half_label = label_size.x as f64 * 0.5;
        let extra = half_label - half_extents.x;

        mid.x -= extra;
        half_extents += extra;
        // half_extents.x = half_extents.x.max(label_size.x as f64);
    }

    // let query_aabb = Aabb::from_half_extents(mid.to_array().into(), half_extents.to_array().into());

    let mut column_collisions = Vec::new();

    qbvh.aabbs_in_rect_callback(mid, half_extents, |label_ix, aabb| {
        column_collisions.push(*aabb);
        // column_collisions.push((label_ix, *aabb));
        true
    });

    column_collisions.sort_by_key(|aabb| aabb.mins.y as u64);

    // let mut current_y = mins.y;
    let mut final_pos: Option<Vec2> = None;

    let label_halfsize = label_size * 0.5;

    let pos = mins + label_halfsize;
    // let pos = mins + label_halfsize + Vec2::Y * 30.0;

    let mut this_aabb = Aabb::from_half_extents(
        pos.as_dvec2().to_array().into(),
        label_halfsize.as_dvec2().to_array().into(),
    );

    if column_collisions.is_empty() {
        final_pos = Some(pos);
    }

    println!(
        "{label_text} potential collisions: {}",
        column_collisions.len()
    );

    // iterating through the other labels that intersect this region, from the top
    for other_aabb in column_collisions.iter() {
        let p0 = this_aabb.center();

        let p1 = other_aabb.center();

        let this_bottom = p0.y + this_aabb.half_extents().y;
        let other_top = p1.y - other_aabb.half_extents().y;

        if !this_aabb.intersects(other_aabb) {
            // if  this_bottom < other_top {
            // this label would fit before this one, so we can use it & finish

            final_pos = Some(Vec2::new(p0.x as f32, p0.y as f32));
            dbg!(&final_pos);
            // dbg!(&final_pos);
            break;
        } else {
            // this label would collide, so move the candidate position
            // down below it

            let delta_y = 50.0;
            println!(
                " > {label_text} attempted at [{}, {}], moving to Y {}",
                p0.x,
                p0.y,
                p0.y + delta_y
            );

            this_aabb = this_aabb.transform_by(&nalgebra::Isometry2::translation(0.0, delta_y));
            dbg!();

            /*
            let new_y = p1.y + other_aabb.half_extents().y + this_aabb.half_extents().y;
            let delta_y = new_y - p0.y;
            this_aabb = this_aabb.transform_by(&nalgebra::Isometry2::translation(0.0, delta_y));
             */
        }
    }

    let mut final_pos_clear = true;

    qbvh.aabbs_in_rect_callback(mid, half_extents, |label_ix, aabb| {
        if aabb.intersects(&this_aabb) {
            final_pos_clear = false;
            return false;
        }
        true
    });

    if final_pos_clear {
        let p0 = this_aabb.center();
        final_pos = Some(Vec2::new(p0.x as f32, p0.y as f32));
    }

    if let Some(pos) = final_pos {
        let pos = pos - label_halfsize;
        let i = qbvh.data.len() as u32;
        let label_aabb = Aabb::from_half_extents(
            pos.as_dvec2().to_array().into(),
            label_size.as_dvec2().to_array().into(),
        );

        println!(" > {label_text} placed at [{}, {}]", pos.x, pos.y);
        qbvh.add(qbvh_workspace, i, label_aabb);

        Some(pos)
    } else {
        // dbg!();
        None
    }
}

fn annotations_element<'a>(
    alignment_lines: &AlignmentCollisionLines,
    screen_dims: Vec2,
    transformed_annotations: impl Iterator<Item = ([Vec2; 2], egui::Color32, &'a str)>,
) -> svg::node::element::Group {
    use avian2d::parry::partitioning::Qbvh;
    use svg::node::element::Rectangle;
    let mut group = svg::node::element::Group::new();

    let mut label_qbvh: AabbQbvh<u32> = AabbQbvh::new();
    let mut qbvh_workspace = QbvhUpdateWorkspace::default();

    for ([mins, maxs], color, label) in transformed_annotations {
        let [r, g, b, a] = color.to_array();

        let color_str = format!("rgb({r} {g} {b})");
        let opac_str = format!("{}", a as f32 / 255.0);

        let target_region = [Vec2::new(mins.x, 0.0), Vec2::new(maxs.x, screen_dims.y)];
        let query_region = [Vec2::new(0.0, mins.y), Vec2::new(screen_dims.x, maxs.y)];

        let target_rect = Rectangle::new()
            .set("x", mins.x)
            .set("y", 0.0)
            .set("width", maxs.x - mins.x)
            .set("height", screen_dims.y)
            .set("stroke", color_str.as_str())
            .set("fill", color_str.as_str())
            .set("opacity", opac_str.as_str());

        let query_rect = Rectangle::new()
            .clone()
            .set("x", 0.0)
            .set("y", mins.y)
            .set("width", screen_dims.x)
            .set("height", maxs.y - mins.y)
            .set("stroke", color_str.as_str())
            .set("fill", color_str.as_str())
            .set("opacity", opac_str.as_str());

        // TODO add labels

        // TODO use real label size
        let label_size = Vec2::X * 20.0 * label.len() as f32 + Vec2::Y * 20.0;

        if let Some(label_pos) = position_target_label(
            &mut label_qbvh,
            &mut qbvh_workspace,
            screen_dims,
            target_region,
            // [mins, maxs],
            label_size,
            label,
        ) {
            let rect = Rectangle::new()
                .set("x", label_pos.x)
                .set("y", label_pos.y)
                .set("width", 20.0 * label.len() as f32)
                .set("height", 20.0)
                .set("stroke", color_str.as_str())
                .set("fill", color_str.as_str());
            group = group.add(rect);

            let text = svg::node::element::Text::new(label)
                .set("font-family", "monospace")
                .set("font-size", "20px")
                .set("x", label_pos.x)
                .set("y", label_pos.y);

            group = group.add(text);
        }

        // TODO avoid alignment lines

        // group = group.add(target_rect).add(query_rect);
    }

    group
}

// fn annotation_labels_element(
//     screen_dims: Vec2,
// )
