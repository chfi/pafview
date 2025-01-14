use avian2d::parry::{partitioning::QbvhUpdateWorkspace, query::PointQuery};
use bevy::{ecs::query, math::DVec2, prelude::*, render::view::RenderLayers};
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
        pipeline::PolylineVertices, spawn_alignment_sampling_tasks, AlignmentCollisionLines,
        SampledAlignmentViewer, VertexSamplingTask,
    },
    view::AlignmentViewport,
};

pub struct SvgExportPlugin;

impl Plugin for SvgExportPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            trigger_svg_export_screenshot.before(spawn_alignment_sampling_tasks),
        );

        app.add_systems(PostUpdate, export_svg_screenshot);
    }
}

#[derive(Component)]
struct SvgExportInProgress;

fn trigger_svg_export_screenshot(
    mut commands: Commands,
    mut main_viewer: Query<(
        Entity,
        &mut SampledAlignmentViewer,
        Has<SvgExportInProgress>,
    )>,

    keyboard: Res<ButtonInput<KeyCode>>,
) {
    for (viewer_entity, mut viewer, is_exporting) in main_viewer.iter_mut() {
        if keyboard.just_pressed(KeyCode::F12) && !is_exporting {
            // TODO only do this if view has changed from sampling params
            viewer.force_resample = true;
            commands.entity(viewer_entity).insert(SvgExportInProgress);
        }
    }
}

fn export_svg_screenshot(
    mut commands: Commands,
    annotations: Res<Annotations>,
    alignment_view: Res<AlignmentViewport>,

    main_viewer: Query<
        (Entity, &SampledAlignmentViewer, &AlignmentCollisionLines),
        (With<SvgExportInProgress>, Without<VertexSamplingTask>),
    >,

    layouts: AlignmentLayoutQuery,

    mut toast_msgs: EventWriter<ToastMessageEvent>,
) {
    let Ok((viewer_entity, viewer, lines)) = main_viewer.get_single() else {
        return;
    };

    println!("exporting view as SVG...");

    let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
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

    let mut alignment_paths = svg::node::element::Group::new();

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

        alignment_paths = alignment_paths.add(path);
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

    let (annot_regions, annot_labels) =
        annotations_element(lines, screen_dims.as_vec2(), annotation_regions);
    document = document
        .add(grid_path)
        .add(annot_regions)
        .add(alignment_paths)
        .add(annot_labels);
    // document = document.add();

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

    commands
        .entity(viewer_entity)
        .remove::<SvgExportInProgress>();
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
    alignment_lines: &AlignmentCollisionLines,
    screen_dims: Vec2,
    // annotated region associated with label, in screenspace
    label_region: [Vec2; 2],
    label_size: Vec2,
    label_text: &str,
) -> Option<(Vec2, avian2d::parry::bounding_volume::Aabb)> {
    // choose offset for label inside `label_region`, adding to `qbvh` if position is found
    //
    // check `qbvh` for aabbs inside `label_region`, extended by `label_size`'s width...
    //
    use avian2d::parry::bounding_volume::{Aabb, BoundingVolume};

    let [mins, maxs] = label_region;

    let mut mid = (mins + maxs).as_dvec2() * 0.5;
    let mut half_extents = (maxs - mins).as_dvec2() * 0.5;
    // ensure the region isn't extremely small along either axis
    half_extents = half_extents.max(DVec2::ONE);

    if label_size.x as f64 > 2.0 * half_extents.x {
        let half_label = label_size.x as f64 * 0.5;
        let extra = half_label - half_extents.x;

        half_extents += extra;
    }

    // let query_aabb = Aabb::from_half_extents(mid.to_array().into(), half_extents.to_array().into());
    let mut column_collisions = Vec::new();

    qbvh.aabbs_in_rect_callback(mid, half_extents, |_, aabb| {
        column_collisions.push(*aabb);
        true
    });

    alignment_lines
        .qbvh
        .aabbs_in_rect_callback(mid, half_extents, |key, aabb| {
            if let Some(polyline) = alignment_lines.polylines.get(&key) {

                // cast vertical rays down from left and right sides of the label we're placing
                // - if a ray doesn't hit, use point projection...

                //
            }
            column_collisions.push(*aabb);
            true
        });

    column_collisions.sort_by_key(|aabb| aabb.mins.y as u64);

    let label_halfsize = label_size * 0.5;

    let pos = mins + label_halfsize * Vec2::Y;

    let mut this_aabb = Aabb::from_half_extents(
        pos.as_dvec2().to_array().into(),
        label_halfsize.as_dvec2().to_array().into(),
    );

    // iterating through the other labels that intersect this region, from the top
    for other_aabb in column_collisions.iter() {
        let p0 = this_aabb.center();
        let other_aabb = other_aabb.loosened(1.0);

        if !this_aabb.intersects(&other_aabb) {
            // if  this_bottom < other_top {
            // this label would fit before this one, so we can use it & finish
            break;
        } else {
            // this label would collide, so move the candidate position
            // down below it
            let new_y =
                other_aabb.center().y + other_aabb.half_extents().y + this_aabb.half_extents().y;
            let delta_y = new_y - p0.y;
            this_aabb = this_aabb.transform_by(&nalgebra::Isometry2::translation(0.0, delta_y));
        }
    }

    let mut final_pos_clear = true;

    qbvh.aabbs_in_rect_callback(this_aabb.center(), this_aabb.half_extents(), |_, aabb| {
        if aabb.intersects(&this_aabb) {
            final_pos_clear = false;
            return false;
        }
        true
    });

    if final_pos_clear {
        // label origin is on its left side, while AABBs are positioned by their center
        let pos = DVec2::from(this_aabb.center().coords.data.0[0]).as_vec2();
        let label_pos = pos - Vec2::X * label_halfsize.x;
        let i = qbvh.data.len() as u32;
        qbvh.add(qbvh_workspace, i, this_aabb);

        Some((label_pos, this_aabb))
    } else {
        None
    }
}

// returns (colored region group, label group)
fn annotations_element<'a>(
    alignment_lines: &AlignmentCollisionLines,
    screen_dims: Vec2,
    transformed_annotations: impl Iterator<Item = ([Vec2; 2], egui::Color32, &'a str)>,
) -> (svg::node::element::Group, svg::node::element::Group) {
    use svg::node::element::Rectangle;

    let mut region_group = svg::node::element::Group::new();
    let mut label_group = svg::node::element::Group::new();

    let mut label_qbvh: AabbQbvh<u32> = AabbQbvh::new();
    let mut qbvh_workspace = QbvhUpdateWorkspace::default();

    for ([mins, maxs], color, label) in transformed_annotations {
        let [r, g, b, a] = color.to_array();

        let color_str = format!("rgb({r} {g} {b})");
        let opac_str = format!("{}", a as f32 / 255.0);

        let target_region = [Vec2::new(mins.x, 0.0), Vec2::new(maxs.x, screen_dims.y)];
        let query_region = [Vec2::new(0.0, mins.y), Vec2::new(screen_dims.x, maxs.y)];

        // TODO use real label size
        let label_size = Vec2::X * 12.0 * label.len() as f32 + Vec2::Y * 20.0;

        // TODO avoid alignment lines
        if let Some((label_pos, label_aabb)) = position_target_label(
            &mut label_qbvh,
            &mut qbvh_workspace,
            alignment_lines,
            screen_dims,
            target_region,
            // [mins, maxs],
            label_size,
            label,
        ) {
            // let rect = Rectangle::new()
            //     .set("x", label_pos.x)
            //     .set("y", label_pos.y)
            //     .set("width", 20.0 * label.len() as f32)
            //     .set("height", 20.0)
            //     .set("stroke", color_str.as_str())
            //     .set("fill", color_str.as_str());
            // group = group.add(rect);

            let text = svg::node::element::Text::new(label)
                .set("font-family", "monospace")
                .set("font-size", "20px")
                .set("x", label_pos.x)
                .set("y", label_pos.y);

            label_group = label_group
                // .add(
                //     svg::node::element::Rectangle::new()
                //         .set("fill", get_color())
                //         .set("x", label_aabb.mins.x)
                //         .set("y", label_aabb.mins.y)
                //         .set("width", label_aabb.extents().x)
                //         .set("height", label_aabb.extents().y),
                // )
                .add(text);
        }

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

        region_group = region_group.add(target_rect).add(query_rect);
    }

    (region_group, label_group)
}
