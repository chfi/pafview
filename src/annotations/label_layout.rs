use std::sync::Arc;

use egui::Galley;
use na::Isometry2;
use rapier2d::{
    dynamics::RigidBodySet,
    geometry::{Collider, ColliderBuilder, ColliderSet, HalfSpace, RayCast, SharedShape},
    na::{self, vector},
    parry::{query::Ray, shape::Cuboid},
    pipeline::QueryPipeline,
};
use rustc_hash::FxHashMap;

// TODO generalize for query annotations that "fall right" too
pub struct LabelDef<'a> {
    text: &'a str,
    world_x_region: std::ops::RangeInclusive<f64>,
}

fn compute_layout_for_labels<'a, Labels>(
    grid: &crate::AlignmentGrid,
    galley_cache: &FxHashMap<String, Arc<Galley>>,
    canvas_size: impl Into<[f32; 2]>,
    view: &crate::view::View,
    labels: Labels,
) -> Vec<(egui::Pos2, Arc<Galley>)>
where
    Labels: IntoIterator<Item = LabelDef<'a>> + 'a,
{
    let canvas_size = canvas_size.into();
    // let dims = [width as u32, height as u32];
    // let [width, height] = canvas_size.into();
    // let dims = [width as u32, height as u32];

    // for the menubar; TODO: actually take the menubar size into
    // account (here and elsewhere)
    let top_padding = 32f32;
    let pad_world = (top_padding as f64 / canvas_size[1] as f64) * view.height();

    let y_min = view.y_min + pad_world;
    // let x_min = view_x_min;

    let mut placed_shapes: Vec<(Cuboid, Isometry2<f32>)> = Vec::new();
    let mut placed_labels = Vec::new();

    // TODO build and iterate the Qbvh directly
    // let bodies = RigidBodySet::default();
    // let mut colliders = ColliderSet::default();

    // let mut query_pipeline = QueryPipeline::default();

    for label in labels {
        let Some(galley) = galley_cache.get(label.text) else {
            continue;
        };

        let size = galley.size();
        // let size =

        let size_arr: [f32; 2] = size.into();

        let shape = Cuboid::new(vector![size.x, size.y] * 0.5);
        let x = (*label.world_x_region.start() + *label.world_x_region.end()) * 0.5;

        let origin_w = [x, y_min];
        let origin = view.map_world_to_screen(canvas_size, origin_w);

        let ray = Ray::new(na::Point2::new(origin.x, origin.y), [0.0, 1.0].into());

        let collides = placed_shapes
            .iter()
            .enumerate()
            .find_map(|(ix, (other_shape, iso))| {
                // let toi = other.cast_local_ray(&ray, 10_000.0, false)?;
                let toi = other_shape.cast_ray(&iso, &ray, 10_000.0, false)?;
                Some((ix, toi))
            });

        if let Some((_other_ix, toi)) = collides {
            let impact = ray.point_at(toi);
            let top = impact - vector![0.0, size.y];
            let iso = Isometry2::new([top.x, top.y].into(), 0.0);
            placed_shapes.push((shape, iso));

            let label_pos = egui::pos2(top.x - size.x * 0.5, top.y);
            placed_labels.push((label_pos, galley.clone()));
        } else {
            let top = vector![ray.origin.coords.x, canvas_size[1] - 32.0];

            let iso = Isometry2::new([top.x, top.y].into(), 0.0);
            placed_shapes.push((shape, iso));

            let label_pos = egui::pos2(top.x - size.x * 0.5, top.y);
            placed_labels.push((label_pos, galley.clone()));
        }
    }

    placed_labels
}
