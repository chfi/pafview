#[allow(dead_code)]
use rapier2d::parry::partitioning::IndexedData;
use rapier2d::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use ultraviolet::Vec2;

use crate::grid::{AlignmentGrid, AxisRange, GridAxis};
use crate::paf::{AlignmentLocation, Alignments};
use crate::sequences::SeqId;
use crate::view::Viewport;
use crate::{paf::Alignment, view::View};

use crate::math_conv::*;

use super::draw::AnnotationPainter;
use super::AnnotationStore;

pub mod debug;

/*

TODO:

- finish heightfield projection (doneish)
- anchor initialization & update (done)
- label creation (doneish)
- enable/disable physics & rendering via existing UI
- hooking up to rendering
-*- DrawAnnotation for AnnotationLabel chooses positions entirely by itself;
-*-*-  we need to feed the positions to the AnnotationPainter somehow
-*-*-  makes sense to have some sort of position associated with pretty much any displayed annotation, anyway


- labels for query annotations

*/

#[derive(Default)]
pub struct LabelPhysics {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub query_pipeline: QueryPipeline,

    physics: Physics,

    pub heightfields: AlignmentHeightFields,

    // annotation_map: AnnotationDefs,
    annotations: FxHashMap<super::AnnotationId, AnnotationData>,
    collider_annot_map: FxHashMap<ColliderHandle, super::AnnotationId>,

    // annotations_label_map: FxHashMap<super::AnnotationId, AnnotationLabelIxs>,
    target_labels: LabelHandles,
    query_labels: LabelHandles,
}

#[derive(Clone)]
struct AnnotationData {
    size: Vec2,
    world_range: AnnotationRange,

    // indices into the vectors stored in the `target_labels` and `query_labels`
    // `LabelHandles` in `LabelPhysics`
    target_label_ix: usize,
    query_label_ix: usize,
}

#[derive(Default)]
struct LabelHandles {
    annotation_id: Vec<super::AnnotationId>,
    anchor_screen_pos: Vec<Option<Vec2>>,
    label_rigid_body: Vec<Option<RigidBodyHandle>>,
}

impl LabelHandles {
    fn push(&mut self, annot_id: super::AnnotationId) -> usize {
        let id = self.annotation_id.len();
        self.annotation_id.push(annot_id);
        self.anchor_screen_pos.push(None);
        self.label_rigid_body.push(None);
        id
    }
}

impl LabelPhysics {
    pub fn prepare_annotations(
        &mut self,
        grid: &AlignmentGrid,
        annotation_store: &AnnotationStore,
        annotations: impl IntoIterator<Item = super::AnnotationId>,
        fonts: &egui::text::Fonts, // needed to derive text size
        annotation_painter: &mut super::draw::AnnotationPainter, // cache laid out text
    ) {
        let len = self.annotations.len();
        for annot_id @ (list_id, record_id) in annotations {
            if self.annotations.contains_key(&annot_id) {
                continue;
            }

            let Some(record) = annotation_store
                .list_by_id(list_id)
                .and_then(|list| list.records.get(record_id))
            else {
                continue;
            };

            let galley = annotation_painter.cache_label_fonts(fonts, &record.label);
            let size = galley.size().as_uv();

            let qry_range = AxisRange::Seq {
                seq_id: record.qry_id,
                range: record.qry_range.clone(),
            };
            let tgt_range = AxisRange::Seq {
                seq_id: record.tgt_id,
                range: record.tgt_range.clone(),
            };
            let world_x_range = grid.x_axis.axis_range_into_global(&tgt_range);
            let world_y_range = grid.y_axis.axis_range_into_global(&qry_range);

            let Some(world_range) = AnnotationRange::new(world_x_range, world_y_range) else {
                continue;
            };

            let target_label_ix = self.target_labels.push(annot_id);
            let query_label_ix = self.query_labels.push(annot_id);

            let data = AnnotationData {
                size,
                world_range,

                target_label_ix,
                query_label_ix,
            };

            self.annotations.insert(annot_id, data);
        }
        println!("prepared {} annotations", self.annotations.len() - len);
    }
}

impl LabelPhysics {
    pub fn update_anchors(
        &mut self,
        debug_painter: &egui::Painter,
        grid: &AlignmentGrid,
        viewport: &Viewport,
    ) {
        use ultraviolet::mat::*;
        use ultraviolet::vec::*;
        let world_screen = viewport.world_screen_dmat3();
        let [xs, ys, zs] = world_screen.cols;
        let ws_x2 = DMat3x2::new(DVec3x2::splat(xs), DVec3x2::splat(ys), DVec3x2::splat(zs));

        let (vx_min, vx_max) = viewport.x_range().into_inner();

        let mut count = 0;
        for (annot_id, annot_data) in self.annotations.iter() {
            // target for now

            // find anchor target (X) pos (middle of screen intersection w/ range)
            let Some(world_range) = annot_data.world_range.target_range() else {
                continue;
            };

            let (wx_min, wx_max) = world_range.clone().into_inner();

            let screen_xrange = ws_x2
                .transform_point2(DVec2x2::new([wx_min, wx_max].into(), [0.0, 0.0].into()))
                .x;

            let &[sx_min, sx_max] = screen_xrange.as_array_ref();

            let cl_min = sx_min.clamp(0.0, viewport.canvas_size.x as f64);
            let cl_max = sx_max.clamp(0.0, viewport.canvas_size.x as f64);

            if cl_min == cl_max && (cl_min == 0.0 || cl_min == viewport.canvas_size.x as f64) {
                // out of view bounds; remove anchor?
                // println!("range out of bounds");
                continue;
            }

            let cl_mid = (cl_min + cl_max) * 0.5;

            // project target onto heightfield
            let Some(mid_y) =
                self.heightfields
                    .project_screen_from_top(grid, viewport, cl_mid as f32)
            else {
                // println!("skipping");
                continue;
            };

            // update annotation anchor position (remove if offscreen)
            let new_anchor = if sx_max < 0.0
                || sx_min > viewport.canvas_size.x as f64
                || mid_y < 0.0
                || mid_y > viewport.canvas_size.y
            {
                None
            } else {
                Some([cl_mid as f32, mid_y].as_uv())
            };

            self.target_labels.anchor_screen_pos[annot_data.target_label_ix] = new_anchor;
            count += 1;
        }
        // println!("updated {count} anchors");
    }

    pub fn update_labels_new(
        &mut self,
        debug_painter: &egui::Painter,
        grid: &AlignmentGrid,
        annotations: &AnnotationStore,
        painter: &mut AnnotationPainter,
        viewport: &Viewport,
    ) {
        let world_screen_d = viewport.world_screen_dmat3();

        let mut label_move_buf: Vec<(ColliderHandle, ultraviolet::Vec2)> = Vec::new();
        // let mut label_move_buf: Vec<(ColliderHandle, ultraviolet::Vec2)> =
        //     self.handle_label_stack_and_swap(grid);

        let mut touched_handles = Vec::with_capacity(label_move_buf.len());

        for (c_handle, new_pos) in label_move_buf.drain(..) {
            let Some(collider) = self.collider_set.get(c_handle) else {
                continue;
            };

            let Some(body) = collider
                .parent()
                .and_then(|h| self.rigid_body_set.get_mut(h))
            else {
                continue;
            };

            let position = Isometry::translation(new_pos.x, new_pos.y);
            body.set_position(position, false);
            touched_handles.push(c_handle);
        }

        self.query_pipeline
            .update_incremental(&self.collider_set, &touched_handles, &[], false);

        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let anchor_pos = self.target_labels.anchor_screen_pos[handle_ix];
            let mut body_handle = self.target_labels.label_rigid_body[handle_ix];

            if anchor_pos.is_none() {
                // if there's no anchor, but the label has an enabled
                // rigid body, disable it

                if let Some(body) = body_handle.and_then(|h| self.rigid_body_set.get_mut(h)) {
                    if body.is_enabled() {
                        body.set_enabled(false);
                    }
                }
            }

            // at this point we're only dealing with annotations that
            // correspond to a target range that's visible
            let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] else {
                continue;
            };

            let mut label_pos: Option<Vec2> = None;

            // initialize rigid body if necessary
            if let Some(body) = body_handle.and_then(|h| self.rigid_body_set.get(h)) {
                label_pos = Some(body.position().translation.as_uv());
            } else {
                // if body_handle.is_none() {
                // body initialized as disabled
                let (collider, rigid_body) = label_collider_body(*annot_id, annot_data.size);

                // set position? or not
                let rb_handle = self.rigid_body_set.insert(rigid_body);
                let collider_handle = self.collider_set.insert_with_parent(
                    collider,
                    rb_handle,
                    &mut self.rigid_body_set,
                );
                self.query_pipeline.update_incremental(
                    &self.collider_set,
                    &[collider_handle],
                    &[],
                    true,
                );
                self.target_labels.label_rigid_body[handle_ix] = Some(rb_handle);
            }

            // NB: computing potential label position before retrieving the rigid body
            // due to aliasing (self.place_label... vs. mutable rigid body);
            if label_pos.is_none() {
                let (canvas_left, canvas_right) = {
                    let l = viewport.canvas_offset.x;
                    let r = l + viewport.canvas_size.x;
                    (l, r)
                };

                let screen_anchor_range = annot_data.world_range.target_range().map(|range| {
                    let left = *range.start();
                    let right = *range.end();
                    let s_left = world_screen_d.transform_point2([left, 0.0].into()).x as f32;
                    let s_right = world_screen_d.transform_point2([right, 0.0].into()).x as f32;

                    let s_left = s_left.clamp(canvas_left, canvas_right);
                    let s_right = s_right.clamp(canvas_left, canvas_right);

                    s_left..=s_right
                });

                let Some(screen_anchor_range) = screen_anchor_range else {
                    continue;
                };

                label_pos = self.place_label_aabb_with_anchor_range(
                    grid,
                    viewport,
                    screen_anchor_range,
                    annot_data.size,
                    &mut label_move_buf,
                );
            };

            // at this point, any annotation that can be visible on screen, given
            // the current view, will have a rigid body; if the body is disabled,
            // it is lacking a position
            let Some(rigid_body) = body_handle.and_then(|h| self.rigid_body_set.get_mut(h)) else {
                continue;
            };

            if !rigid_body.is_enabled() {
                // if a position for the label was successfully found,
                // update the position and enable the body if the
                // rigid body is already enabled, just move on to the
                // next step
                if let Some(pos) = label_pos {
                    rigid_body.set_translation(pos.as_na().into(), true);
                    rigid_body.set_enabled(true);
                }
            }

            let shape_id = annotations.target_shape_for(annot_id.0, annot_id.1);
            if let Some(label) = shape_id.and_then(|id| painter.get_shape_mut(id)) {
                let pos = rigid_body.position().translation;
                label.set_position(Some(pos.as_epos2()));
            }
        }

        //- move anchor to closest point on annotation range (on screen) to the label
        //  - if that point is directly below/overlapping the label, there's no need
        //    to apply any force

        // alternatively, use the overlap of the label with the annot. range (project
        // the label onto the world range) -- need some normalization but might make sense

        // *finally* (???) create the labels for the newly anchored annotations (i.e.
        // those that entered the screen on the last frame or for some other reason
        // are on the screen but lack a label position & body)

        // this includes labels that have a rigid body already, but have been disabled
        // due to leaving the screen bounds at some point

        /*
        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let Some(shape_id) = annotations.target_shape_for(annot_id.0, annot_id.1) else {
                continue;
            };

            let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] else {
                todo!(); // disable label rigid body if it exists etc.
                continue;
            };

            // if there's no rigid body for this label, try to initialize it
            // or if it has been previously disabled, reset it
            // -- the only difference between the two is having to create the collider etc.;
            // the result is the same

            // if the label already has an enabled rigid body, handle swaps & stacking?
        }
        */

        //
    }

    pub fn update_labels(
        &mut self,
        debug_painter: &egui::Painter,
        grid: &AlignmentGrid,
        annotations: &AnnotationStore,
        painter: &mut AnnotationPainter,
        viewport: &Viewport,
    ) {
        let mut position_count = 0;

        let mut label_move_buf: Vec<(ColliderHandle, ultraviolet::Vec2)> = Vec::new();

        // let contact_graph: &InteractionGraph<ColliderHandle, ContactPair> =
        //     self.physics.narrow_phase.contact_graph();

        // TODO may want to use a custom physics event handler that populates a structure
        // containing adjacent (contacting) labels

        let mut to_swap_set: FxHashSet<ColliderHandle> = FxHashSet::default();
        let mut to_swap: Vec<_> = Vec::new();
        // let mut to_swap: bimap::BiMap<ColliderHandle,

        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        struct ContactDirs {
            up: bool,
            down: bool,
            right: bool,
            left: bool,
        }

        let mut contact_dirs =
            vec![ContactDirs::default(); self.target_labels.anchor_screen_pos.len()];

        let mut update_contacts = |contact: &ContactPair| {
            // let collider = self.collider_set.get(handle)?;
            // let annot_id = u128_usize_pair(collider.user_data);
            let aid1 = self.collider_annot_map.get(&contact.collider1)?;
            let aid2 = self.collider_annot_map.get(&contact.collider2)?;
            // let annot_id = self.collider_annot_map.get(handle)?;
            let ix1 = self.annotations.get(aid1)?.target_label_ix;
            let ix2 = self.annotations.get(aid2)?.target_label_ix;

            for manifold in contact.manifolds.iter() {
                //
                if manifold.local_n1.y < 0.0 {
                    contact_dirs[ix1].up = true;
                    contact_dirs[ix2].down = true;
                } else if manifold.local_n1.y > 0.0 {
                    contact_dirs[ix1].down = true;
                    contact_dirs[ix2].up = true;
                }
            }
            Some(())
        };

        for pair in self.physics.narrow_phase.contact_pairs() {
            update_contacts(pair);

            // swap labels if it'd reduce the total distance to their respective anchors
            if to_swap_set.contains(&pair.collider1) || to_swap_set.contains(&pair.collider2) {
                continue;
            }

            // TODO: if contact point is on the bottom of label1, and on its right side,
            // i.e. so that label1 is partially supported by label2 (with nothing to the left
            // of label2), they should slide apart so that label1 can drop down a level, etc.

            let get_annot_for = |handle: ColliderHandle| {
                let collider = self.collider_set.get(handle)?;
                let annot_id = u128_usize_pair(collider.user_data);
                let pos = collider.position().translation.as_uv();
                let size = collider.shape().as_cuboid()?.half_extents.as_uv() * 2.0;
                Some((annot_id, pos, size))
            };

            let Some((label1, label2)) =
                get_annot_for(pair.collider1).zip(get_annot_for(pair.collider2))
            else {
                continue;
            };

            let data1 = self.annotations.get(&label1.0);
            let data2 = self.annotations.get(&label2.0);

            let Some((data1, data2)) = data1.zip(data2) else {
                continue;
            };

            let anchor1 = self.target_labels.anchor_screen_pos[data1.target_label_ix];
            let anchor2 = self.target_labels.anchor_screen_pos[data2.target_label_ix];

            let Some((anchor1, anchor2)) = anchor1.zip(anchor2) else {
                continue;
            };

            let dist1a1 = (anchor1 - label1.1).mag();
            let dist1a2 = (anchor2 - label1.1).mag();
            let dist2a1 = (anchor1 - label2.1).mag();
            let dist2a2 = (anchor2 - label2.1).mag();

            // let sum_before = dist1a1 + dist2a2;
            // let sum_after = dist1a2 + dist2a1;
            // println!("sum_before: {sum_before}\tsum_after: {sum_after}");

            if dist1a1 < dist1a2 && dist2a2 < dist2a1 {
                let Some((rb_handle1, rb_handle2)) = self.target_labels.label_rigid_body
                    [data1.target_label_ix]
                    .zip(self.target_labels.label_rigid_body[data2.target_label_ix])
                else {
                    continue;
                };

                to_swap_set.insert(pair.collider1);
                to_swap_set.insert(pair.collider2);

                to_swap.push([(rb_handle1, label2.1), (rb_handle2, label1.1)]);
                // to_swap.push([(pair.collider1, label2.1), (pair.collider2, label1.1)]);
            }
        }

        for (tgt_label_ix, contact_dirs) in contact_dirs.iter().enumerate() {
            if !contact_dirs.down {}
        }

        /*
        for [(rb1, new_p1), (rb2, new_p2)] in to_swap {
            // println!("swapping! {rb1:?}<->{rb2:?}");
            if let Some(body) = self.rigid_body_set.get_mut(rb1) {
                body.set_position(new_p1.as_na().into(), false);
            }
            if let Some(body) = self.rigid_body_set.get_mut(rb2) {
                body.set_position(new_p2.as_na().into(), false);
            }
        }
        */

        // let mut contact_count = 0;
        // for (a, b, edge) in contact_graph.interactions_with_endpoints() {
        //     println!("contact: {a:?} - {b:?}");
        //     contact_count += 1;
        // }

        // for edge in contact_graph.interactions() {
        //     let e: &ContactPair = edge;
        //     // println!("contact: {edge:?}");
        //     contact_count += 1;
        // }

        // println!("contact count: {contact_count}");

        // contact_graph.

        let (canvas_left, canvas_right) = {
            let l = viewport.canvas_offset.x;
            let r = l + viewport.canvas_size.x;
            (l, r)
        };

        let world_screen_d = viewport.world_screen_dmat3();

        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let Some(shape_id) = annotations.target_shape_for(annot_id.0, annot_id.1) else {
                continue;
            };

            let screen_anchor_range = annot_data.world_range.target_range().map(|range| {
                let left = *range.start();
                let right = *range.end();
                let s_left = world_screen_d.transform_point2([left, 0.0].into()).x as f32;
                let s_right = world_screen_d.transform_point2([right, 0.0].into()).x as f32;

                let s_left = s_left.clamp(canvas_left, canvas_right);
                let s_right = s_right.clamp(canvas_left, canvas_right);

                s_left..=s_right
            });

            let Some(screen_anchor_range) = screen_anchor_range else {
                continue;
            };

            if let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] {
                position_count += 1;
                let mut not_enough_space = false;

                if self.target_labels.label_rigid_body[handle_ix].is_none() {
                    // initialize label rigid body
                    let size = annot_data.size;

                    // if let Some(label_pos) =
                    //     self.try_to_place_label(grid, viewport, anchor_pos, size)
                    // {
                    println!("trying to place label for <{annot_id:?}> (screen pos {screen_anchor_range:?})");
                    if let Some(label_pos) = self.place_label_aabb_with_anchor_range(
                        grid,
                        viewport,
                        screen_anchor_range,
                        size,
                        &mut label_move_buf,
                    ) {
                        let collider = ColliderBuilder::cuboid(size.x * 0.5, size.y * 0.5)
                            .mass(1.0)
                            .friction(0.0)
                            .user_data(usize_pair_u128(*annot_id))
                            .build();

                        // println!("try_to_place_label results: {label_pos:?}");
                        let rigid_body = RigidBodyBuilder::dynamic()
                            .translation(label_pos.as_na())
                            .lock_rotations()
                            .linear_damping(3.0)
                            // .linear_damping(5.0)
                            .build();

                        let rb_handle = self.rigid_body_set.insert(rigid_body);
                        let collider_handle = self.collider_set.insert_with_parent(
                            collider,
                            rb_handle,
                            &mut self.rigid_body_set,
                        );
                        self.query_pipeline.update_incremental(
                            &self.collider_set,
                            &[collider_handle],
                            &[],
                            true,
                        );
                        self.collider_annot_map.insert(collider_handle, *annot_id);
                        self.target_labels.label_rigid_body[handle_ix] = Some(rb_handle);
                    } else {
                        not_enough_space = true;
                    };
                }

                // if label not visible,
                // show label, enable physics object

                let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                    .and_then(|rb_handle| {
                        let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                        Some((rb_handle, rigid_body))
                    })
                else {
                    continue;
                };

                if !rigid_body.is_enabled() {
                    rigid_body.set_enabled(true);
                }
                if let Some(label) = painter.get_shape_mut(shape_id) {
                    let pos = rigid_body.position().translation;
                    // println!("setting label position to {pos:?}");
                    label.set_position(Some(pos.as_epos2()));
                    // if !pos.x.is_nan() && !pos.y.is_nan() {
                    //     println!("setting label position to {pos:?}");
                    // }
                }
            } else {
                // hide label, disable physics object

                let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                    .and_then(|rb_handle| {
                        let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                        Some((rb_handle, rigid_body))
                    })
                else {
                    continue;
                };

                if rigid_body.is_enabled() {
                    rigid_body.set_enabled(false);
                    if let Some(label) = painter.get_shape_mut(shape_id) {
                        println!("unsetting label");
                        label.set_position(None);
                    }
                }
            }
        }

        // println!("labels with anchor positions: {position_count}");
    }

    pub fn step(&mut self, grid: &AlignmentGrid, dt: f32, viewport: &Viewport) {
        const SPRING_K: f32 = 10.0;
        const LABEL_ANCHOR_DIST_THRESHOLD: f32 = 300.0;
        // const CLUSTER_TIME_MIN_SEC: f32 = 1.0;
        const CLUSTER_TIME_MIN_SEC: f32 = 0.2;

        const STACK_MIN_FORCE_THRESHOLD: f32 = 10.0;

        for (annot_id, annot_data) in self.annotations.iter() {
            let handle_ix = annot_data.target_label_ix;

            let Some((rb_handle, rigid_body)) = self.target_labels.label_rigid_body[handle_ix]
                .and_then(|rb_handle| {
                    let rigid_body = self.rigid_body_set.get_mut(rb_handle)?;
                    Some((rb_handle, rigid_body))
                })
            else {
                continue;
            };

            let Some(anchor_pos) = self.target_labels.anchor_screen_pos[handle_ix] else {
                continue;
            };

            if !rigid_body.is_enabled() {
                continue;
            }

            let label_pos = rigid_body.position().translation.as_uv();
            let label_anchor_diff = label_pos - anchor_pos;
            // TODO check if stacking/clustering is needed (prediction etc.)

            if rigid_body.user_force().norm() > 0.0 {
                rigid_body.reset_forces(false);
            }

            let force_x = -label_anchor_diff.x * SPRING_K;

            let y_diff = label_anchor_diff.y;
            // println!("anchor: {anchor_pos:?}\tlabel: {label_pos:?}");
            // println!("y_diff: {y_diff}");
            // let hfield_y_diff = self.heightfields.project_screen_from_top(grid, viewport, screen_x)

            // TODO apply vertical force (to clear heightfield/match lines)
            // let force_y = 0f32;
            let y_tgt = (anchor_pos.y - 100.0).max(10.0);
            let force_y = -(label_pos.y - y_tgt) * SPRING_K;
            // let force_y = -label_anchor_diff.y * SPRING_K;
            // let force_y = if label_anchor_diff.y.abs() > 200.0 {
            //     -label_anchor_diff.y * SPRING_K
            // } else {
            //     0.0
            // };
            let force = [force_x, force_y].as_uv();
            let force_min = 1.0; // arbitrary
            if force.mag() > force_min {
                rigid_body.add_force(force.as_na(), true);
            }

            // TODO handle contacts, stack
        }

        self.physics.step(
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.query_pipeline,
        );
    }
}

impl LabelPhysics {
    fn try_to_place_label(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        // anchor_screen_x_range: std::
        anchor_pos: Vec2,
        rect_size: impl Into<[f32; 2]>,
    ) -> Option<ultraviolet::Vec2> {
        // println!("try to place anchor pos: {anchor_pos:?}");
        let proposed_center = anchor_pos + [0.0, -40.0].as_uv();
        self.find_position_for_screen_rectangle(grid, viewport, proposed_center, rect_size)
    }

    fn place_label_aabb_with_anchor_range(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        screen_anchor_range: std::ops::RangeInclusive<f32>,
        rect_size: impl Into<[f32; 2]>,
        move_buf: &mut Vec<(ColliderHandle, ultraviolet::Vec2)>,
    ) -> Option<ultraviolet::Vec2> {
        // println!("screen anchor range:  {screen_anchor_range:?}");
        let screen_world = viewport.screen_world_mat3();
        let screen_world_d = viewport.screen_world_dmat3();

        let world_screen_d = viewport.world_screen_dmat3();

        let (smin, smax) = screen_anchor_range.into_inner();

        let this_size = rect_size.as_uv();
        let this_shape = Cuboid::new((this_size * 0.5).as_na());

        let initial_x = (smin + smax) * 0.5;

        // return None;

        // pick (alignment pair) tile to use as anchor; take (rough) label position
        // into account (avoid tiles that don't have space in the viewport above for labels,
        // but don't do a full collision check)
        let initial_y = {
            let world_x = screen_world_d
                .transform_point2([initial_x as f64, 0.0].into())
                .x;
            // println!("screen x: {initial_x}\t world_x: {world_x}");
            let (tgt_id, _) = grid.x_axis.global_to_axis_local(world_x)?;
            // dbg!();

            let world_x = grid.x_axis.axis_local_to_global(tgt_id, 0.5)?;
            // dbg!();

            let top_visible_alignment =
                grid.topmost_visible_tile_at_target(viewport, world_x, true);
            // dbg!();

            // let top = (viewport.view_center.y + viewport.view_size.y * 0.35) as f32;
            let top = (viewport.view_center.y + viewport.view_size.y * 0.5) as f32;
            // let top = grid.y_axis.total_len as f32;

            let src = [world_x as f32, top];
            // println!("casting ray from {src:?}");
            let ((tgt_id, qry_id), hit_world) = grid.cast_ray(src, [0.0, -1.0], false)?;
            // dbg!();
            // grid.cast_ray([world_target as f32, 0.0], [0.0, 1.0])?;

            // let hfield = self.heightfields.heightfields.get(&(tgt_id, qry_id))?;
            // dbg!();
            // let (qry_id, hfield, hit_world) = self
            //     .heightfields
            //     .top_heightfield_in_visible_column(grid, viewport, world_x)?;

            world_screen_d.transform_point2(hit_world).y as f32
        };

        // dbg!(&initial_y);
        // should correspond to the spring distance factor but whatever
        const ANCHOR_EXTRA_RANGE: f32 = 300.0;
        // let initial_y =

        // let mut cur_pos = [(smin + smax) * 0.5,
        // let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, ground_screen_y);

        let lim_x_min = smin - ANCHOR_EXTRA_RANGE - this_size.x * 0.5;
        let lim_x_max = smax + ANCHOR_EXTRA_RANGE + this_size.x * 0.5;

        let mut cur_position = [initial_x, initial_y].as_uv();

        // let mut intersect_handle_buf = Vec::new();
        let mut intersect_buf = Vec::new();

        // loop per "row" (move up one level & stack when iterating)
        let mut iter_count = 0;
        loop {
            if iter_count > 5 {
                break;
            }
            iter_count += 1;
            let this_pos = nalgebra::Isometry2::translation(cur_position.x, cur_position.y);
            // dbg!(&cur_position);

            // check for other labels inside an AABB at the current position;
            // the AABB should extend horizontally to cover the entire space
            // available given the anchor range, without "straining" too much
            // (i.e. stay within the minimum spring force distance)

            // vertically it should only cover the label itself (?)

            // NB: the AABB should be clipped by the heightfield, though this only matters when zoomed in
            // (the AABB is offset above the heightfield anyway -- but it is an AABB, corresponding to
            // a single "line" or "row")

            let aabb = Aabb::from_points([
                &[lim_x_min, cur_position.y - this_size.y * 0.5]
                    .as_na()
                    .into(),
                &[lim_x_min, cur_position.y + this_size.y * 0.5]
                    .as_na()
                    .into(),
                &[lim_x_max, cur_position.y - this_size.y * 0.5]
                    .as_na()
                    .into(),
                &[lim_x_max, cur_position.y + this_size.y * 0.5]
                    .as_na()
                    .into(),
            ]);
            // intersect_handle_buf.clear();
            intersect_buf.clear();

            self.query_pipeline
                .colliders_with_aabb_intersecting_aabb(&aabb, |&handle| {
                    if let Some((collider, annot_data)) =
                        self.collider_set.get(handle).and_then(|c| {
                            Some((c, self.annotations.get(&u128_usize_pair(c.user_data))?))
                        })
                    {
                        intersect_buf.push((handle, collider, annot_data.clone()));
                    }

                    false
                });

            // sort intersections by X position (maybe lookup anchor/annotation range here?)
            intersect_buf.sort_by_cached_key(|(_handle, _collider, annot_data)| {
                annot_data
                    .world_range
                    .target_range()
                    .map(|r| ordered_float::OrderedFloat((*r.start() + *r.end()) * 0.5))
            });

            // iterate through intersections (should only be a few; we're only looking at a row here)

            // "figure out" if there's space in the AABB for this label; if so, we're done;
            // otherwise, if it's possible to move one (or more) of the intersecting labels
            // to make space, push the new/shifted positions to `move_buf` & return a position
            // that doesn't intersect any of the shifted labels
            // (since the label we're adding is constrained by the AABB, we know that it won't
            // intersect any other labels either)

            let mut lim_left = lim_x_min;
            let mut lim_right = lim_x_max;

            // far from complete!!
            for (handle, collider, annot_data) in &intersect_buf {
                let Some(other_size) = collider.shape().as_cuboid().map(|c| c.half_extents * 2.0)
                else {
                    continue;
                };
                let other_pos = collider.position().translation;
                let other_left = other_pos.x - other_size.x * 0.5;
                let other_right = other_pos.x + other_size.x * 0.5;

                let this_left = this_pos.translation.x - this_size.x * 0.5;
                let this_right = this_pos.translation.x + this_size.x * 0.5;

                if other_pos.x > cur_position.x {
                    // other to right; try pushing further right
                    let right_ray =
                        Ray::new([other_right, cur_position.y].into(), [1.0, 0.0].as_na());

                    let other_right_raycast = self.query_pipeline.cast_ray(
                        &self.rigid_body_set,
                        &self.collider_set,
                        &right_ray,
                        this_size.x,
                        false,
                        QueryFilter::default(),
                    );

                    let left_ray =
                        Ray::new([this_left, cur_position.y].into(), [-1.0, 0.0].as_na());

                    let this_left_raycast = self.query_pipeline.cast_ray(
                        &self.rigid_body_set,
                        &self.collider_set,
                        &left_ray,
                        this_size.x,
                        false,
                        QueryFilter::default(),
                    );

                    let no_collisions =
                        other_right_raycast.is_none() && this_left_raycast.is_none();

                    if no_collisions {
                        // NB these aren't correct/very thought out (`this` should move more here);
                        let new_other_pos =
                            other_pos.as_uv() + [this_left - other_left, 0.0].as_uv();
                        move_buf.push((*handle, new_other_pos));

                        return Some(cur_position);
                    }

                    /*
                    if let Some((collider, toi)) = other_right_raycast {
                        // is there anything to do?
                    } else {
                        let new_pos = other_pos.as_uv() + [this_left - other_left, 0.0].as_uv();
                        move_buf.push((*handle, new_pos));
                    }
                    */
                } else {
                    //
                    let right_ray =
                        Ray::new([this_right, cur_position.y].into(), [1.0, 0.0].as_na());

                    let this_right_raycast = self.query_pipeline.cast_ray(
                        &self.rigid_body_set,
                        &self.collider_set,
                        &right_ray,
                        this_size.x,
                        false,
                        QueryFilter::default(),
                    );

                    let left_ray =
                        Ray::new([other_left, cur_position.y].into(), [-1.0, 0.0].as_na());

                    let other_left_raycast = self.query_pipeline.cast_ray(
                        &self.rigid_body_set,
                        &self.collider_set,
                        &left_ray,
                        this_size.x,
                        false,
                        QueryFilter::default(),
                    );

                    let no_collisions =
                        this_right_raycast.is_none() && other_left_raycast.is_none();

                    if no_collisions {
                        let new_other_pos =
                            other_pos.as_uv() - [other_left - this_left, 0.0].as_uv();
                        // other_pos.as_uv() + [this_left - other_left, 0.0].as_uv();
                        move_buf.push((*handle, new_other_pos));

                        return Some(cur_position);
                    }
                }

                // TODO far from complete!
                if other_left > lim_left {
                    if other_left - lim_left > this_size.x {
                        // can add here
                        let x = other_left - this_size.x * 0.5;
                        let pos = [x, cur_position.y].as_uv();
                        return Some(pos);
                    }

                    // if other_right > lim_right {
                    //     lim_right = lim_right.min(other_left);
                    // }
                }

                if other_right < lim_right {
                    if lim_right - other_right > this_size.x {
                        // can add here
                        let x = other_right + this_size.x * 0.5;
                        let pos = [x, cur_position.y].as_uv();
                        return Some(pos);
                    }

                    // if other_left < lim_left {
                    //     lim_left = lim_left.max(other_right);
                    // }
                }
            }
        }

        Some(cur_position)
    }

    fn find_position_for_screen_rectangle(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        proposed_center: impl Into<[f32; 2]>,
        rect_size: impl Into<[f32; 2]>,
    ) -> Option<ultraviolet::Vec2> {
        let screen_world = viewport.screen_world_mat3();

        let center = proposed_center.as_uv();
        let size = rect_size.as_uv();

        // idk if this is right
        let screen_norm_size = size / viewport.canvas_size;
        let world_size =
            screen_norm_size * Vec2::new(viewport.view_size.x as f32, viewport.view_size.y as f32);

        let world_center = screen_world.transform_point2(center);

        // find height using heightmap
        let ground_screen_y = self
            .heightfields
            .project_screen_from_top(grid, viewport, center.x)?;
        // let ground_y = todo!();
        // let ground_y = self.heightfield_project_world_x(world_center.x + world_size.x * 0.5)?;

        let screen_center = viewport
            .world_screen_mat3()
            .transform_point2([world_center.x, 0.0].as_uv());

        let this_shape = Cuboid::new((size * 0.5).as_na());

        let mut intersecting = Vec::new();
        let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, ground_screen_y);
        // let mut cur_position = nalgebra::Isometry2::translation(screen_center.x, screen_center.y);

        let mut can_add = false;

        let mut iter_count = 0;
        const ITER_LIMIT: usize = 10;

        loop {
            let this_aabb = this_shape.aabb(&cur_position);

            // self.query_pipeline.intersections_with_shape(bodies, colliders, shape_pos, shape, filter, callback)
            intersecting.clear();
            self.query_pipeline
                .colliders_with_aabb_intersecting_aabb(&this_aabb, |other_handle| {
                    //
                    intersecting.push(*other_handle);
                    false
                });

            // {
            // let pos = (cur_position.translation.as_uv())
            //         painter.rect_stroke(
            //             egui::Rect::from_center_size(pos.as_epos2(), size.as_evec2()),
            //             0.0,
            //             egui::Stroke::new(1.0, egui::Color32::BLACK),
            //         );
            // }

            if intersecting.is_empty() {
                can_add = true;
                break;
            }

            for other_handle in &intersecting {
                let Some(other_aabb) = self
                    .collider_set
                    .get(*other_handle)
                    .map(|c| c.compute_aabb())
                else {
                    continue;
                };
                let overlap = aabb_overlap(&this_aabb, &other_aabb);
                if overlap.x.abs() > 0.0 && overlap.y.abs() > 0.0 {
                    cur_position.translation.y = other_aabb.center().y
                        - other_aabb.half_extents().y
                        - this_shape.half_extents.y
                        - 0.1;
                    //
                }
                //
            }

            if iter_count >= ITER_LIMIT {
                break;
            }
            iter_count += 1;
        }

        if can_add {
            Some(cur_position.translation.as_uv())
        } else {
            None
        }
    }

    /*
    fn handle_label_stack_and_swap(
        &mut self,
        grid: &AlignmentGrid,
    ) -> Vec<(ColliderHandle, ultraviolet::Vec2)> {
        /*

        step through the contact pairs from last frame/step, identifying
        which labels would be "better off" stacked on top of another label,
        or swapped with another label

        - if there's a horizontal contact

        --  compute the anchor position & (horizontal) forces that
        --   would result if the two labels had their positions
        --   swapped; if they'd both be lower, it's a swapping
        --   candidate, if only one would be lower, it may be a
        --   stacking candidate


        - if there's a vertical pair


        */

        // let mut swap_candidates: FxHashSet<(ColliderHandle, ColliderHandle)> = FxHashSet::default();

        // let mut covered_top: FxHashSet<ColliderHandle> = FxHashSet::default();
        // let mut stack_parents: FxHashMap<ColliderHandle, ColliderHandle> = FxHashMap::default();

        let mut label_move_buf: Vec<(ColliderHandle, ultraviolet::Vec2)> = Vec::new();

        // step through all (of last frame's) contacts, identifying
        // which labels can be stacked on which
        for pair in self.physics.narrow_phase.contact_pairs() {
            // need to track which labels
            // - *should* stack -- which labels would be "better off" sitting on top
            //     of one of its neighbors
            // - *can* be stacked *on* -- which labels have *no* contacts on top (?)

            // dbg!();
            let Some((manifold, contact)) = pair.find_deepest_contact() else {
                continue;
            };

            /*
            let covered1 = manifold.local_n1.y < 0.0;
            let covered2 = manifold.local_n2.y < 0.0;
            if covered1 {
                covered_top.insert(pair.collider1);
            }
            if covered2 {
                covered_top.insert(pair.collider2);
            }
            */

            let collider1 = self.collider_set.get(pair.collider1);
            let collider2 = self.collider_set.get(pair.collider2);

            // dbg!();
            let Some((collider1, collider2)) = collider1.zip(collider2) else {
                continue;
            };

            let annot1: super::AnnotationId = u128_usize_pair(collider1.user_data);
            let annot2: super::AnnotationId = u128_usize_pair(collider2.user_data);

            let get_anchor_and_data = |a| {
                self.annotations.get(&a).and_then(|a_data| {
                    let anchor = self.target_labels.anchor_screen_pos[a_data.target_label_ix]?;
                    let body_handle = self.target_labels.label_rigid_body[a_data.target_label_ix]?;
                    let rigid_body = self.rigid_body_set.get(body_handle)?;
                    Some((anchor, a_data, rigid_body))
                })
            };

            // dbg!();
            let Some(((anchor1, data1, body1), (anchor2, data2, body2))) =
                get_anchor_and_data(annot1).zip(get_anchor_and_data(annot2))
            else {
                continue;
            };

            // if both labels would experience lower "tension" wrt
            // their anchor (or at least neither would experience
            // higher), (consider) swap them
            let label_pos1 = body1.position().translation;
            let label_pos2 = body2.position().translation;

            // current
            let diff_a1b1 = anchor1.x - label_pos1.x;
            let diff_a2b2 = anchor2.x - label_pos2.x;

            // after swap
            let diff_a1b2 = anchor1.x - label_pos2.x;
            let diff_a2b1 = anchor2.x - label_pos1.x;

            let closer1 = diff_a1b1.abs() > diff_a1b2.abs();
            let closer2 = diff_a2b2.abs() > diff_a2b1.abs();

            // println!("before: [{diff_a1b1}, {diff_a2b2}]\t => [{diff_a1b2}, {diff_a2b1}]");

            if (closer1 && diff_a2b2.abs() >= diff_a2b1.abs())
                || (closer2 && diff_a1b1.abs() >= diff_a1b2.abs())
            {
                // println!("swapping!");
                // at least one would be closer, the other not further

                // if diff_a1b1 > diff_a1b2 && diff_a2b2 > diff_a2b1 {
                // both would be closer

                // should probably check whether it's possible to move them;
                // but whatever

                label_move_buf.push((pair.collider1, label_pos2.as_uv()));
                label_move_buf.push((pair.collider2, label_pos1.as_uv()));
            }
        }

        label_move_buf
    }
    */
}

fn choose_anchor_tile_for_world_x(
    grid: &AlignmentGrid,
    viewport: &Viewport,
    target: f64,
    label_width: f32,
) -> Option<(SeqId, SeqId)> {
    todo!();
}

#[derive(Default)]
pub struct AlignmentHeightFields {
    heightfields: FxHashMap<(SeqId, SeqId), LabelHeightField>,
}

impl AlignmentHeightFields {
    pub fn from_alignments(alignments: &Alignments) -> Self {
        let bin_count = 4096;

        let mut heightfields = FxHashMap::default();

        for (&tile, pair_alignments) in &alignments.pairs {
            // TODO fix this
            let alignment = &pair_alignments[0];
            let hfield = LabelHeightField::from_alignment(alignment, bin_count);
            heightfields.insert(tile, hfield);
        }

        Self { heightfields }
    }
}

impl AlignmentHeightFields {
    fn project_screen_from_top(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        screen_x: f32,
    ) -> Option<f32> {
        let mat = viewport.screen_world_dmat3();
        let world_x = mat.transform_point2([screen_x as f64, 0.0].as_duv()).x;

        let (qry_id, hfield, hit_world) =
            self.top_heightfield_in_visible_column(grid, viewport, world_x)?;
        let (tgt_id, norm_x) = grid.x_axis.global_to_axis_local(world_x)?;
        let screen = viewport.world_screen_dmat3().transform_point2(hit_world);

        Some(screen.y as f32)

        /*
        // might be more correct
        let tgt_range = &hfield.location.target_range;

        let x_in_range = {
            let l = hfield.location.target_total_len as f64;
            let s = tgt_range.start as f64 / l;
            let e = tgt_range.end as f64 / l;

            (norm_x - s) * (e - s)
        };

        let hfield_x = x_in_range * hfield.heightfield.scale().x as f64 * x_in_range;
        let hfield_y = hfield.heightfield_project_x(hfield_x as f32)?;
        */

        // let local_x =

        // need to shift `world_x` to account for the (intended) offset of the heightfield

        /*
        let offset = grid.x_axis.sequence_offset(tgt_id)? as f64;
        let hfield_y = hfield.heightfield_project_x((world_x - offset) as f32)?;

        let y_offset = grid.y_axis.sequence_offset(qry_id)? as f64;
        let world_y = y_offset + hfield_y as f64;
        let world = [world_x, world_y];

        let screen = viewport
            .world_screen_dmat3()
            .transform_point2(world.as_duv());

        Some(viewport.canvas_size.y as f32 - screen.y as f32)
        */

        // Some(screen.y as f32)
    }

    fn top_heightfield_in_visible_column(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        world_target: f64,
    ) -> Option<(SeqId, &LabelHeightField, ultraviolet::DVec2)> {
        // 0.35 is kind of arbitrary, but should keep labels from going too far up
        let top = (viewport.view_center.y + viewport.view_size.y * 0.35) as f32;
        // let top = grid.y_axis.total_len as f32;

        let ((tgt_id, qry_id), hit_world) =
            grid.cast_ray([world_target as f32, top], [0.0, -1.0], true)?;
        // grid.cast_ray([world_target as f32, 0.0], [0.0, 1.0])?;

        let hfield = self.heightfields.get(&(tgt_id, qry_id))?;
        // let qry_name = grid.sequence_names.get_by_right(&qry_id);
        // println!("using heightfield for query: {qry_id:?} ({qry_name:?})");

        Some((qry_id, hfield, hit_world))
    }

    /*
    fn left_heightfield_in_visible_row(
        &self,
        grid: &AlignmentGrid,
        viewport: &Viewport,
        world_query: f64,
    ) -> Option<(SeqId, &LabelHeightField)> {
        let ((tgt_id, qry_id), _hit_world) =
            grid.cast_ray([0.0, world_query as f32], [1.0, 0.0])?;

        let hfield = self.heightfields.get(&(tgt_id, qry_id))?;

        Some((qry_id, hfield))
    }
    */
}

struct LabelHeightField {
    target_id: SeqId,
    query_id: SeqId,
    heightfield: HeightField,
    location: AlignmentLocation,
}

impl LabelHeightField {
    fn from_alignment(alignment: &Alignment, bin_count: usize) -> Self {
        let location = alignment.location.clone();
        let target_id = alignment.target_id;
        let query_id = alignment.query_id;

        let bin_size = location.aligned_target_len() as f32 / bin_count as f32;

        let mut bins = vec![0.0; bin_count];

        let mut y_max = f32::NEG_INFINITY;

        // TODO this can be done faster

        let mut bin_ix = 0;
        let mut current_bin_end = bin_size.floor() as usize;

        let line_vertices =
            line_vertices_from_cigar(&alignment.location, alignment.cigar.whole_cigar());

        for &[p0, p1] in &line_vertices {
            // shouldn't happen, but
            if bin_ix >= bins.len() {
                break;
            }
            let mid = (p0 + p1) * 0.5;
            let x = mid.x;
            let y = mid.y;

            let yf = y as f32;
            if bins[bin_ix] > yf {
                bins[bin_ix] = yf;
                y_max = y_max.max(yf);
            }

            if x as usize >= current_bin_end {
                bin_ix += 1;
                current_bin_end = ((bin_ix + 1) as f32 * bin_size).floor() as usize;
            }
        }

        let scale_x = location.aligned_target_len() as f32;
        let heights = nalgebra::DVector::from_vec(bins);
        let heightfield = HeightField::new(heights, [scale_x, 1.0].into());

        Self {
            heightfield,
            target_id,
            query_id,
            location,
        }
    }
}

impl LabelHeightField {
    fn heightfield_aabb(&self) -> Aabb {
        let pos = nalgebra::Isometry2::translation(self.heightfield.scale().x * 0.5, 0.0);
        self.heightfield.compute_aabb(&pos)
    }

    // input and output are in "heightfield-local" (but unnormalized) coordinates
    fn heightfield_project_x(&self, x: f32) -> Option<f32> {
        if x < 0.0 || x > self.heightfield.scale().x {
            println!(
                "out of scale: {x} not in [0.0, {}]",
                self.heightfield.scale().x
            );
            return None;
        }
        let x_ = x - self.heightfield.scale().x * 0.5;

        let point = nalgebra::Point2::new(x_, 0.0);

        // NB: parry's height_at_point() implementation is bugged (offsets by the segment's y)
        let inter_y = self
            .heightfield
            .cell_at_point(&point)
            .and_then(|cell| self.heightfield.segment_at(cell))
            .map(|seg| {
                rapier2d::parry::query::details::closest_points_line_line_parameters(
                    &seg.a,
                    &seg.scaled_direction(),
                    &point,
                    &Vector::y(),
                )
                .1
            });

        inter_y
    }

    /*
    fn heightfield_project_screen(
        &self,
        viewport: &Viewport,
        // screen_point: impl Into<[f32; 2]>,
        screen_x: f32,
    ) -> Option<Vec2> {
        // let pt = screen_point.as_uv();
        let world = viewport
            .screen_world_mat3()
            .transform_point2([screen_x, 0.0].as_uv());
        let y = self.heightfield_project_x(world.x)?;
        Some([world.x, y].as_uv())
    }
    */

    fn heightfield_screen_segments(
        &self,
        viewport: &Viewport,
    ) -> impl Iterator<Item = [egui::Pos2; 2]> + '_ {
        let aabb = self.heightfield_aabb();
        let offset = [aabb.half_extents().x, 0.0].as_uv();
        let mat = viewport.world_screen_mat3();

        self.heightfield.segments().map(move |Segment { a, b }| {
            let pa = mat.transform_point2(a.as_uv() + offset);
            let pb = mat.transform_point2(b.as_uv() + offset);
            [pa.as_epos2(), pb.as_epos2()]
        })
    }

    // fn height_at_target(&self, align_tgt_pos: f64) -> Option<f64> {
    //     todo!();
    // }

    // fn height_at_query(&self, align_qry_pos: f64) -> Option<f64> {
    //     todo!();
    // }
}

#[derive(Debug, Clone, PartialEq)]
enum AnnotationRange {
    // Symmetric {
    //     world_range: std::ops::RangeInclusive<f64>,
    // },
    Both {
        target_range: std::ops::RangeInclusive<f64>,
        query_range: std::ops::RangeInclusive<f64>,
    },
    TargetOnly(std::ops::RangeInclusive<f64>),
    QueryOnly(std::ops::RangeInclusive<f64>),
}

impl AnnotationRange {
    fn target_range(&self) -> Option<&std::ops::RangeInclusive<f64>> {
        match self {
            // AnnotationRange::Symmetric { world_range } => Some(world_range),
            AnnotationRange::Both { target_range, .. } => Some(target_range),
            AnnotationRange::TargetOnly(range) => Some(range),
            AnnotationRange::QueryOnly(_) => None,
        }
    }

    fn query_range(&self) -> Option<&std::ops::RangeInclusive<f64>> {
        match self {
            // AnnotationRange::Symmetric { world_range } => Some(world_range),
            AnnotationRange::Both { query_range, .. } => Some(query_range),
            AnnotationRange::TargetOnly(_) => None,
            AnnotationRange::QueryOnly(range) => Some(range),
        }
    }

    fn new(
        target_range: Option<std::ops::RangeInclusive<f64>>,
        query_range: Option<std::ops::RangeInclusive<f64>>,
    ) -> Option<Self> {
        match (target_range, query_range) {
            (None, None) => None,
            (None, Some(range)) => Some(Self::QueryOnly(range)),
            (Some(range), None) => Some(Self::TargetOnly(range)),
            (Some(target_range), Some(query_range)) => Some(Self::Both {
                target_range,
                query_range,
            }),
        }
    }
}

fn aabb_overlap(
    aabb1: &rapier2d::geometry::Aabb,
    aabb2: &rapier2d::geometry::Aabb,
) -> nalgebra::Vector2<f32> {
    let center1 = aabb1.center();
    let center2 = aabb2.center();

    let half_extents1 = aabb1.half_extents();
    let half_extents2 = aabb2.half_extents();

    let overlap_x = (half_extents1.x + half_extents2.x) - (center2.x - center1.x).abs();
    let overlap_y = (half_extents1.y + half_extents2.y) - (center2.y - center1.y).abs();

    let sign_x = (center2.x - center1.x).signum();
    let sign_y = (center2.y - center1.y).signum();

    nalgebra::Vector2::new(overlap_x * sign_x, overlap_y * sign_y)
}

struct Physics {
    gravity: nalgebra::Vector2<f32>,
    physics_pipeline: PhysicsPipeline,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
}

impl std::default::Default for Physics {
    fn default() -> Self {
        Self {
            gravity: vector![0.0, 0.0],
            // gravity: vector![0.0, 9.81],
            // gravity: vector![0.0, 20.0],
            // gravity: vector![0.0, 8.0],
            physics_pipeline: PhysicsPipeline::default(),
            integration_parameters: IntegrationParameters::default(),
            island_manager: IslandManager::default(),
            broad_phase: DefaultBroadPhase::default(),
            narrow_phase: NarrowPhase::default(),
            impulse_joint_set: ImpulseJointSet::default(),
            multibody_joint_set: MultibodyJointSet::default(),
            ccd_solver: CCDSolver::default(),
        }
    }
}

impl Physics {
    fn step(
        &mut self,
        rigid_bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        query_pipeline: &mut QueryPipeline,
        // physics_hooks: Option<&dyn PhysicsHooks>,
        // event_handler: Option<&dyn EventHandler>,
    ) {
        let physics_hooks = ();
        let event_handler = ();

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            rigid_bodies,
            colliders,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(query_pipeline),
            &physics_hooks,
            &event_handler,
        );
    }
}

/*
fn label_collider(annot_id: super::AnnotationId, size: impl Into<[f32; 2]>) -> ColliderBuilder {
    let size = size.into();
    ColliderBuilder::cuboid(size[0] * 0.5, size[1] * 0.5)
        .mass(1.0)
        .friction(0.0)
        .user_data(usize_pair_u128(annot_id))
}
*/

// fn label_rigid_body(

fn label_collider_body(
    annot_id: super::AnnotationId,
    size: impl Into<[f32; 2]>,
) -> (ColliderBuilder, RigidBodyBuilder) {
    let size = size.into();
    let collider = ColliderBuilder::cuboid(size[0] * 0.5, size[1] * 0.5)
        .mass(1.0)
        .friction(0.0)
        .user_data(usize_pair_u128(annot_id));

    // let collider = label_collider(annot_id, size);
    let rigid_body = RigidBodyBuilder::dynamic()
        .enabled(false)
        .lock_rotations()
        .linear_damping(3.0);
    (collider, rigid_body)
}

pub(crate) fn line_vertices_from_cigar(
    location: &crate::paf::AlignmentLocation,
    cigar_ops: impl Iterator<Item = (crate::cigar::CigarOp, u32)>,
) -> Vec<[ultraviolet::DVec2; 2]> {
    use crate::cigar::CigarOp;
    use ultraviolet::DVec2;

    let mut vertices = Vec::new();

    let mut tgt_cg = 0;
    let mut qry_cg = 0;

    for (op, count) in cigar_ops {
        // tgt_cg and qry_cg are offsets from the start of the cigar
        let tgt_start = tgt_cg;
        let qry_start = qry_cg;

        let (tgt_end, qry_end) = match op {
            CigarOp::Eq | CigarOp::X | CigarOp::M => {
                tgt_cg += count as u64;
                qry_cg += count as u64;
                //
                (tgt_start + count as u64, qry_start + count as u64)
            }
            CigarOp::I => {
                qry_cg += count as u64;
                //
                (tgt_start, qry_start + count as u64)
            }
            CigarOp::D => {
                tgt_cg += count as u64;
                //
                (tgt_start + count as u64, qry_start)
            }
        };

        let tgt_range = location.map_from_aligned_target_range(tgt_start..tgt_end);
        let qry_range = location.map_from_aligned_query_range(qry_start..qry_end);

        let mut from = DVec2::new(tgt_range.start as f64, qry_range.start as f64);
        let mut to = DVec2::new(tgt_range.end as f64, qry_range.end as f64);

        if location.query_strand.is_rev() {
            std::mem::swap(&mut from.y, &mut to.y);
        }

        vertices.push([from, to]);
    }

    vertices
}
