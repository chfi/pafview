use bevy::{math::DVec2, prelude::*, utils::HashMap};

// use rapier2d::parry;
use avian2d::parry::{
    self,
    bounding_volume::{Aabb, BoundingVolume},
};

use crate::{app::SequencePairTile, sequences::SeqId};

pub struct AlignmentLayoutPlugin;

impl Plugin for AlignmentLayoutPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SeqPairLayout>()
            .add_event::<LayoutChangedEvent>()
            .add_plugins(editor::AlignmentLayoutGuiPlugin);
    }
}

#[derive(Asset, Clone, TypePath)]
pub struct SeqPairLayout {
    pub aabbs: HashMap<SequencePairTile, Aabb>,

    pub layout_qbvh: AabbQbvh<SequencePairTile>,

    pub mins: DVec2,
    pub maxs: DVec2,

    // TODO: this will have to change if/when arbitrarily placed tile layouts
    // are enabled again
    pub target_offsets: HashMap<SeqId, f64>,
    pub query_offsets: HashMap<SeqId, f64>,

    // TODO: don't really like storing these like this, but for now used to compute
    // which grid material to use for the sequence pair tiles
    pub target_edges: [SeqId; 2],
    pub query_edges: [SeqId; 2],
}

#[derive(Resource, Clone)]
pub struct DefaultLayout {
    // pub layout: SeqPairLayout,
    pub layout: Handle<SeqPairLayout>,
    builder: LayoutBuilder,
}

#[derive(Component, Default, Deref, DerefMut)]
pub struct LayoutEntityIndex(pub HashMap<SequencePairTile, Entity>);

impl DefaultLayout {
    pub fn new(layout: Handle<SeqPairLayout>, builder: LayoutBuilder) -> Self {
        Self { layout, builder }
    }
    pub fn builder(&self) -> &LayoutBuilder {
        &self.builder
    }
}

#[derive(PartialEq, Clone, Reflect)]
pub struct LayoutBuilder {
    /// stack tiles using this uniform (cumulative) offset, allowing overlap,
    /// instead of lining them up side-by-side
    /// ignored if builder was created from tile positions
    pub vertical_offset: Option<f64>,
    pub vertical_limit: Option<f64>,

    /// see vertical_offset
    pub horizontal_offset: Option<f64>,
    pub horizontal_limit: Option<f64>,

    pub query_total: u64,
    pub target_total: u64,

    data: LayoutInput,
}

impl LayoutBuilder {
    pub fn build(self, sequences: &crate::Sequences) -> SeqPairLayout {
        let mut mins = DVec2::INFINITY;
        let mut maxs = DVec2::NEG_INFINITY;

        let mut target_offsets = HashMap::default();
        let mut query_offsets = HashMap::default();

        let mut left_edge: Option<SeqId> = None;
        let mut right_edge: Option<SeqId> = None;

        let mut top_edge: Option<SeqId> = None;
        let mut bottom_edge: Option<SeqId> = None;

        let aabbs = match self.data {
            LayoutInput::Axes { targets, queries } => {
                let sum_axis = |seqs: &[SeqId]| -> u64 {
                    seqs.iter()
                        .filter_map(|s| {
                            let seq = sequences.get(*s)?;
                            Some(seq.len())
                        })
                        .sum()
                };

                let total_target_len = sum_axis(&targets);
                let total_query_len = sum_axis(&queries);

                let h_limit = self.horizontal_limit.unwrap_or(total_target_len as f64);
                let v_limit = self.vertical_limit.unwrap_or(total_query_len as f64);

                let h_prop = h_limit / total_target_len as f64;
                let v_prop = v_limit / total_query_len as f64;

                let mut aabbs = HashMap::default();

                let mut x_offset = 0.0;

                for target in targets {
                    let Some(tgt_seq) = sequences.get(target) else {
                        continue;
                    };

                    if left_edge.is_none() {
                        left_edge = Some(target);
                    }
                    right_edge = Some(target);

                    let tgt_len = tgt_seq.len() as f64;

                    let x0 = x_offset;

                    // let h_offset = self.horizontal_limit.unwrap_or(total_target_len as f64)

                    x_offset += tgt_len * h_prop;
                    // x_offset += tgt_len / h_limit;

                    // x_offset += self.horizontal_offset.unwrap_or(tgt_len);

                    let mut y_offset = 0.0;

                    target_offsets.insert(target, x0);

                    for &query in &queries {
                        let Some(qry_seq) = sequences.get(query) else {
                            continue;
                        };

                        if top_edge.is_none() {
                            top_edge = Some(query);
                        }
                        bottom_edge = Some(query);

                        let qry_len = qry_seq.len() as f64;

                        let y0 = y_offset;

                        query_offsets.insert(target, y0);

                        y_offset += qry_len * v_prop;

                        // y_offset += self.vertical_offset.unwrap_or(qry_len);

                        // let center = [x0, y0];
                        let center = [x0 + tgt_len * 0.5, y0 + qry_len * 0.5];

                        let half_extents = [tgt_len * 0.5, qry_len * 0.5];

                        let aabb = Aabb::from_half_extents(center.into(), half_extents.into());
                        let max = aabb.maxs;
                        let min = aabb.mins;

                        mins = mins.min(DVec2::new(min.x, min.y));
                        maxs = maxs.max(DVec2::new(max.x, max.y));

                        aabbs.insert(SequencePairTile { target, query }, aabb);
                    }
                }

                aabbs
            } // LayoutInput::TilePositions { offsets } => offsets
              //     .iter()
              //     .filter_map(|(&seq_pair, &offset)| {
              //         let tgt_len = sequences.get(seq_pair.target)?.len() as f64;
              //         let qry_len = sequences.get(seq_pair.query)?.len() as f64;
              //         let half_extents = [tgt_len * 0.5, qry_len * 0.5];
              //         let aabb =
              //             Aabb::from_half_extents([offset.x, offset.y].into(), half_extents.into());

              //         let max = aabb.maxs;
              //         let min = aabb.mins;

              //         mins = mins.min(DVec2::new(min.x, min.y));
              //         maxs = maxs.max(DVec2::new(max.x, max.y));

              //         Some((seq_pair, aabb))
              //     })
              //     .collect(),
        };

        let layout_qbvh = AabbQbvh::from_aabbs(aabbs.iter().map(|(&sp, &aabb)| (sp, aabb)));

        let target_edges = left_edge.zip(right_edge).map(|(l, r)| [l, r]).unwrap();
        let query_edges = top_edge.zip(bottom_edge).map(|(l, r)| [l, r]).unwrap();

        SeqPairLayout {
            aabbs,
            layout_qbvh,
            mins,
            maxs,

            target_offsets,
            query_offsets,

            target_edges,
            query_edges,
        }
    }

    pub fn from_axes<T, Q>(sequences: &crate::Sequences, targets: T, queries: Q) -> Self
    where
        T: IntoIterator<Item = SeqId>,
        Q: IntoIterator<Item = SeqId>,
    {
        let mut target_total = 0;
        let mut query_total = 0;

        let targets = targets
            .into_iter()
            .inspect(|s| {
                if let Some(seq) = sequences.get(*s) {
                    target_total += seq.len();
                }
            })
            .collect::<Vec<_>>();
        let queries = queries
            .into_iter()
            .inspect(|s| {
                if let Some(seq) = sequences.get(*s) {
                    query_total += seq.len();
                }
            })
            .collect::<Vec<_>>();

        println!("axis lengths: {target_total}, {query_total}");

        let data = LayoutInput::Axes { targets, queries };
        Self {
            vertical_offset: None,
            vertical_limit: Some(query_total as f64),
            horizontal_offset: None,
            horizontal_limit: Some(target_total as f64),
            query_total,
            target_total,
            data,
        }
    }

    /*
    pub fn from_positions<P: Into<DVec2>>(
        tile_positions: impl IntoIterator<Item = (SequencePairTile, P)>,
    ) -> Self {


        let offsets = tile_positions
            .into_iter()
            .map(|(t, p)| (t, p.into()))
            .collect();

        Self {
            vertical_offset: None,
            vertical_limit: None,
            horizontal_offset: None,
            horizontal_limit: None,
            data: LayoutInput::TilePositions { offsets },
        }
    }
    */

    pub fn with_vertical_offset(mut self, offset: Option<f64>) -> Self {
        self.vertical_offset = offset;
        self
    }

    pub fn with_horizontal_offset(mut self, offset: Option<f64>) -> Self {
        self.horizontal_offset = offset;
        self
    }
}

#[derive(PartialEq, Clone, Reflect)]
enum LayoutInput {
    Axes {
        targets: Vec<SeqId>,
        queries: Vec<SeqId>,
    },
    // NB: disabled to ensure that annotated regions are kept simple
    // TilePositions {
    //     offsets: HashMap<SequencePairTile, DVec2>,
    // },
}

#[derive(Clone)]
pub struct AabbQbvh<Data: Copy> {
    qbvh: parry::partitioning::Qbvh<usize>,
    data: Vec<Data>,
    aabbs: Vec<Aabb>,
}

impl<T: Copy> AabbQbvh<T> {
    pub fn from_aabbs<I>(tiles: I) -> Self
    where
        I: Iterator<Item = (T, Aabb)>,
    {
        use parry::partitioning::Qbvh;

        let (data, leaf_data): (Vec<T>, Vec<(usize, parry::bounding_volume::Aabb)>) = tiles
            .enumerate()
            .map(|(ix, (seq_pair, aabb))| (seq_pair, (ix, aabb)))
            .unzip();

        let aabbs = leaf_data.iter().map(|(_, aabb)| *aabb).collect::<Vec<_>>();
        let mut qbvh = Qbvh::new();
        qbvh.clear_and_rebuild(leaf_data.into_iter(), 1.0);

        Self { qbvh, data, aabbs }
    }

    pub fn aabbs_in_rect_callback(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
        mut callback: impl FnMut(T, &Aabb) -> bool,
    ) {
        let center = center.into();
        let half_extents = half_extents.into();

        let query_aabb =
            parry::bounding_volume::Aabb::from_half_extents(center.into(), half_extents.into());

        let leaf_cb = &mut |index: &usize| {
            let aabb = &self.aabbs[*index];
            if query_aabb.intersects(aabb) {
                let value = self.data[*index];
                callback(value, aabb)
            } else {
                true
            }
        };

        let mut visitor =
            parry::query::visitors::BoundingVolumeIntersectionsVisitor::new(&query_aabb, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn aabbs_in_rect(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
    ) -> Vec<T> {
        let mut results = Vec::new();

        self.aabbs_in_rect_callback(center, half_extents, |value, _| {
            results.push(value);
            true
        });

        results
    }

    pub fn aabbs_at_point_callback(
        &self,
        point: impl Into<[f64; 2]>,
        mut callback: impl FnMut(T) -> bool,
    ) {
        let query_pt = point.into();

        let leaf_cb = &mut |index: &usize| {
            let aabb = &self.aabbs[*index];
            if aabb.contains_local_point(&query_pt.into()) {
                let value = self.data[*index];
                callback(value)
            } else {
                true
            }
        };

        let query_pt = query_pt.into();
        let mut visitor =
            parry::query::visitors::PointIntersectionsVisitor::new(&query_pt, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn aabbs_at_point(&self, point: impl Into<[f64; 2]>) -> Vec<T> {
        let mut results = Vec::new();

        self.aabbs_at_point_callback(point, |value| {
            results.push(value);
            true
        });

        results
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Event, Reflect)]
pub struct LayoutChangedEvent {
    pub entity: Entity,
    pub need_respawn: bool,
}

pub mod editor {

    use crate::app::{
        alignments::AlignmentLayoutQuery, input::cursor::CursorPosition, view::AlignmentViewport,
    };

    use super::*;
    use bevy::{render::view::RenderLayers, sprite::Mesh2dHandle};
    use bevy_egui::EguiContexts;
    use bevy_mod_picking::{
        backend,
        pointer::{PointerId, PointerLocation},
        prelude::*,
    };
    use events::{send_click_and_drag_events, DragMap};

    pub struct AlignmentLayoutGuiPlugin;

    impl Plugin for AlignmentLayoutGuiPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(LayoutEditorOpen(false))
                .init_resource::<LayoutEditor>()
                // .add_systems(
                //     PreUpdate,
                //     show_live_layout_editor.after(bevy_egui::EguiSet::BeginPass),
                // );
                .add_systems(
                    PreUpdate,
                    (
                        prepare_layout_gizmos,
                        update_layout_gizmos.after(send_click_and_drag_events),
                    )
                        .chain(),
                )
                .add_systems(
                    PreUpdate,
                    (prepare_layout_editor, layout_config_editor)
                        .chain()
                        .after(bevy_egui::EguiSet::BeginPass),
                )
                .add_systems(
                    PreUpdate,
                    layout_gizmo_picker.in_set(bevy_mod_picking::picking_core::PickSet::Backend),
                    // )
                    // .add_systems(
                    //     PreUpdate,
                    //     drag_gizmos
                    //         .after(update_layout_gizmos)
                    //         .after(send_click_and_drag_events),
                );
        }
    }

    #[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Reflect)]
    pub struct LayoutEditorOpen(pub bool);

    #[derive(Default, PartialEq, Clone)]
    struct LayoutEditorState {
        custom_vertical_limit: Option<f64>,
        custom_horizontal_limit: Option<f64>,
        // vertical_limit_on: bool,
        // vertical_limit: f64,

        // horizontal_limit_on: bool,
        // horizontal_limit: f64,
    }

    // #[derive(Resource, Default, Clone, PartialEq)]
    // pub struct LiveLayoutBuilder {
    //     builder: Option<LayoutBuilder>,
    // }

    #[derive(Resource, Default)]
    struct LayoutEditor {
        // layout root entity to modify; if None, modify the default layout
        target_layout_entity: Option<Entity>,
        builder: Option<LayoutBuilder>,
        state: LayoutEditorState,

        enable_drag_gizmos: bool,
    }

    // #[derive(Resource)]
    // struct LayoutDragGizmos {
    //     horizontal: Entity,
    //     vertical: Entity,
    // }

    #[derive(Component)]
    struct VerticalDragGizmo;

    #[derive(Component)]
    struct HorizontalDragGizmo;

    #[derive(Component, Clone)]
    struct BeingDragged;

    fn prepare_layout_gizmos(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
        let rect_mesh = Mesh2dHandle(meshes.add(Rectangle::from_length(1.0)));
        let material = materials.add(Color::from(LinearRgba::rgb(1.0, 0.0, 0.0)));

        let bundle = (
            RenderLayers::layer(1),
            rect_mesh.clone(),
            material.clone(),
            SpatialBundle {
                visibility: Visibility::Hidden,
                ..default()
            },
        );

        commands.spawn(bundle.clone()).insert((
            VerticalDragGizmo,
            PickableBundle::default(),
            On::<Pointer<DragStart>>::target_insert(BeingDragged),
            On::<Pointer<DragEnd>>::target_remove::<BeingDragged>(),
            // On::<Pointer<Over>>::run(|| {
            // println!("hovering vertical gizmo");
            // }),
        ));
        commands.spawn(bundle.clone()).insert((
            HorizontalDragGizmo,
            PickableBundle::default(),
            On::<Pointer<DragStart>>::target_insert(BeingDragged),
            On::<Pointer<DragEnd>>::target_remove::<BeingDragged>(),
        ));
    }

    fn layout_gizmo_picker(
        pointers: Query<(&PointerId, &PointerLocation)>,
        camera: Query<(Entity, &Camera), With<crate::app::ScreenspaceCamera>>,
        editor: Res<LayoutEditor>,
        drag_gizmos: Query<
            (Entity, &Transform),
            Or<(With<VerticalDragGizmo>, With<HorizontalDragGizmo>)>,
        >,

        mut pointer_hits: EventWriter<backend::PointerHits>,
    ) {
        if !editor.enable_drag_gizmos {
            return;
        }

        let Ok((camera_ent, _camera)) = camera.get_single() else {
            return;
        };

        for (gizmo_ent, transform) in drag_gizmos.iter() {
            let gizmo_aabb = Aabb::from_half_extents(
                transform.translation.xy().as_dvec2().to_array().into(),
                (transform.scale.xy().as_dvec2() * 0.5).to_array().into(),
            );

            for (ptr_id, ptr_loc) in pointers.iter() {
                let Some(loc) = ptr_loc.location() else {
                    continue;
                };

                if gizmo_aabb
                    .contains_local_point(&[loc.position.x as f64, loc.position.y as f64].into())
                {
                    let hit_data = backend::HitData::new(
                        camera_ent,
                        100.0,
                        Some(Vec3::new(loc.position.x, loc.position.y as f32, 100.0)),
                        None,
                    );
                    pointer_hits.send(backend::PointerHits::new(
                        *ptr_id,
                        vec![(gizmo_ent, hit_data)],
                        1.0,
                    ));
                }
            }
        }
    }

    fn update_layout_gizmos(
        editor: Res<LayoutEditor>,

        cursor: Res<CursorPosition>,
        // drag_map: Res<DragMap>,
        mut drag_gizmos: Query<
            (
                Entity,
                &mut Transform,
                &mut Visibility,
                Has<BeingDragged>,
                Has<VerticalDragGizmo>,
                Has<HorizontalDragGizmo>,
            ),
            (
                Or<(With<VerticalDragGizmo>, With<HorizontalDragGizmo>)>,
                Without<Handle<SeqPairLayout>>,
            ),
        >,

        layouts: AlignmentLayoutQuery,

        view: Res<AlignmentViewport>,
        camera: Query<&Camera, With<crate::app::ScreenspaceCamera>>,
        window: Query<&Window>,
    ) {
        let Ok(_camera) = camera.get_single() else {
            return;
        };

        let Ok(screen_dims) = window.get_single().map(|w| w.size()) else {
            return;
        };

        let Some(layout) = layouts.layout_assets.get(&layouts.default_layout.layout) else {
            return;
        };

        for (_gizmo_ent, mut transform, mut visibility, is_dragged, is_vert, is_horiz) in
            drag_gizmos.iter_mut()
        {
            if editor.enable_drag_gizmos {
                *visibility = Visibility::Visible;
            } else {
                *visibility = Visibility::Hidden;
            }

            let mins = view.view.map_world_to_screen(screen_dims, layout.mins);
            let maxs = view.view.map_world_to_screen(screen_dims, layout.maxs);

            if is_vert {
                let x = (mins.x + maxs.x) * 0.5;
                let mut y = maxs.y;

                if is_dragged {
                    if let Some(cursor) = cursor.screen {
                        y = cursor.y;
                    }
                }

                transform.translation = Vec3::new(x as f32, screen_dims.y - y as f32, 100.0);
                transform.scale = Vec3::new((maxs.x - mins.x) as f32, 2.0, 1.0);
            } else if is_horiz {
                let mut x = maxs.x;
                let y = (mins.y + maxs.y) * 0.5;

                if is_dragged {
                    if let Some(cursor) = cursor.screen {
                        x = cursor.x;
                    }
                }

                transform.translation = Vec3::new(x as f32, screen_dims.y - y as f32, 100.0);
                transform.scale = Vec3::new(2.0, (maxs.y - mins.y) as f32, 1.0);
            }
        }
    }

    // fn block_pan_action(
    //     mut user_actions: ResMut<ActionState<UserAction>>,
    //     dragged: Query<&BeingDragged>,
    // ) {
    //     if !dragged.is_empty() {
    //         // let action = UserAction::
    //     }
    // }

    fn prepare_layout_editor(mut editor: ResMut<LayoutEditor>, default_layout: Res<DefaultLayout>) {
        if editor.builder.is_none() {
            editor.builder = Some(default_layout.builder().clone());
        }
    }

    fn layout_config_editor(
        mut editor: ResMut<LayoutEditor>,
        mut contexts: EguiContexts,
        mut editor_open: ResMut<LayoutEditorOpen>,

        mut state: Local<LayoutEditorState>,
    ) {
        let ctx = contexts.ctx_mut();

        *state = editor.state.clone();

        let Some(builder) = editor.builder.as_ref() else {
            return;
        };

        let mut enable_drag_gizmos = editor.enable_drag_gizmos;

        egui::Window::new("Layout Editor")
            .open(&mut editor_open.0)
            .show(ctx, |ui| {
                //
                ui.vertical(|ui| {
                    ui.checkbox(&mut enable_drag_gizmos, "Enable layout gizmos");

                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Vertical limit");

                        let mut limit_on = state.custom_vertical_limit.is_some();
                        ui.checkbox(&mut limit_on, "Enable");

                        if limit_on && state.custom_vertical_limit.is_none() {
                            state.custom_vertical_limit = Some(builder.query_total as f64);
                        }

                        ui.add_enabled(
                            limit_on,
                            egui::DragValue::from_get_set(|val: Option<f64>| -> f64 {
                                if let Some(limit) = state.custom_vertical_limit.as_mut() {
                                    if let Some(val) = val {
                                        *limit = val;
                                    }
                                    *limit
                                } else {
                                    if let Some(val) = val {
                                        state.custom_vertical_limit = Some(val);
                                        val
                                    } else {
                                        builder.query_total as f64
                                    }
                                }
                            }),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Horizontal offset");

                        let mut limit_on = state.custom_horizontal_limit.is_some();
                        ui.checkbox(&mut limit_on, "Enable");

                        if limit_on && state.custom_horizontal_limit.is_none() {
                            state.custom_horizontal_limit = Some(builder.target_total as f64);
                        }

                        ui.add_enabled(
                            limit_on,
                            egui::DragValue::from_get_set(|val: Option<f64>| -> f64 {
                                if let Some(limit) = state.custom_horizontal_limit.as_mut() {
                                    if let Some(val) = val {
                                        *limit = val;
                                    }
                                    *limit
                                } else {
                                    if let Some(val) = val {
                                        state.custom_horizontal_limit = Some(val);

                                        val
                                    } else {
                                        builder.target_total as f64
                                    }
                                }
                            }),
                        );
                        /*
                        ui.checkbox(&mut editor_state.horizontal_limit_on, "Enable");
                        ui.add_enabled(
                            editor_state.horizontal_limit_on,
                            egui::DragValue::new(&mut editor_state.horizontal_limit),
                        );
                        */
                    });

                    //
                });
            });

        editor.enable_drag_gizmos = enable_drag_gizmos;

        if *state != editor.state {
            // apply changes
            editor.state = state.clone();
        }
    }

    /*
    fn show_live_layout_editor(
        mut contexts: EguiContexts,
        mut editor_open: ResMut<LayoutEditorOpen>,
        mut builder: ResMut<LiveLayoutBuilder>,

        mut layouts: ResMut<Assets<SeqPairLayout>>,
        mut default_layout: ResMut<DefaultLayout>,
        // mut layout_assets: ResMut<Assets<SeqPairLayout>>,
        sequences: Res<crate::Sequences>,

        mut editor_state: Local<LayoutEditorState>,

        default_layout_root: Res<crate::app::alignments::DefaultLayoutRoot>,
        mut layout_events: EventWriter<LayoutChangedEvent>,
        mut update_layout_debounce: Local<Option<std::time::Instant>>,
    ) {
        let init_builder = builder.bypass_change_detection().builder.is_none();

        if init_builder {
            builder.builder = Some(default_layout.builder().clone());
        }

        let ctx = contexts.ctx_mut();

        egui::Window::new("Layout Editor")
            .open(&mut editor_open.0)
            .show(ctx, |ui| {
                //
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Vertical limit");
                        ui.checkbox(&mut editor_state.vertical_limit_on, "Enable");
                        ui.add_enabled(
                            editor_state.vertical_limit_on,
                            egui::DragValue::new(&mut editor_state.vertical_limit),
                        );
                        // if ui.button("Reset").clicked() {
                        // }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Horizontal offset");
                        ui.checkbox(&mut editor_state.horizontal_limit_on, "Enable");
                        ui.add_enabled(
                            editor_state.horizontal_limit_on,
                            egui::DragValue::new(&mut editor_state.horizontal_limit),
                        );
                    });

                    //
                });
            });

        let mut builder_changed = false;

        let builder_state: Option<LayoutEditorState> = builder
            .bypass_change_detection()
            .builder
            .as_ref()
            .map(|b| LayoutEditorState {
                vertical_limit_on: b.vertical_limit.is_some(),
                vertical_limit: b.query_total as f64,
                horizontal_limit_on: b.horizontal_limit.is_some(),
                horizontal_limit: b.target_total as f64,
            });

        {
            let editor_state: &LayoutEditorState = &editor_state;
            if builder_state.map(|s| &s != editor_state).unwrap_or(false) {
                if let Some(builder) = builder.builder.as_mut() {
                    if editor_state.vertical_limit_on {
                        if Some(editor_state.vertical_limit) != builder.vertical_limit {
                            builder.vertical_limit = Some(editor_state.vertical_limit);
                            builder_changed = true;
                        }
                    } else {
                        if builder.vertical_limit.is_some() {
                            builder.vertical_limit = None;
                            // builder.vertical_limit = Some(builder.query_total as f64);
                            builder_changed = true;
                        }
                    }

                    if editor_state.horizontal_limit_on {
                        if Some(editor_state.horizontal_limit) != builder.horizontal_offset {
                            builder.horizontal_offset = Some(editor_state.horizontal_limit);
                            builder_changed = true;
                        }
                    } else if builder.horizontal_limit.is_some() {
                        builder.horizontal_limit = None;
                        builder_changed = true;
                    }
                }
            }
        }

        if builder_changed {
            let builder = builder.builder.as_mut();
            let layout = layouts.get_mut(&default_layout.layout);
            if let Some((builder, layout)) = builder.zip(layout) {
                *layout = builder.clone().build(&sequences);
                default_layout.builder = builder.clone();
            }
            *update_layout_debounce = Some(std::time::Instant::now());
        }

        if let Some(time) = update_layout_debounce.take() {
            if time.elapsed().as_millis() < 100 {
                *update_layout_debounce = Some(time);
            } else {
                layout_events.send(LayoutChangedEvent {
                    entity: default_layout_root.0,
                    need_respawn: false,
                });
            }
        }
    }
    */
}

// pub struct NewLayoutBuilder {
//     tile_sizes: HashMap<SequencePairTile, U64Vec2>,
// }
