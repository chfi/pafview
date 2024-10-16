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
            .add_plugins(gui::AlignmentLayoutGuiPlugin);
    }
}

#[derive(Asset, Clone, TypePath)]
pub struct SeqPairLayout {
    pub aabbs: HashMap<SequencePairTile, Aabb>,

    pub layout_qbvh: LayoutQbvh,
}

#[derive(Resource, Clone)]
pub struct DefaultLayout {
    // pub layout: SeqPairLayout,
    pub layout: Handle<SeqPairLayout>,
    builder: LayoutBuilder,
}

#[derive(Component, Deref, DerefMut)]
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

    /// see vertical_offset
    pub horizontal_offset: Option<f64>,

    data: LayoutInput,
}

impl LayoutBuilder {
    pub fn build(self, sequences: &crate::Sequences) -> SeqPairLayout {
        let aabbs = match self.data {
            LayoutInput::Axes { targets, queries } => {
                let mut aabbs = HashMap::default();

                let mut x_offset = 0.0;

                for target in targets {
                    let Some(tgt_seq) = sequences.get(target) else {
                        continue;
                    };

                    let tgt_len = tgt_seq.len() as f64;

                    let x0 = x_offset;
                    x_offset += self.horizontal_offset.unwrap_or(tgt_len);

                    let mut y_offset = 0.0;

                    for &query in &queries {
                        let Some(qry_seq) = sequences.get(query) else {
                            continue;
                        };
                        let qry_len = qry_seq.len() as f64;

                        let y0 = y_offset;
                        y_offset += self.vertical_offset.unwrap_or(qry_len);

                        // let center = [x0, y0];
                        let center = [x0 + tgt_len * 0.5, y0 + qry_len * 0.5];

                        let half_extents = [tgt_len * 0.5, qry_len * 0.5];

                        let aabb = Aabb::from_half_extents(center.into(), half_extents.into());

                        aabbs.insert(SequencePairTile { target, query }, aabb);
                    }
                }

                aabbs
            }
            LayoutInput::TilePositions { offsets } => offsets
                .iter()
                .filter_map(|(&seq_pair, &offset)| {
                    let tgt_len = sequences.get(seq_pair.target)?.len() as f64;
                    let qry_len = sequences.get(seq_pair.query)?.len() as f64;
                    let half_extents = [tgt_len * 0.5, qry_len * 0.5];
                    let aabb =
                        Aabb::from_half_extents([offset.x, offset.y].into(), half_extents.into());
                    Some((seq_pair, aabb))
                })
                .collect(),
        };

        let layout_qbvh = LayoutQbvh::from_tiles(aabbs.iter().map(|(&sp, &aabb)| (sp, aabb)));

        SeqPairLayout { aabbs, layout_qbvh }
    }

    pub fn from_axes<T, Q>(targets: T, queries: Q) -> Self
    where
        T: IntoIterator<Item = SeqId>,
        Q: IntoIterator<Item = SeqId>,
    {
        let mut targets = targets.into_iter().collect::<Vec<_>>();
        let mut queries = queries.into_iter().collect::<Vec<_>>();

        let data = LayoutInput::Axes { targets, queries };
        Self {
            vertical_offset: None,
            horizontal_offset: None,
            data,
        }
    }

    pub fn from_positions<P: Into<DVec2>>(
        tile_positions: impl IntoIterator<Item = (SequencePairTile, P)>,
    ) -> Self {
        let offsets = tile_positions
            .into_iter()
            .map(|(t, p)| (t, p.into()))
            .collect();

        Self {
            vertical_offset: None,
            horizontal_offset: None,
            data: LayoutInput::TilePositions { offsets },
        }
    }

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
    TilePositions {
        offsets: HashMap<SequencePairTile, DVec2>,
    },
}

#[derive(Clone)]
pub struct LayoutQbvh {
    qbvh: parry::partitioning::Qbvh<usize>,
    tile_index_map: Vec<SequencePairTile>,
    aabbs: Vec<Aabb>,
    // tile_index_map: HashMap<usize,
}

impl LayoutQbvh {
    pub fn from_tiles<T>(tiles: T) -> Self
    where
        T: ExactSizeIterator<Item = (SequencePairTile, Aabb)>,
    {
        use parry::partitioning::Qbvh;

        let (tile_index_map, mut leaf_data): (
            Vec<SequencePairTile>,
            Vec<(usize, parry::bounding_volume::Aabb)>,
        ) = tiles
            .enumerate()
            .map(|(ix, (seq_pair, aabb))| (seq_pair, (ix, aabb)))
            .unzip();

        let aabbs = leaf_data.iter().map(|(_, aabb)| *aabb).collect::<Vec<_>>();
        let mut qbvh = Qbvh::new();
        qbvh.clear_and_rebuild(leaf_data.into_iter(), 1.0);

        Self {
            qbvh,
            tile_index_map,
            aabbs,
        }
    }

    pub fn tiles_in_rect_callback(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
        mut callback: impl FnMut(SequencePairTile) -> bool,
    ) {
        let center = center.into();
        let half_extents = half_extents.into();

        let query_aabb =
            parry::bounding_volume::Aabb::from_half_extents(center.into(), half_extents.into());

        let leaf_cb = &mut |index: &usize| {
            let aabb = &self.aabbs[*index];
            if query_aabb.intersects(aabb) {
                let seq_pair = self.tile_index_map[*index];
                callback(seq_pair)
            } else {
                true
            }
        };

        let mut visitor =
            parry::query::visitors::BoundingVolumeIntersectionsVisitor::new(&query_aabb, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn tiles_in_rect(
        &self,
        center: impl Into<[f64; 2]>,
        half_extents: impl Into<[f64; 2]>,
    ) -> Vec<SequencePairTile> {
        let mut results = Vec::new();

        self.tiles_in_rect_callback(center, half_extents, |tile| {
            results.push(tile);
            true
        });

        results
    }

    pub fn tiles_at_point_callback(
        &self,
        point: impl Into<[f64; 2]>,
        mut callback: impl FnMut(SequencePairTile) -> bool,
    ) {
        let query_pt = point.into();

        let leaf_cb = &mut |index: &usize| {
            let aabb = &self.aabbs[*index];
            if aabb.contains_local_point(&query_pt.into()) {
                let seq_pair = self.tile_index_map[*index];
                callback(seq_pair)
            } else {
                true
            }
        };

        let query_pt = query_pt.into();
        let mut visitor =
            parry::query::visitors::PointIntersectionsVisitor::new(&query_pt, leaf_cb);
        self.qbvh.traverse_depth_first(&mut visitor);
    }

    pub fn tiles_at_point(&self, point: impl Into<[f64; 2]>) -> Vec<SequencePairTile> {
        let mut results = Vec::new();

        self.tiles_at_point_callback(point, |tile| {
            results.push(tile);
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

pub mod gui {

    use super::*;
    use bevy_egui::EguiContexts;

    pub struct AlignmentLayoutGuiPlugin;

    impl Plugin for AlignmentLayoutGuiPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(LayoutEditorOpen(false))
                .init_resource::<LiveLayoutBuilder>()
                .add_systems(
                    PreUpdate,
                    show_live_layout_editor.after(bevy_egui::EguiSet::BeginFrame),
                );
        }
    }

    #[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Reflect)]
    pub struct LayoutEditorOpen(pub bool);

    #[derive(Default, PartialEq)]
    struct LayoutEditorState {
        vertical_offset_on: bool,
        vertical_offset: f64,

        horizontal_offset_on: bool,
        horizontal_offset: f64,
    }

    #[derive(Resource, Default, Clone, PartialEq)]
    pub struct LiveLayoutBuilder {
        builder: Option<LayoutBuilder>,
    }

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
                        ui.label("Vertical offset");
                        ui.checkbox(&mut editor_state.vertical_offset_on, "Enable");
                        ui.add_enabled(
                            editor_state.vertical_offset_on,
                            egui::DragValue::new(&mut editor_state.vertical_offset),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Horizontal offset");
                        ui.checkbox(&mut editor_state.horizontal_offset_on, "Enable");
                        ui.add_enabled(
                            editor_state.horizontal_offset_on,
                            egui::DragValue::new(&mut editor_state.horizontal_offset),
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
                vertical_offset_on: b.vertical_offset.is_some(),
                vertical_offset: b.vertical_offset.unwrap_or_default(),
                horizontal_offset_on: b.horizontal_offset.is_some(),
                horizontal_offset: b.horizontal_offset.unwrap_or_default(),
            });

        {
            let editor_state: &LayoutEditorState = &editor_state;
            if builder_state.map(|s| &s != editor_state).unwrap_or(false) {
                if let Some(builder) = builder.builder.as_mut() {
                    if editor_state.vertical_offset_on {
                        if Some(editor_state.vertical_offset) != builder.vertical_offset {
                            builder.vertical_offset = Some(editor_state.vertical_offset);
                            builder_changed = true;
                        }
                    }

                    if editor_state.horizontal_offset_on {
                        if Some(editor_state.horizontal_offset) != builder.horizontal_offset {
                            builder.horizontal_offset = Some(editor_state.horizontal_offset);
                            builder_changed = true;
                        }
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
}
