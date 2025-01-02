use std::ops::DerefMut;

use bevy::{prelude::*, utils::HashMap};

use bevy_egui::EguiContexts;

use crate::{gui::AppWindowStates, sequences::SeqId, Sequences};

use super::{
    alignments::{AlignmentLayoutQuery, DefaultLayoutRoot},
    annotations::gui::AnnotationsWindow,
    view::{AlignmentViewport, ViewEvent},
};

pub(super) struct MenubarPlugin;

impl Plugin for MenubarPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenubarSize>()
            .init_resource::<AnnotationsWindow>()
            .add_systems(Startup, setup)
            .add_systems(
                PreUpdate,
                (
                    menubar_system,
                    annotations_window,
                    // regions_of_interest_system,
                    settings_window,
                    goto_region_window,
                )
                    .after(bevy_egui::EguiSet::BeginPass),
            );
        // .add_systems(
        //     Update,
        //     (
        //         menubar_system,
        //         regions_of_interest_system,
        //         settings_window,
        //         goto_region_window,
        //     ),
        // );
        // .add_systems(Update, (menubar_system, regions_of_interest_system).chain());
    }
}

#[derive(Default, Resource)]
pub(crate) struct WindowStates {
    pub(crate) window_states: AppWindowStates,
}

#[derive(Default, Resource)]
pub struct MenubarSize {
    pub height: f32,
}

fn setup(mut commands: Commands, annotations: Res<super::annotations::Annotations>) {
    commands.insert_resource(WindowStates {
        window_states: AppWindowStates::new(&annotations.0),
    });
}

pub(crate) fn menubar_system(
    mut contexts: EguiContexts,
    mut window_states: ResMut<WindowStates>,
    // mut figure_export_open: Option<ResMut<super::figure_export::FigureExportWindowOpen>>,
    mut layout_editor_open: ResMut<super::alignments::layout::editor::LayoutEditorOpen>,
    mut menubar_size: ResMut<MenubarSize>,
) {
    let window_states = &mut window_states.window_states;
    let ctx = contexts.ctx_mut();
    let menubar_resp = egui::TopBottomPanel::top("menu_panel").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Regions of Interest").clicked() {
                window_states.regions_of_interest_open = !window_states.regions_of_interest_open;
            }

            // show/hide goto range window
            if ui.button("Go to range").clicked() {
                window_states.goto_region_open = !window_states.goto_region_open;
            }

            if ui.button("Edit layout").clicked() {
                layout_editor_open.0 = !layout_editor_open.0;
            }

            // show/hide annotations list window
            if let Some(open) = window_states.annotation_list_open.as_mut() {
                if ui.button("Annotations").clicked() {
                    *open = !*open;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                let open = &mut window_states.config_open;
                if ui.button("Settings").clicked() {
                    *open = !*open;
                }

                #[cfg(debug_assertions)]
                {
                    let open = &mut window_states.label_physics_debug_open;
                    if ui.button("Debug label physics").clicked() {
                        *open = !*open;
                    }
                }

                // if let Some(fig_export) = figure_export_open.as_mut() {
                //     let open = &mut fig_export.is_open;
                //     if ui.button("Figure Export").clicked() {
                //         *open = !*open;
                //     }
                // }
            });
            // ui.spacing()
        })
    });

    menubar_size.height = menubar_resp.response.rect.height();
}

fn settings_window(
    mut contexts: EguiContexts,

    mut app_config: ResMut<crate::AppConfig>,

    mut window_states: ResMut<WindowStates>,
) {
    let ctx = contexts.ctx_mut();
    crate::gui::config::application_settings_window(
        ctx,
        &mut window_states.window_states.config_open,
        app_config.as_mut(),
    );
}

fn annotations_window(
    mut contexts: EguiContexts,

    annotations: Res<super::annotations::Annotations>,
    annot_entity_map: Res<super::annotations::AnnotationEntityMap>,
    mut window_states: ResMut<WindowStates>,

    mut annots_window: ResMut<AnnotationsWindow>,

    annotation_query: Query<(
        Entity,
        &super::annotations::Annotation,
        &super::annotations::DisplayEntities,
    )>,
    display_query: Query<&mut Visibility>,
    //
) {
    let ctx = contexts.ctx_mut();
    annots_window.show_window(
        &annotations.0,
        &mut window_states.window_states,
        annot_entity_map.as_ref(),
        annotation_query,
        display_query,
        ctx,
    );

    //
}

fn goto_region_window(
    mut contexts: EguiContexts,
    // alignment_grid: Res<crate::AlignmentGrid>,
    sequences: Res<Sequences>,
    mut window_states: ResMut<WindowStates>,

    viewport: Res<AlignmentViewport>,

    mut view_events: EventWriter<ViewEvent>,

    layouts: AlignmentLayoutQuery,
    default_layout_root: Res<DefaultLayoutRoot>,

    mut target_text: Local<String>,
    mut query_text: Local<String>,
) {
    let ctx = contexts.ctx_mut();

    let mut view = viewport.view;

    let parse_seq_range = |text: &str| -> Option<(SeqId, std::ops::Range<u64>)> {
        let mut split = text.split(':');

        let name = split.next()?;
        let id = sequences.get_id(name)?;
        dbg!();

        let parsed_range = split.next().and_then(|t| {
            let mut range = t.split('-').filter_map(|s| s.parse::<u64>().ok());
            range.next().zip(range.next())
        });

        let range = if let Some((start, end)) = parsed_range {
            start..end
        } else {
            0..sequences.get(id)?.len()
        };

        Some((id, range))
    };

    fn make_range_map(
        offsets: &HashMap<SeqId, f64>,
    ) -> impl Fn((SeqId, std::ops::Range<u64>)) -> Option<std::ops::RangeInclusive<f64>> + '_ {
        |(seq, range)| {
            dbg!(&seq, &range);
            // let offset = offsets.get(&seq)?;
            let offset = offsets.get(&seq);
            dbg!(&offset);
            let offset = offset?;
            let start = range.start as f64 + *offset;
            let end = range.end as f64 + *offset;
            Some(start..=end)
        }
    }

    let Ok((_, _transform, layout_handle, _)) = layouts.layout_roots.get(default_layout_root.0)
    else {
        return;
    };

    let Some(layout) = layouts.layout_assets.get(layout_handle) else {
        return;
    };

    egui::Window::new("Go to region")
        .open(&mut window_states.window_states.goto_region_open)
        .show(ctx, |ui| {
            let (pressed_enter, aspect_ratio) = ui.input(|i| {
                let pressed = i.key_pressed(egui::Key::Enter);
                let rect = i.screen_rect();
                let aspect = rect.width() as f64 / rect.height() as f64;
                (pressed, aspect)
            });

            // Target/X

            let goto_btn = ui.button("Go to range");

            let mut goto = goto_btn.clicked();

            ui.horizontal(|ui| {
                ui.label("Target");
                // let target_text = ui.text_edit_singleline(target_text.as_mut());
                let target_text = ui.text_edit_singleline(target_text.deref_mut());
                goto |= target_text.lost_focus() && pressed_enter;
            });

            // Query/Y

            ui.horizontal(|ui| {
                ui.label("Query");
                let query_text = ui.text_edit_singleline(query_text.deref_mut());
                goto |= query_text.lost_focus() && pressed_enter;
            });

            // sequence-local ranges

            if goto {
                dbg!("\nTarget");
                let target_range = parse_seq_range(target_text.as_str());
                let x_range = target_range.and_then(make_range_map(&layout.target_offsets));
                dbg!("\nQuery");
                let query_range = parse_seq_range(query_text.as_str());
                let y_range = query_range.and_then(make_range_map(&layout.query_offsets));

                let y_range = y_range.map(|y_range| {
                    let y0 = layout.maxs.y - *y_range.end();
                    let y1 = layout.maxs.y - *y_range.start();
                    y0..=y1
                });

                println!();
                println!("target offsets {:?}", layout.target_offsets);
                println!("query offsets {:?}", layout.query_offsets);

                let new_view =
                    view.fit_ranges_in_view_with_aspect_f64(aspect_ratio, x_range, y_range);
                view = new_view;
            }
        });

    if viewport.view != view {
        println!("sending view event {view:?}");
        view_events.send(ViewEvent { view });
    }
}

/*
#[allow(dead_code)]
#[derive(Default, Resource)]
struct RegionsOfInterest {
    gui: crate::gui::regions::RegionsOfInterestGui,
}

fn regions_of_interest_system(
    mut contexts: EguiContexts,
    // viewer: Res<super::PafViewer>,

    mut alignment_view: ResMut<AlignmentViewport>,
    mut annotation_painter: ResMut<AnnotationPainter>,
    mut window_states: ResMut<WindowStates>,
    mut roi_gui: ResMut<RegionsOfInterest>,
) {
    let ctx = contexts.ctx_mut();

    let roi_gui = &mut roi_gui.gui;

    let mut view = alignment_view.view;

    roi_gui.show_window(
        ctx,
        &viewer.app,
        &mut annotation_painter.0,
        &mut view,
        &mut window_states.window_states,
    );

    if view != alignment_view.view {
        alignment_view.view = view;
    }

    // roi_gui.show_window(
    //     ctx,
    //     &viewer.app,
}
*/
