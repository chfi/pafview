use std::{
    collections::{BTreeMap, VecDeque},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{anyhow, Result};
use bimap::BiMap;
use egui::Galley;
use rustc_hash::FxHashMap;
use ultraviolet::{DVec2, Vec2};

use crate::{gui::AppWindowStates, PafViewerApp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnnotShapeId(usize);

#[derive(Default)]
pub struct AnnotationPainter {
    // galley_cache: FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
    galley_cache: FxHashMap<String, Arc<Galley>>,
    annotations: Vec<Box<dyn DrawAnnotation>>,

    enabled: Vec<bool>,
    // annotations: FxHashMap<usize, Box<dyn DrawAnnotation>>,
}

const LABEL_TEXT_SIZE: f32 = 12.0;

impl AnnotationPainter {
    pub fn add_shape(&mut self, draw: Box<dyn DrawAnnotation>) -> AnnotShapeId {
        let id = AnnotShapeId(self.annotations.len());
        self.annotations.push(draw);
        self.enabled.push(false);
        id
    }

    pub fn add_collection(
        &mut self,
        draw: impl IntoIterator<Item = Box<dyn DrawAnnotation>>,
    ) -> AnnotShapeId {
        let draw_collection = AnnotationDrawCollection {
            draw: draw.into_iter().collect(),
        };
        self.add_shape(Box::new(draw_collection))
    }

    pub fn cache_label_fonts(&mut self, fonts: &egui::text::Fonts, text: &str) -> Arc<Galley> {
        if let Some(galley) = self.galley_cache.get(text) {
            galley.clone()
        } else {
            let galley = fonts.layout_no_wrap(
                text.to_string(),
                egui::FontId::monospace(LABEL_TEXT_SIZE),
                egui::Color32::BLACK,
            );

            self.galley_cache.insert(text.to_string(), galley.clone());
            galley
        }
    }

    pub fn cache_label(&mut self, ctx: &egui::Context, text: &str) -> Arc<Galley> {
        if let Some(galley) = self.galley_cache.get(text) {
            galley.clone()
        } else {
            let galley = ctx.fonts(|fonts| {
                fonts.layout_no_wrap(
                    text.to_string(),
                    egui::FontId::monospace(LABEL_TEXT_SIZE),
                    egui::Color32::BLACK,
                )
            });

            self.galley_cache.insert(text.to_string(), galley.clone());
            galley
        }
    }

    pub fn set_shape_color(&mut self, shape_id: AnnotShapeId, color: egui::Color32) {
        self.annotations[shape_id.0].set_color(color);
    }

    // pub fn with_enable_shape_mut(&mut self, shape_id: AnnotShapeId, f: impl FnOnce(&mut bool)) {
    //     f(&mut self.enabled[shape_id.0])
    // }

    pub fn is_shape_enabled(&self, shape_id: AnnotShapeId) -> bool {
        self.enabled[shape_id.0]
    }

    pub fn enable_shape_mut(&mut self, shape_id: AnnotShapeId) -> &mut bool {
        &mut self.enabled[shape_id.0]
    }

    pub fn set_enable_shape(&mut self, shape_id: AnnotShapeId, enabled: bool) {
        self.enabled[shape_id.0] = enabled;
    }

    pub fn draw(&mut self, ctx: &egui::Context, view: &crate::view::View) {
        //
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Background,
            "annotation_painter_painter".into(),
        ));
        let screen_size = ctx.screen_rect().size();

        for (_id, (annot, enabled)) in std::iter::zip(&self.annotations, &self.enabled).enumerate()
        {
            if *enabled {
                annot.draw(&mut self.galley_cache, &painter, view, screen_size);
            }

            // todo draw labels separately; handle collision/avoid overlap (also handle tooltips, eventually?)
        }
    }
}

pub trait DrawAnnotation {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    );

    fn set_color(&mut self, _color: egui::Color32);

    // fn text(&self) -> Option<(&str, egui::Align2)> {
    //     None
    // }
}

pub struct AnnotationDrawCollection {
    draw: Vec<Box<dyn DrawAnnotation>>,
}

impl DrawAnnotation for AnnotationDrawCollection {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        for item in self.draw.iter() {
            item.draw(galley_cache, painter, view, screen_size);
        }
    }

    fn set_color(&mut self, color: egui::Color32) {
        for item in self.draw.iter_mut() {
            item.set_color(color);
        }
    }
}

pub struct AnnotationLabel {
    pub world_x_range: Option<std::ops::RangeInclusive<f64>>,
    pub world_y_range: Option<std::ops::RangeInclusive<f64>>,
    pub align: egui::Align2,

    pub text: String,
    // can't use TextFormat as key bc not Eq; hash manually and key w/ u64, later
    // pub format: egui::TextFormat,
}

impl DrawAnnotation for AnnotationLabel {
    fn draw(
        &self,
        galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        if !galley_cache.contains_key(&self.text) {
            let mut job = egui::text::LayoutJob::default();
            job.append(
                &self.text,
                0.0,
                egui::text::TextFormat {
                    font_id: egui::FontId::monospace(12.0),
                    color: egui::Color32::PLACEHOLDER,
                    ..Default::default()
                },
            );
            let galley = painter.layout_job(job);
            galley_cache.insert(self.text.clone(), galley);
        }

        let Some(galley) = galley_cache.get(&self.text).cloned() else {
            unreachable!();
        };

        let view_min = DVec2::new(view.x_min, view.y_min);
        let view_max = DVec2::new(view.x_max, view.y_max);

        // TODO take `align` into account
        let [p0, p1] = match (&self.world_x_range, &self.world_y_range) {
            (Some(xs), Some(ys)) => {
                // draw in top left of screen rect, for now
                let p0 = DVec2::new(*xs.start(), *ys.start());
                let p1 = DVec2::new(*xs.end(), *ys.end());
                [p0, p1]
            }
            (Some(xs), None) => {
                // draw at top of screen of vertical region
                let p0 = DVec2::new(*xs.start(), view_min.y);
                let p1 = DVec2::new(*xs.end(), view_max.y);
                [p0, p1]
            }
            (None, Some(ys)) => {
                // draw at left of screen of horizontal region
                let p0 = DVec2::new(view_min.x, *ys.start());
                let p1 = DVec2::new(view_min.x, *ys.end());
                [p0, p1]
            }
            _ => {
                return;
            }
        };

        let q0: [f32; 2] = view.map_world_to_screen(screen_size, p0).into();
        let q1: [f32; 2] = view.map_world_to_screen(screen_size, p1).into();

        // let q0: [f32; 2] = view.map_world_to_screen(screen_size, min).into();
        // let q1: [f32; 2] = view.map_world_to_screen(screen_size, max).into();

        let rect = egui::Rect::from_two_pos(q0.into(), q1.into());
        painter.galley(rect.left_top(), galley, egui::Color32::BLACK);
    }

    fn set_color(&mut self, _color: egui::Color32) {}
}

pub struct AnnotationWorldRegion {
    pub world_x_range: Option<std::ops::RangeInclusive<f64>>,
    pub world_y_range: Option<std::ops::RangeInclusive<f64>>,
    pub color: egui::Color32,
}

impl DrawAnnotation for AnnotationWorldRegion {
    fn draw(
        &self,
        // galley_cache: &mut FxHashMap<(String, egui::TextFormat), Arc<Galley>>,
        _galley_cache: &mut FxHashMap<String, Arc<Galley>>,
        painter: &egui::Painter,
        view: &crate::view::View,
        screen_size: egui::Vec2,
    ) {
        let view_min = DVec2::new(view.x_min, view.y_min);
        let view_max = DVec2::new(view.x_max, view.y_max);

        let [p0, p1] = match (&self.world_x_range, &self.world_y_range) {
            (Some(xs), Some(ys)) => {
                let p0 = DVec2::new(*xs.start(), *ys.start());
                let p1 = DVec2::new(*xs.end(), *ys.end());
                [p0, p1]
            }
            (Some(xs), None) => {
                let p0 = DVec2::new(*xs.start(), view_min.y);
                let p1 = DVec2::new(*xs.end(), view_max.y);
                [p0, p1]
            }
            (None, Some(ys)) => {
                let p0 = DVec2::new(view_min.x, *ys.start());
                let p1 = DVec2::new(view_max.x, *ys.end());
                [p0, p1]
            }
            _ => {
                return;
            }
        };

        let q0: [f32; 2] = view.map_world_to_screen(screen_size, p0).into();
        let q1: [f32; 2] = view.map_world_to_screen(screen_size, p1).into();

        let mut rect = egui::Rect::from_two_pos(q0.into(), q1.into());

        if rect.width() < 1.0 {
            rect.set_width(1.0);
        }
        if rect.height() < 1.0 {
            rect.set_height(1.0);
        }

        painter.rect_filled(rect, 0.0, self.color);
    }

    fn set_color(&mut self, color: egui::Color32) {
        self.color = color;
    }
}
