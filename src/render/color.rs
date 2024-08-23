use rustc_hash::FxHashMap;

use crate::app::alignments::AlignmentIndex;
// use crate::paf::AlignmentIndex;

#[derive(Debug)]
pub struct PafColorSchemes {
    pub overrides: FxHashMap<AlignmentIndex, AlignmentColorScheme>,
    pub default: AlignmentColorScheme,
}

impl std::default::Default for PafColorSchemes {
    fn default() -> Self {
        Self {
            overrides: Default::default(),
            default: AlignmentColorScheme::light_mode(),
        }
    }
}

impl PafColorSchemes {
    pub fn dark_mode() -> Self {
        Self {
            overrides: Default::default(),
            default: AlignmentColorScheme::dark_mode(),
        }
    }

    pub fn get(&self, alignment: &AlignmentIndex) -> &AlignmentColorScheme {
        if let Some(colors) = self.overrides.get(alignment) {
            colors
        } else {
            &self.default
        }
    }

    pub fn fill_from_paf_like(
        &mut self,
        sequences: &crate::sequences::Sequences,
        alignments: &crate::paf::Alignments,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        use crate::Strand;
        use std::io::prelude::*;
        use std::io::BufReader;

        use crate::paf::parse_paf_name_range;

        let reader = std::fs::File::open(path).map(BufReader::new)?;

        // let mut result = Self::default();

        for line in reader.lines() {
            let line = line?;

            let mut fields = line.split('\t');

            let query = parse_paf_name_range(&mut fields);
            let strand = fields.next();
            let target = parse_paf_name_range(&mut fields);

            let Some((query_name, query_seq_len, query_seq_start, query_seq_end)) = query else {
                continue;
            };

            let Some(strand) = strand else {
                continue;
            };

            let Some((tgt_name, tgt_seq_len, tgt_seq_start, tgt_seq_end)) = target else {
                continue;
            };

            let Some(color_str) = fields.next() else {
                continue;
            };

            let query_id = *sequences.sequence_names.get_by_left(query_name).unwrap();
            let target_id = *sequences.sequence_names.get_by_left(tgt_name).unwrap();

            let target_range = tgt_seq_start..tgt_seq_end;
            let query_range = query_seq_start..query_seq_end;

            let query_strand = if strand == "+" {
                Strand::Forward
            } else {
                Strand::Reverse
            };

            let location = crate::paf::AlignmentLocation {
                target_range,
                query_range,
                query_strand,
                target_total_len: tgt_seq_len,
                query_total_len: query_seq_len,
            };

            let pair_id = (target_id, query_id);

            let Some(pair) = alignments.pairs.get(&pair_id) else {
                continue;
            };

            let Some(index) = pair.iter().position(|al| al.location == location) else {
                continue;
            };

            let al_index = AlignmentIndex {
                target: target_id,
                query: query_id,
                pair_index: index,
            };

            if let Some(color_scheme) = AlignmentColorScheme::from_def_str(color_str) {
                self.overrides.insert(al_index, color_scheme);
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlignmentColorScheme {
    pub m_fg: egui::Color32,
    pub m_bg: egui::Color32,

    pub eq_fg: egui::Color32,
    pub eq_bg: egui::Color32,

    pub x_fg: egui::Color32,
    pub x_bg: egui::Color32,

    pub i_fg: egui::Color32,
    pub i_bg: egui::Color32,

    pub d_fg: egui::Color32,
    pub d_bg: egui::Color32,
}

impl AlignmentColorScheme {
    pub fn get(&self, op: crate::CigarOp) -> (egui::Color32, egui::Color32) {
        let fg = self.get_fg(op);
        let bg = self.get_bg(op);
        (fg, bg)
    }

    pub fn set_fg(&mut self, op: crate::CigarOp, color: impl Into<egui::Color32>) {
        let field = match op {
            crate::CigarOp::M => &mut self.m_fg,
            crate::CigarOp::Eq => &mut self.eq_fg,
            crate::CigarOp::X => &mut self.x_fg,
            crate::CigarOp::I => &mut self.i_fg,
            crate::CigarOp::D => &mut self.d_fg,
        };

        *field = color.into();
    }
    pub fn set_bg(&mut self, op: crate::CigarOp, color: impl Into<egui::Color32>) {
        let field = match op {
            crate::CigarOp::M => &mut self.m_bg,
            crate::CigarOp::Eq => &mut self.eq_bg,
            crate::CigarOp::X => &mut self.x_bg,
            crate::CigarOp::I => &mut self.i_bg,
            crate::CigarOp::D => &mut self.d_bg,
        };

        *field = color.into();
    }

    pub fn get_fg(&self, op: crate::CigarOp) -> egui::Color32 {
        match op {
            crate::CigarOp::M => self.m_fg,
            crate::CigarOp::Eq => self.eq_fg,
            crate::CigarOp::X => self.x_fg,
            crate::CigarOp::I => self.i_fg,
            crate::CigarOp::D => self.d_fg,
        }
    }

    pub fn get_bg(&self, op: crate::CigarOp) -> egui::Color32 {
        match op {
            crate::CigarOp::M => self.m_bg,
            crate::CigarOp::Eq => self.eq_bg,
            crate::CigarOp::X => self.x_bg,
            crate::CigarOp::I => self.i_bg,
            crate::CigarOp::D => self.d_bg,
        }
    }
}

impl AlignmentColorScheme {
    pub const fn light_mode() -> Self {
        use egui::Color32 as C;

        Self {
            m_fg: C::WHITE,
            m_bg: C::BLACK,

            eq_fg: C::WHITE,
            eq_bg: C::BLACK,

            x_fg: C::WHITE,
            x_bg: C::RED,

            i_fg: C::BLACK,
            i_bg: C::WHITE,

            d_fg: C::BLACK,
            d_bg: C::WHITE,
        }
    }

    pub const fn dark_mode() -> Self {
        use egui::Color32 as C;

        Self {
            m_fg: C::BLACK,
            m_bg: C::WHITE,

            eq_fg: C::BLACK,
            eq_bg: C::WHITE,

            x_fg: C::WHITE,
            x_bg: C::RED,

            i_fg: C::WHITE,
            i_bg: C::BLACK,

            d_fg: C::WHITE,
            d_bg: C::BLACK,
        }
    }

    /// parse a color scheme definition from string consisting of a comma-separated
    /// list of `<op>:<color>:<color>` pairs, where `<op>` is one of `{M, =, X, I, D}` and
    /// `<color>` is a hex-coded RGB color formatted as `#RRGGBB`.
    /// the first `<color>` is the background color, the second the foreground color.
    pub fn from_def_str(color_def: &str) -> Option<Self> {
        let mut result = Self::light_mode();

        let pairs = color_def.split(',');

        fn parse_color(color_str: &str) -> Option<egui::Color32> {
            let get_chan = |range| {
                let v = &color_str[range];
                u32::from_str_radix(v, 16)
            };

            let r = get_chan(0..2).unwrap();
            let g = get_chan(2..4).unwrap();
            let b = get_chan(4..6).unwrap();

            let rgb = egui::Rgba::from_rgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
            Some(rgb.into())
        }

        for pair in pairs {
            let Some((op_str, col_str)) = pair.split_once(':') else {
                continue;
            };

            let Some(op) = op_str
                .chars()
                .next()
                .and_then(|op| crate::CigarOp::try_from(op).ok())
            else {
                continue;
            };

            let Some((bg_str, fg_str)) = col_str.split_once(':') else {
                continue;
            };

            if let Some(color_str) = bg_str.strip_prefix('#').filter(|s| s.len() == 6) {
                if let Some(color) = parse_color(color_str) {
                    result.set_bg(op, color);
                }
            }

            if let Some(color_str) = fg_str.strip_prefix('#').filter(|s| s.len() == 6) {
                if let Some(color) = parse_color(color_str) {
                    result.set_fg(op, color);
                }
            }
        }

        Some(result)
    }
}

#[derive(Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GPUColorScheme {
    pub m_bg: [f32; 4],
    pub eq_bg: [f32; 4],
    pub x_bg: [f32; 4],
    pub i_bg: [f32; 4],
    pub d_bg: [f32; 4],
}

impl GPUColorScheme {
    pub fn from_color_scheme(color: &AlignmentColorScheme) -> Self {
        let map_color = |c: egui::Color32| egui::Rgba::from(c).to_array();
        Self {
            m_bg: map_color(color.m_bg),
            eq_bg: map_color(color.eq_bg),
            x_bg: map_color(color.x_bg),
            i_bg: map_color(color.i_bg),
            d_bg: map_color(color.d_bg),
        }
    }
}

/*
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_color_def_parse() {
        let test_str = "M:#112233:#445566,=:#FFFFFF:#FF0000";

        let parsed = AlignmentColorScheme::from_def_str(test_str);

        println!("{parsed:?}");
    }
}
*/
