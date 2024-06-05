use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use std::io::prelude::*;

pub const CONFIG_FILE_NAME: &'static str = "config.ron";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub alignment_line_width: f32,
    pub grid_line_width: f32,
}

impl std::default::Default for AppConfig {
    fn default() -> Self {
        Self {
            alignment_line_width: 8.0,
            grid_line_width: 1.0,
        }
    }
}

pub fn app_dir() -> Option<ProjectDirs> {
    ProjectDirs::from("com", "", "PafView")
}

pub fn load_app_config() -> anyhow::Result<AppConfig> {
    let app_dirs = app_dir().ok_or(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Could not find application config directory",
    ))?;

    let mut cfg_path = app_dirs.config_dir().to_path_buf();
    cfg_path.push(CONFIG_FILE_NAME);

    let mut file = std::fs::File::open(&cfg_path)?;
    let mut cfg_buf = String::new();
    let len = file.read_to_string(&mut cfg_buf)?;

    let cfg = ron::de::from_str(&cfg_buf[..len])?;

    Ok(cfg)
}

pub fn save_app_config(config: &AppConfig) -> anyhow::Result<()> {
    let app_dirs = app_dir().ok_or(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Could not find application config directory",
    ))?;

    let mut cfg_path = app_dirs.config_dir().to_path_buf();

    if !cfg_path.exists() {
        std::fs::create_dir(&cfg_path)?;
    }

    if !cfg_path.is_dir() {
        anyhow::bail!(
            "A file exists at the config directory path `{cfg_path:?}` but it is not a directory"
        );
    }

    cfg_path.push(CONFIG_FILE_NAME);

    let mut file = std::fs::File::create(&cfg_path)?;

    ron::ser::to_writer_pretty(&mut file, config, ron::ser::PrettyConfig::new())?;

    Ok(())
}
