use bevy::{asset::embedded_asset, prelude::*};

pub struct ViewerAssetsPlugin;

impl Plugin for ViewerAssetsPlugin {
    fn build(&self, app: &mut App) {
        // TODO: fix the paths
        embedded_asset!(app, "../../assets/icons/xmark-circle.png");
        embedded_asset!(app, "../../assets/icons/xmark-circle-solid.png");
        embedded_asset!(app, "../../assets/icons/paste-clipboard.png");
        embedded_asset!(app, "../../assets/icons/clipboard-check.png");

        app.add_systems(Startup, load_icons);
    }
}

#[derive(Resource)]
pub struct Icons {
    pub xmark: Handle<Image>,
    pub xmark_solid: Handle<Image>,

    pub paste_clipboard: Handle<Image>,
    pub clipboard_check: Handle<Image>,
}

fn load_icons(mut commands: Commands, asset_server: Res<AssetServer>) {
    let xmark = asset_server.load("embedded://pafview/app/../../assets/icons/xmark-circle.png");
    let xmark_solid =
        asset_server.load("embedded://pafview/app/../../assets/icons/xmark-circle-solid.png");
    let paste_clipboard =
        asset_server.load("embedded://pafview/app/../../assets/icons/paste-clipboard.png");
    let clipboard_check =
        asset_server.load("embedded://pafview/app/../../assets/icons/clipboard-check.png");

    commands.insert_resource(Icons {
        xmark,
        xmark_solid,
        paste_clipboard,
        clipboard_check,
    })
}
