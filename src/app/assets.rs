use bevy::{
    asset::{embedded_asset, embedded_path, load_internal_asset, load_internal_binary_asset},
    prelude::*,
};

// const XMARK_HANDLE: Handle<Image> = Handle::weak_from_u128(50522492056905164513056341922265697668);
// const XMARK_SOLID_HANDLE: Handle<Image> =
//     Handle::weak_from_u128(207337932068400138304756780918668463570);
// const PASTE_CLIPBOARD_HANDLE: Handle<Image> =
//     Handle::weak_from_u128(242143070246553915116460954439226237175);
// const CLIPBOARD_CHECK_HANDLE: Handle<Image> =
//     Handle::weak_from_u128(239571613109265572516485758445731624569);

pub struct ViewerAssetsPlugin;

impl Plugin for ViewerAssetsPlugin {
    fn build(&self, app: &mut App) {
        // TODO: fix the dots
        embedded_asset!(app, "../../assets/icons/xmark-circle.png");
        embedded_asset!(app, "../../assets/icons/xmark-circle-solid.png");
        embedded_asset!(app, "../../assets/icons/paste-clipboard.png");
        embedded_asset!(app, "../../assets/icons/clipboard-check.png");
        // embedded_asset!(app, "/app/../../", "../../assets/icons/xmark-circle.png");

        // let path = embedded_path!("../../assets/icons/xmark-circle.png");
        // println!("{path:?}");

        // let xmark = embedded_asset!(omit_prefix, "icons/xmark-circle.png");
        // let xmark_solid = embedded_asset!(omit_prefix, "icons/xmark-circle-solid.png");
        // let paste_clipboard = embedded_asset!(omit_prefix, "icons/paste-clipboard.png");
        // let clipboard_check = embedded_asset!(omit_prefix, "icons/clipboard-check.png");

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
