use bevy::prelude::*;
use leafwing_input_manager::prelude::*;

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        todo!()
    }
}

/*


*/

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum SelectedToolAction {
    Primary,
    Secondary,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum RectangleSelectAction {
    StartSelect,
    CancelSelect,
    CompleteSelect,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct RectangleZoomAction(RectangleSelectAction);

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct RulerAction(RectangleSelectAction);

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub enum Tools {
    #[default]
    Pan,
    Select,
    Ruler,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum ViewAction {
    #[actionlike(DualAxis)]
    Pan,
    #[actionlike(Axis)]
    Zoom,
    PanAroundPoint, // for click & drag, exact
    Reset,
}

#[derive(Default, Resource)]
pub struct ActiveTool {
    tool: Tools,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum UserAction {
    SelectedTool(SelectedToolAction),
    View(ViewAction),
}

fn default_input_map() -> InputMap<UserAction> {
    let mut input_map = InputMap::default();

    input_map.insert(
        UserAction::SelectedTool(SelectedToolAction::Primary),
        MouseButton::Left,
    );
    input_map.insert(
        UserAction::SelectedTool(SelectedToolAction::Secondary),
        MouseButton::Right,
    );

    // input_map.insert

    input_map
}

/*
fn initialize_actions(
    mut commands: Commands,
) {

}
*/
