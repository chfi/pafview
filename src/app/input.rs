use bevy::prelude::*;
use leafwing_input_manager::prelude::*;

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            InputManagerPlugin::<UserAction>::default(),
            InputManagerPlugin::<ViewAction>::default(),
            InputManagerPlugin::<RulerAction>::default(),
        ))
        .init_resource::<ActiveTool>()
        .configure_sets(
            PreUpdate,
            (InputSet::ForwardUserActions, InputSet::HandleActions)
                .chain()
                .in_set(leafwing_input_manager::plugin::InputManagerSystem::ManualControl), // .after(leafwing_input_manager::plugin::InputManagerSystem::ManualControl),
        )
        .add_systems(Startup, setup_input)
        .add_systems(
            PreUpdate,
            (forward_tool_actions, forward_view_actions).in_set(InputSet::ForwardUserActions),
        );
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum InputSet {
    ForwardUserActions,
    HandleActions,
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
    StartOrEndSelect,
    CancelSelect,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct RectangleZoomAction(RectangleSelectAction);

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct RulerAction(RectangleSelectAction);

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub enum Tools {
    #[default]
    Pan,
    Ruler,
    // Select,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum ViewAction {
    #[actionlike(DualAxis)]
    Pan,
    #[actionlike(Axis)]
    Zoom,
    AnchoredPan, // for click & drag, exact
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

    Cancel,
    // Undo,
}

fn setup_input(mut commands: Commands) {
    let user_action_map = default_input_map();

    commands.init_resource::<ActionState<UserAction>>();
    commands.insert_resource(user_action_map);

    commands.init_resource::<ActionState<ViewAction>>();
}

fn forward_tool_actions(
    active_tool: Res<ActiveTool>,
    user_actions: Res<ActionState<UserAction>>,

    mut view_actions: ResMut<ActionState<ViewAction>>,
    mut ruler_actions: ResMut<ActionState<RulerAction>>,
    // ruler_actions:

    // mut tool_actions
) {
    let primary_tool_data = user_actions
        .button_data(&UserAction::SelectedTool(SelectedToolAction::Primary))
        .cloned()
        .unwrap_or_default();
    let secondary_tool_data = user_actions
        .button_data(&UserAction::SelectedTool(SelectedToolAction::Secondary))
        .cloned()
        .unwrap_or_default();

    match active_tool.tool {
        Tools::Pan => {
            view_actions.set_button_data(ViewAction::AnchoredPan, primary_tool_data);
        }
        Tools::Ruler => {
            ruler_actions.set_button_data(
                RulerAction(RectangleSelectAction::StartOrEndSelect),
                primary_tool_data,
            );

            let cancel_data = user_actions
                .button_data(&UserAction::Cancel)
                .cloned()
                .unwrap_or_default();
            ruler_actions.set_button_data(
                RulerAction(RectangleSelectAction::CancelSelect),
                cancel_data,
            );
        }
    }
    //
}

fn forward_view_actions(
    user_actions: Res<ActionState<UserAction>>,
    mut view_actions: ResMut<ActionState<ViewAction>>,
    // user_actions: Events<
) {
    let action = ViewAction::Zoom;
    let view = view_actions.axis_data_mut_or_default(&action);
    *view = user_actions
        .axis_data(&UserAction::View(action))
        .cloned()
        .unwrap_or_default();

    let action = ViewAction::Pan;
    let view = view_actions.dual_axis_data_mut_or_default(&action);
    *view = user_actions
        .dual_axis_data(&UserAction::View(action))
        .cloned()
        .unwrap_or_default();

    for btnlike in [ViewAction::AnchoredPan, ViewAction::Reset] {
        view_actions.set_button_data(
            btnlike,
            user_actions
                .button_data(&UserAction::View(btnlike))
                .cloned()
                .unwrap_or_default(),
        );
    }

    // let view_action = user_actions.action_data(UserAction::V)
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

    input_map.insert_dual_axis(
        UserAction::View(ViewAction::Pan),
        KeyboardVirtualDPad::new(KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD),
    );
    input_map.insert_dual_axis(
        UserAction::View(ViewAction::Pan),
        KeyboardVirtualDPad::new(
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
        ),
    );
    input_map.insert_axis(
        UserAction::View(ViewAction::Zoom),
        KeyboardVirtualAxis::new(KeyCode::PageUp, KeyCode::PageDown),
    );

    input_map.insert_axis(UserAction::View(ViewAction::Zoom), MouseScrollAxis::Y);

    input_map
}

/*
fn initialize_actions(
    mut commands: Commands,
) {

}
*/
