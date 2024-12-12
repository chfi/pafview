use bevy::{input::touch::touch_screen_input_system, prelude::*};
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
        .add_plugins(cursor::CursorInputPlugin)
        .configure_sets(
            PreUpdate,
            (
                InputSet::BuildUserActions,
                InputSet::ForwardUserActions,
                InputSet::HandleActions,
            )
                .chain()
                .in_set(leafwing_input_manager::plugin::InputManagerSystem::ManualControl),
        )
        .configure_sets(
            PreUpdate,
            InputSet::BuildUserActions.after(bevy_mod_picking::picking_core::PickSet::PostFocus),
        )
        .add_systems(Startup, setup_input)
        .add_systems(
            PreUpdate,
            touch_view_actions
                .after(touch_screen_input_system)
                .in_set(InputSet::BuildUserActions),
        )
        .add_systems(
            PreUpdate,
            add_cursor_zoom_origin
                .in_set(InputSet::ForwardUserActions)
                .before(forward_view_actions),
        )
        .add_systems(
            PreUpdate,
            (forward_view_actions, forward_tool_actions)
                .chain()
                .in_set(InputSet::ForwardUserActions),
        );
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum InputSet {
    BuildUserActions,
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
pub struct RectangleZoomAction(pub RectangleSelectAction);

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub struct RulerAction(pub RectangleSelectAction);

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub enum Tools {
    #[default]
    Pan,
    Ruler,
    // Select,
}

#[derive(Actionlike, Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum ViewAction {
    /// Pan the view by the axes' value times the view size
    #[actionlike(DualAxis)]
    Pan,
    /// Scale the view
    #[actionlike(Axis)]
    Zoom,
    /// Provides the origin/center for a `Zoom` action
    #[actionlike(DualAxis)]
    ZoomOrigin,
    AnchoredPan, // for click & drag, exact
    Reset,
    RectangleZoom(RectangleSelectAction),
}

#[derive(Default, Resource)]
pub struct ActiveTool {
    tool: Tools,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
pub enum UserAction {
    SelectedTool(SelectedToolAction),
    View(ViewAction),

    Paste,

    Cancel,

    ModifierPlus,
    ModifierMinus,
}

impl Actionlike for UserAction {
    fn input_control_kind(&self) -> InputControlKind {
        use UserAction::*;
        match self {
            SelectedTool(tool) => tool.input_control_kind(),
            View(view) => view.input_control_kind(),
            Paste | Cancel | ModifierPlus | ModifierMinus => InputControlKind::Button,
        }
    }
}

fn setup_input(mut commands: Commands) {
    let user_action_map = default_input_map();
    commands.insert_resource(user_action_map);

    // commands.init_resource::<ActionState<UserAction>>();
    let mut initial_actions = ActionState::<UserAction>::default();
    initial_actions.set_axis_pair(&UserAction::View(ViewAction::ZoomOrigin), [0.5, 0.5].into());
    initial_actions.set_value(&UserAction::View(ViewAction::Zoom), 1.0);
    commands.insert_resource(initial_actions);

    let mut view_actions = ActionState::<ViewAction>::default();
    view_actions.set_axis_pair(&ViewAction::ZoomOrigin, [0.5, 0.5].into());
    view_actions.set_value(&ViewAction::Zoom, 1.0);
    commands.insert_resource(view_actions);
    // commands.init_resource::<ActionState<ViewAction>>();

    commands.init_resource::<ActionState<RulerAction>>();
}

fn forward_tool_actions(
    active_tool: Res<ActiveTool>,
    user_actions: Res<ActionState<UserAction>>,

    mut view_actions: ResMut<ActionState<ViewAction>>,
    mut ruler_actions: ResMut<ActionState<RulerAction>>,
    // ruler_actions:

    // mut tool_actions
    hover_map: Res<bevy_mod_picking::focus::HoverMap>,
) {
    let primary_tool_data = user_actions
        .button_data(&UserAction::SelectedTool(SelectedToolAction::Primary))
        .cloned()
        .unwrap_or_default();
    let secondary_tool_data = user_actions
        .button_data(&UserAction::SelectedTool(SelectedToolAction::Secondary))
        .cloned()
        .unwrap_or_default();

    let tool = if user_actions.pressed(&UserAction::ModifierMinus) {
        Tools::Ruler
    } else {
        Tools::Pan
    };

    // match active_tool.tool {
    match tool {
        Tools::Pan => {
            view_actions.set_button_data(ViewAction::AnchoredPan, primary_tool_data);
        }
        Tools::Ruler => {
            ruler_actions.set_button_data(
                RulerAction(RectangleSelectAction::StartOrEndSelect),
                primary_tool_data,
            );
        }
    }
}

// run before `forward_view_actions`, after `update_cursor_input`
fn add_cursor_zoom_origin(
    cursor: Res<cursor::CursorPosition>,
    mut user_actions: ResMut<ActionState<UserAction>>,
    touches: Res<Touches>,
) {
    let Some(cursor) = cursor.unit else {
        return;
    };

    if let Some(zoom_data) = user_actions
        .axis_data(&UserAction::View(ViewAction::Zoom))
        .cloned()
    {
        if zoom_data.value != 0.0 {
            let action = UserAction::View(ViewAction::ZoomOrigin);
            // NB: this is the easiest way of seeing if there's no touch at all
            if touches.first_pressed_position().is_none() {
                let mut pos = cursor;
                pos.y *= -1.0;
                pos += Vec2::splat(0.5);
                // println!("setting zoom center to {pos:?}");
                user_actions.set_axis_pair(&action, pos);
            }
        }
    }
}

/*
fn egui_block_mouse_inputs(mut inputs: ResMut<CentralInputStore>, egui_focus: Res<EguiFocus>) {
    // if egui_focus.cursor_over_egui {
    //     println!("blocking...");
    //     for button in [MouseButton::Left, MouseButton::Right, MouseButton::Middle] {
    //         if inputs.pressed(&button) {
    //             inputs.update_buttonlike(button, false);
    //         }
    //     }

    //     // if inputs.pressed(&MouseButton::Left) {
    //     //     inputs.update_buttonlike(MouseButton::Left, false)
    //     // }
    // }
}
*/

fn forward_view_actions(
    user_actions: Res<ActionState<UserAction>>,
    mut view_actions: ResMut<ActionState<ViewAction>>,
) {
    let action = ViewAction::Zoom;
    let view_zoom = view_actions.axis_data_mut_or_default(&action);
    let zoom_data = user_actions
        .axis_data(&UserAction::View(action))
        .cloned()
        .unwrap_or_default();

    // println!("zoom_data value: {}", zoom_data.value);
    *view_zoom = zoom_data;

    let zoom_center = view_actions.dual_axis_data_mut_or_default(&ViewAction::ZoomOrigin);
    *zoom_center = user_actions
        .dual_axis_data(&UserAction::View(ViewAction::ZoomOrigin))
        .cloned()
        .unwrap_or_default();

    let action = ViewAction::Pan;
    let view = view_actions.dual_axis_data_mut_or_default(&action);
    *view = user_actions
        .dual_axis_data(&UserAction::View(action))
        .cloned()
        .unwrap_or_default();

    let cancel = user_actions
        .button_data(&UserAction::Cancel)
        .cloned()
        .unwrap_or_default();
    *view_actions.button_data_mut_or_default(&ViewAction::Reset) = cancel;

    for btnlike in [
        ViewAction::AnchoredPan,
        ViewAction::RectangleZoom(RectangleSelectAction::StartOrEndSelect),
    ] {
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

// run after (bevy_input's) `touch_screen_input_system`, before `forward_view_actions`
fn touch_view_actions(
    // viewport: Res<AlignmentViewport>,
    touches: Res<Touches>,
    mut user_actions: ResMut<ActionState<UserAction>>,

    mut frame_touches: Local<Vec<bevy::input::touch::Touch>>,

    windows: Query<&Window>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };

    let win_size = window.size();
    // let &[vw, vh] = viewport.view.size().as_array();
    // let view_size = bevy::math::DVec2::new(vw, vh);

    frame_touches.clear();
    frame_touches.extend(touches.iter().copied());

    match frame_touches.as_slice() {
        [] => {}
        [touch] => {
            // pan
            let delta = touch.delta();
            let scaled = delta / win_size;
            user_actions.set_axis_pair(&UserAction::View(ViewAction::Pan), -scaled);
        }
        [ta, tb, ..] => {
            // zoom
            let pa_0 = ta.previous_position();
            let pa_1 = ta.position();
            let pb_0 = tb.previous_position();
            let pb_1 = tb.position();

            let d_0 = pa_0 - pb_0;
            let d_1 = pa_1 - pb_1;

            let len_scale = d_1.length() / d_0.length();

            let origin = ((pa_1 + pb_1) / win_size) * 0.5;
            user_actions.set_axis_pair(&UserAction::View(ViewAction::ZoomOrigin), origin);
            user_actions.set_value(&UserAction::View(ViewAction::Zoom), 1.0 - len_scale);
        }
    }
}

fn default_input_map() -> InputMap<UserAction> {
    let mut input_map = InputMap::default();

    input_map.insert(
        UserAction::SelectedTool(SelectedToolAction::Primary),
        MouseButton::Left,
    );

    input_map.insert(
        UserAction::View(ViewAction::RectangleZoom(
            RectangleSelectAction::StartOrEndSelect,
        )),
        MouseButton::Right,
    );
    // input_map.insert(
    //     UserAction::SelectedTool(SelectedToolAction::Secondary),
    //     MouseButton::Right,
    // );

    input_map.insert(
        UserAction::Paste,
        ButtonlikeChord::new([KeyCode::ControlLeft, KeyCode::KeyV]),
    );
    input_map.insert(
        UserAction::Paste,
        ButtonlikeChord::new([KeyCode::ControlRight, KeyCode::KeyV]),
    );

    input_map.insert(UserAction::ModifierPlus, KeyCode::ShiftLeft);
    input_map.insert(UserAction::ModifierPlus, KeyCode::ShiftRight);
    input_map.insert(UserAction::ModifierMinus, KeyCode::ControlLeft);
    input_map.insert(UserAction::ModifierMinus, KeyCode::ControlRight);

    input_map.insert(UserAction::Cancel, KeyCode::Escape);
    // input_map.insert(UserAction::View(ViewAction::Reset), KeyCode::Escape);
    // input_map.insert

    input_map.insert_dual_axis(
        UserAction::View(ViewAction::Pan),
        VirtualDPad::new(KeyCode::KeyW, KeyCode::KeyS, KeyCode::KeyA, KeyCode::KeyD),
    );
    input_map.insert_dual_axis(
        UserAction::View(ViewAction::Pan),
        VirtualDPad::new(
            KeyCode::ArrowUp,
            KeyCode::ArrowDown,
            KeyCode::ArrowLeft,
            KeyCode::ArrowRight,
        ),
    );
    input_map.insert_axis(
        UserAction::View(ViewAction::Zoom),
        VirtualAxis::new(KeyCode::PageDown, KeyCode::PageUp), // .with_processor(input_processors::ScalingAxisProcessor),
    );

    input_map.insert_axis(
        UserAction::View(ViewAction::Zoom),
        MouseScrollAxis::Y,
        // .with_processor(input_processors::ScalingAxisProcessor),
        // .replace_processing_pipeline([input_processors::ScalingAxisProcessor.into()]),
    );

    input_map
}

pub mod cursor {
    use bevy::math::DVec2;

    use crate::app::view::AlignmentViewport;

    use super::*;

    pub struct CursorInputPlugin;

    impl Plugin for CursorInputPlugin {
        fn build(&self, app: &mut App) {
            app
                // .add_plugins(InputManagerPlugin::<CursorInput>::default())
                .init_resource::<CursorPosition>()
                .add_systems(
                    PreUpdate,
                    update_cursor_input.in_set(InputSet::BuildUserActions),
                );
            // .insert_resource(ActionState::<CursorInput>::default());

            //
        }
    }

    #[derive(Resource, Default, Debug, Reflect)]
    pub struct CursorPosition {
        pub world: Option<DVec2>,
        pub screen: Option<Vec2>,
        pub unit: Option<Vec2>,
    }

    // NB: this should probably not update while the cursor is over UI/egui
    pub fn update_cursor_input(
        viewport: Res<AlignmentViewport>,

        mut cursor: ResMut<CursorPosition>,

        windows: Query<&Window>,
    ) {
        let Ok(window) = windows.get_single() else {
            return;
        };

        let win_dims = window.size();

        let view = &viewport.view;

        if let Some(cursor_pos) = window.cursor_position() {
            let world_pos = {
                let p: [f32; 2] = cursor_pos.into();
                let wp: [f64; 2] = view.map_screen_to_world(win_dims, p).into();
                bevy::math::DVec2::from(wp)
            };
            let screen_pos = Vec2::new(
                cursor_pos.x - win_dims.x * 0.5,
                win_dims.y - cursor_pos.y - win_dims.y * 0.5,
            );

            cursor.world = Some(world_pos);
            cursor.screen = Some(screen_pos);
            cursor.unit = Some(screen_pos / win_dims);
        } else {
            cursor.world = None;
            cursor.screen = None;
            cursor.unit = None;
        }
    }
}

/*
pub mod input_processors {
    use bevy::math::FloatOrd;
    use bevy::prelude::*;
    use leafwing_input_manager::prelude::*;
    use serde::{Deserialize, Serialize};
    use std::hash::{Hash, Hasher};

    pub struct InputProcessorsPlugin;

    impl Plugin for InputProcessorsPlugin {
        fn build(&self, app: &mut App) {
            app.register_axis_processor::<ScalingAxisProcessor>();
        }
    }

    /// Axis input processor that maps 0 to 1, positive values above 1, negative
    /// below 1 but above 0 (exact scaling to be decided)
    #[derive(Debug, Clone, Copy, PartialEq, Reflect, Serialize, Deserialize, Eq, Hash)]
    pub struct ScalingAxisProcessor;

    #[serde_typetag]
    impl CustomAxisProcessor for ScalingAxisProcessor {
        fn process(&self, input_value: f32) -> f32 {
            let zoom_rate = 0.05;
            let val = (1.0 - input_value * zoom_rate).clamp(0.1, 10.0);
            println!("scaling {input_value} -> {val}");
            val
        }
    }
}
*/
