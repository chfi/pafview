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

#[derive(Actionlike, Clone, PartialEq, Eq, Hash, Reflect)]
pub enum SelectedToolActions {
    Primary,
    Secondary,
}
