//! Sun programs — strategy-level drivers selected by a compiled topology.
//!
//! Each module implements [`crate::compile::SunProgram`] for one strategy and
//! owns the state, actions, effects, and node-side loops that strategy needs.

pub mod checkpoint_evaluate;
pub mod forward_only;
pub mod pipeline_backward;
pub mod two_sided_zo;
