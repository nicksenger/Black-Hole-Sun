//! Example: zeroth-order optimization of a ResNet-18 corgi classifier.
//!
//! The graph is intentionally the same six-operation decomposition as
//! `corgi-fwd`; only the cell primordium and top-level program change:
//!
//! ```text
//! dataset generator -> stem -> stage1 -> stage2 -> stage3 -> stage4 -> head
//!       ^                                                               |
//!       +---------------- TwoSidedZo losses and updates ----------------+
//! ```

pub mod contracts;
pub mod jungle;
pub mod operations;

pub use contracts::*;
