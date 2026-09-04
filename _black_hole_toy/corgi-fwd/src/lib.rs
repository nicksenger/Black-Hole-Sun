//! A forward-only ResNet-18 pipeline for classifying Stanford Dogs images.
//!
//! The model is split at the natural ResNet boundaries:
//!
//! ```text
//! dataset generator -> stem -> stage1 -> stage2 -> stage3 -> stage4 -> binary head -> policy
//! ```

pub mod contracts;
pub mod jungle;
pub mod model;
pub mod operations;

pub use contracts::*;
pub use model::*;

/// Re-exported so the other corgi examples can share one dataset definition.
pub use toy_common::dataset::{DATASET_SAMPLES, SampleMetadata};
