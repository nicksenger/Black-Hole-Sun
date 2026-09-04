//! Shared plumbing for the black-hole toy examples: local server/runtime
//! helpers and (with the `dataset` feature) Stanford Dogs dataset access.

pub mod runtime;

#[cfg(feature = "dataset")]
pub mod dataset;
