//! Hybrid pipeline- and data-parallel ResNet-18 corgi training.
#![allow(clippy::manual_async_fn)]

pub mod flow;
pub mod spec;

pub use spec::*;
