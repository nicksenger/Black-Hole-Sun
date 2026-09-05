//! Pipeline-parallel ResNet-18 training for a binary corgi identifier.
#![allow(clippy::manual_async_fn)]

pub mod spec;
pub mod flow;
pub mod op;

pub use spec::*;
