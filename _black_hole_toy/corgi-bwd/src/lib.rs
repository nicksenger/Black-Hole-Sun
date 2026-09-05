//! Pipeline-parallel ResNet-18 training for a binary corgi identifier.
#![allow(clippy::manual_async_fn)]

pub mod flow;
pub mod op;
pub mod spec;

pub use spec::*;
