//! Pipeline-parallel ResNet-18 training for a binary corgi identifier.
#![allow(clippy::manual_async_fn)]

pub mod contracts;
pub mod jungle;
pub mod operations;

pub use contracts::*;
