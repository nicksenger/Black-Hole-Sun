//! A small forward-only tensor pipeline.
//!
//! The example deliberately keeps the tensors and operations simple so the
//! topology and shape transition are easy to inspect:
//!
//! ```text
//! Generator (2x3) -> Matmul (2x4) -> Scale -> ReLU -> LogPolicy
//! ```
//!
//! The three operation cells form a statically typed DAG. In particular, the
//! `TypedEdges` declarations make each downstream input contract part of the
//! graph definition, while the enclosing flow is assembled through the
//! canonical `<Topology as BlackHole>::Sun<Program>` entrypoint.

pub mod contracts;
pub mod jungle;
pub mod operations;
