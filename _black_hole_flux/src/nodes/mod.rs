//! Spawned animal hosts — the node-side execution units of a Sun graph.
//!
//! These modules own the per-node flows (atoms, cells, fusions, boundaries).
//! Strategy-specific training loops that run inside them are grouped under
//! [`crate::programs`].

pub mod atom;
pub mod cell;
pub mod fusion;
pub mod warp;
