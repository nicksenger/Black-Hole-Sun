//! Boundary effects reused from cell transport/wait primitives.

pub use crate::cell::effect::{
    Transmit as TransmitEffect, WaitForPotentiationEffect as WaitForBoundaryPotentiationEffect,
    WaitForPropagationEffect,
};
