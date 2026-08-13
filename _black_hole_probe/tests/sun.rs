#![allow(dead_code, unused_imports, clippy::manual_async_fn)]

#[path = "sun/mod.rs"]
mod sun_impl;

#[allow(unused_imports)]
pub(crate) use sun_impl::*;
