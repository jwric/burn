#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod eval;

#[cfg(target_family = "wasm")]
pub mod web;
