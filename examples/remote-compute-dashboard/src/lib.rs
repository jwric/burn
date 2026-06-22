//! Shared dashboard code for the Burn Remote compute peers.
//!
//! [`Aggregator`] folds raw `burn-remote` telemetry into a current [`DashboardState`] (no UI
//! dependency, so a headless server can run it and serialize the result). The `render` feature
//! adds the egui [`Dashboard`] that draws a [`DashboardState`], shared by the browser peer (fed
//! in-process) and the HTTP viewer (fed over SSE).

mod state;
pub use state::*;

pub use burn_remote::telemetry;

#[cfg(feature = "render")]
mod render;
#[cfg(feature = "render")]
pub use render::*;
