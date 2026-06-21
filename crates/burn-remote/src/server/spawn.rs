//! Detached task spawning for the server's session and transfer streams.
//!
//! Mirrors the client's executor split: on native targets tasks run on the Tokio runtime that owns
//! the Iroh endpoint and must be `Send`; in the browser there is no such runtime, so tasks run on
//! the JS event loop through [`wasm_bindgen_futures::spawn_local`] and are not required to be
//! `Send`. Callers that need to observe completion pair this with a [`oneshot`](tokio::sync::oneshot)
//! channel rather than joining a handle, so the two targets share one code path.

use core::future::Future;

/// Spawn `future` to run independently of the caller.
#[cfg(not(target_family = "wasm"))]
pub(crate) fn spawn_detached<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    tokio::spawn(future);
}

/// Spawn `future` on the browser event loop.
#[cfg(target_family = "wasm")]
pub(crate) fn spawn_detached<F>(future: F)
where
    F: Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}
