//! Cross-target timers for the Iroh tensor-transfer service.
//!
//! Native targets use `tokio::time`; the browser has no tokio timer driver, so it uses the JS
//! timer through `gloo-timers`. Only the Iroh transfer needs these (capability TTL and download
//! wait), so the module is gated with it.

use core::{future::Future, time::Duration};

/// Wait for `duration` to elapse.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await;
}

/// Wait for `duration` to elapse.
#[cfg(target_family = "wasm")]
pub(crate) async fn sleep(duration: Duration) {
    gloo_timers::future::sleep(duration).await;
}

/// Run `future` to completion, returning `Err(())` if `duration` elapses first.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn timeout<F: Future>(duration: Duration, future: F) -> Result<F::Output, ()> {
    tokio::time::timeout(duration, future).await.map_err(|_| ())
}

/// Run `future` to completion, returning `Err(())` if `duration` elapses first.
#[cfg(target_family = "wasm")]
pub(crate) async fn timeout<F: Future>(duration: Duration, future: F) -> Result<F::Output, ()> {
    use futures_util::future::{Either, select};

    let timer = gloo_timers::future::sleep(duration);
    futures_util::pin_mut!(future, timer);
    match select(future, timer).await {
        Either::Left((output, _)) => Ok(output),
        Either::Right(((), _)) => Err(()),
    }
}
