pub(crate) mod local_comm;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod spawn;
#[cfg(feature = "iroh")]
pub(crate) mod time;
pub(crate) mod transfer;
pub(crate) mod worker;

mod base;
#[cfg(feature = "iroh")]
mod iroh;

#[cfg(feature = "websocket")]
pub use base::{start_websocket, start_websocket_async};
#[cfg(feature = "iroh")]
pub use iroh::{AuthorizationRequest, IrohRemoteProtocol, PeerAuthorizer};
// The blocking process entry points exist only on native targets; the browser server is driven by
// the JS event loop and composed through `RemoteNode::serve` / `protocol` directly.
#[cfg(all(feature = "iroh", not(target_family = "wasm")))]
pub use iroh::{start_iroh, start_iroh_async};
