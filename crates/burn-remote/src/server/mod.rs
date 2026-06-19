pub(crate) mod local_comm;
pub(crate) mod service;
pub(crate) mod session;
pub(crate) mod transfer;
pub(crate) mod worker;

mod base;
#[cfg(feature = "iroh")]
mod iroh;

#[cfg(feature = "websocket")]
pub use base::{start_websocket, start_websocket_async};
#[cfg(feature = "iroh")]
pub use iroh::{
    AuthorizationRequest, IrohRemoteProtocol, PeerAuthorizer, start_iroh, start_iroh_async,
};
