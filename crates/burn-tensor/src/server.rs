//! Remote-execution server entry points.
//!
//! Hosts a Burn server that executes tensor operations on behalf of remote
//! clients. The backend is selected by the [`Device`] passed in (the same
//! device handle used by tensor ops); the transport is selected by [`Channel`]. Iroh is the
//! primary transport and WebSocket is retained for compatibility.
//!
//! ```rust,ignore
//! use burn::{Device, server::{start_async, Channel, RemoteNode}};
//!
//! let node = RemoteNode::bind().await?;
//! start_async(Device::default(), Channel::Iroh { node }).await;
//! ```
//!
//! User-defined backends that implement `BackendIr` but aren't part of
//! `DispatchDevice` should call `burn_remote::server::start_iroh_async` (or the legacy
//! `start_websocket_async`) directly with the concrete backend type parameter.

use crate::Device;
pub use burn_dispatch::backends::remote::RemoteNode;

/// Transport used to serve remote clients.
#[derive(Debug, Clone)]
pub enum Channel {
    /// Iroh peer-to-peer transport. Direct paths are preferred and configured relays are used
    /// when direct connectivity is unavailable.
    Iroh {
        /// Process-level Iroh node, including identity, relays, and address lookup.
        node: RemoteNode,
    },
    /// WebSocket server bound to `0.0.0.0:port`.
    WebSocket {
        /// Port to bind on.
        port: u16,
    },
}

/// Start a remote-execution server, blocking the current thread.
///
/// The backend is determined by `device`: e.g. `Device::cuda(0)` runs ops on
/// CUDA, `Device::flex()` on the Flex CPU backend. Autodiff devices are
/// transparently stripped — the autodiff graph is a client-side concern.
///
/// # Panics
///
/// Panics if `device` selects a backend that doesn't support remote execution
/// (currently `LibTorch`, or a `Remote` device — hosting on a remote device
/// makes no sense).
pub fn start(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { node } => {
            burn_dispatch::remote_server::start_iroh(device.into_dispatch(), node)
        }
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket(device.into_dispatch(), port)
        }
    }
}

/// Start a remote-execution server on the caller's async runtime.
///
/// See [`start`] for backend-selection rules.
pub async fn start_async(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { node } => {
            burn_dispatch::remote_server::start_iroh_async(device.into_dispatch(), node).await
        }
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket_async(device.into_dispatch(), port).await
        }
    }
}
