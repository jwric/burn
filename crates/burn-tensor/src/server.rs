//! Remote-execution server entry points.
//!
//! Hosts a Burn server that executes tensor operations on behalf of remote
//! clients. The backend is selected by the [`Device`] passed in (the same
//! device handle used by tensor ops); the transport is selected by [`Channel`]. Iroh is the
//! primary transport and WebSocket is retained for compatibility.
//!
//! ```rust,ignore
//! use burn::{Device, server::{serve, RemoteNode}};
//!
//! let node = RemoteNode::bind().await?;
//! let _router = serve(Device::default(), node);
//! ```
//!
//! [`serve`] is the portable entry and works in the browser. The blocking [`start`] / [`start_async`]
//! helpers and the WebSocket transport are native-only.
//!
//! User-defined backends that implement `BackendIr` but aren't part of
//! `DispatchDevice` should call `burn_remote::server::serve` directly with the concrete backend.

use crate::Device;
pub use burn_dispatch::backends::remote::RemoteNode;
pub use burn_dispatch::backends::remote::server::{Router, RouterBuilder};
pub use burn_dispatch::backends::remote::telemetry;
pub use burn_dispatch::devices::BURN_REMOTE_ALPN;

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
    #[cfg(feature = "remote")]
    WebSocket {
        /// Port to bind on.
        port: u16,
    },
}

/// Start a non-blocking Iroh server and return its [`Router`]; drop it to stop serving.
///
/// The accept loop runs on the ambient executor (a tokio runtime on native, the event loop in the
/// browser), so unlike [`start_async`] this also works in wasm. See [`start`] for backend selection.
pub fn serve(device: Device, node: RemoteNode) -> Router {
    burn_dispatch::remote_server::serve_iroh(device.into_dispatch(), node)
}

/// Like [`serve`], emitting per-session telemetry into `probe` for live monitoring.
///
/// Pair with [`telemetry::TelemetryProbe::channel`] to obtain a subscription a dashboard can
/// drain. Works in the browser and natively.
pub fn serve_with_telemetry(
    device: Device,
    node: RemoteNode,
    probe: telemetry::TelemetryProbe,
) -> Router {
    burn_dispatch::remote_server::serve_iroh_with_telemetry(device.into_dispatch(), node, probe)
}

/// Build the Iroh [`RouterBuilder`] pre-loaded with the Burn Remote compute protocol, without
/// spawning it.
///
/// Use this to share one endpoint between Burn Remote and other Iroh protocols — e.g. iroh-gossip
/// for peer discovery. Register the extra protocols on the returned builder, then call `.spawn()`:
///
/// ```rust,ignore
/// let gossip = Gossip::builder().spawn(endpoint.clone());
/// let router = serve_builder(device, node.clone())
///     .accept(GOSSIP_ALPN, gossip.clone())
///     .spawn();
/// ```
///
/// Works in the browser and natively.
pub fn serve_builder(device: Device, node: RemoteNode) -> RouterBuilder {
    burn_dispatch::remote_server::serve_iroh_builder(device.into_dispatch(), node)
}

/// Like [`serve_builder`], emitting per-session telemetry into `probe`.
pub fn serve_builder_with_telemetry(
    device: Device,
    node: RemoteNode,
    probe: telemetry::TelemetryProbe,
) -> RouterBuilder {
    burn_dispatch::remote_server::serve_iroh_builder_with_telemetry(
        device.into_dispatch(),
        node,
        probe,
    )
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
#[cfg(not(target_family = "wasm"))]
pub fn start(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { node } => {
            burn_dispatch::remote_server::start_iroh(device.into_dispatch(), node)
        }
        #[cfg(feature = "remote")]
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket(device.into_dispatch(), port)
        }
    }
}

/// Start a remote-execution server on the caller's async runtime.
///
/// See [`start`] for backend-selection rules.
#[cfg(not(target_family = "wasm"))]
pub async fn start_async(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { node } => {
            burn_dispatch::remote_server::start_iroh_async(device.into_dispatch(), node).await
        }
        #[cfg(feature = "remote")]
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket_async(device.into_dispatch(), port).await
        }
    }
}
