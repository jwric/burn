#![cfg(feature = "websocket")]

use std::sync::Arc;

use burn_backend::tensor::Device;
use burn_ir::BackendIr;
use tokio_util::sync::CancellationToken;

use burn_communication::{
    ProtocolServer,
    external_comm::{ExternalCommServer, ExternalCommService},
    util::os_shutdown_signal,
    websocket::{WebSocket, WsServer},
};

use super::{
    service::{FetchHandler, SubmitHandler},
    session::SessionManager,
};

use super::transfer::WebSocketTransfer;

/// Start a legacy WebSocket compute node on the given port.
pub async fn start_websocket_async<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    let cancel_token = CancellationToken::new();
    let external = Arc::new(ExternalCommService::<B, WebSocket>::new(cancel_token));
    let transfer = Arc::new(WebSocketTransfer {
        inner: external.clone(),
    });
    let sessions = Arc::new(SessionManager::new(devices, transfer));

    let server = WsServer::new(port)
        .route("/fetch", {
            let sessions = sessions.clone();
            move |stream| FetchHandler::new(sessions, stream).run()
        })
        .route("/submit", {
            let sessions = sessions.clone();
            move |stream| SubmitHandler::new(sessions, stream).run()
        })
        .route_external_comm(external);

    if let Err(err) = server.serve(os_shutdown_signal()).await {
        log::error!("Burn Remote WebSocket server stopped: {err:?}");
    }
}

#[tokio::main]
/// Start a legacy WebSocket compute node, blocking the current thread.
pub async fn start_websocket<B: BackendIr>(devices: Vec<Device<B>>, port: u16) {
    start_websocket_async::<B>(devices, port).await;
}
