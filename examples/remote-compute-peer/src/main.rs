//! A Burn Remote compute peer reachable over Iroh.
//!
//! The peer derives its endpoint identity from a shared topic string, so a client that knows the
//! same topic can reach it without copying a node id around. Tensor operations submitted by
//! clients run on this process's backend (CPU `flex` by default, or `wgpu` with `--features wgpu`).
//!
//! Build with `--features dashboard` to open a live egui telemetry window while serving.

use std::sync::Arc;

use burn::server::{BURN_REMOTE_ALPN, RemoteNode};
use burn::tensor::Device;
use iroh::{Endpoint, SecretKey, endpoint::presets};
use tracing_subscriber::{EnvFilter, fmt};

/// Derive a stable secret key from a topic. The client derives the matching public endpoint id
/// from the same string, which is how the two sides find each other.
fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

fn compute_device() -> Device {
    cfg_select! {
        feature = "cuda" => {
            Device::cuda(0)
        },
        feature = "wgpu" => {
            Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
        },
        _ => {
            Device::flex()
        }
    }
}

fn main() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,burn_remote=debug"));
    fmt().with_env_filter(filter).init();

    let topic = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "burn-web".to_string());

    let runtime = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Can build the Tokio runtime"),
    );

    let node = runtime.block_on(async {
        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(topic_key(topic.as_str()))
            .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
            .bind()
            .await
            .expect("Failed to bind Iroh endpoint");
        RemoteNode::from_endpoint(endpoint)
    });

    tracing::info!(topic, node_id = %node.id(), "compute peer ready");
    tracing::info!("clients reach this peer with the same topic string");

    run(runtime, node, topic);
}

#[cfg(not(feature = "dashboard"))]
fn run(runtime: Arc<tokio::runtime::Runtime>, node: RemoteNode, _topic: String) {
    use burn::server::Channel;
    runtime.block_on(burn::server::start_async(
        compute_device(),
        Channel::Iroh { node },
    ));
}

#[cfg(feature = "dashboard")]
fn run(runtime: Arc<tokio::runtime::Runtime>, node: RemoteNode, _topic: String) {
    use burn::server::{serve_with_telemetry, telemetry::TelemetryProbe};

    let addr = std::env::var("BURN_DASHBOARD_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let probe = TelemetryProbe::new(8192);
    let _router = runtime.block_on(async {
        serve_with_telemetry(compute_device(), node.clone(), probe.clone())
    });

    let snapshots = std::sync::Arc::new(
        tokio::sync::watch::channel(remote_compute_dashboard::DashboardState::default()).0,
    );
    runtime.spawn(dashboard::aggregate(probe, node, snapshots.clone()));

    tracing::info!("dashboard on http://{addr}");
    runtime.block_on(dashboard::serve(snapshots, addr));
}

#[cfg(feature = "dashboard")]
mod dashboard {
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use axum::extract::State;
    use axum::http::{Uri, header::CONTENT_TYPE};
    use axum::response::sse::{Event, KeepAlive, Sse};
    use axum::response::{IntoResponse, Response};
    use axum::routing::get;
    use burn::server::RemoteNode;
    use burn::server::telemetry::TelemetryProbe;
    use futures_core::Stream;
    use remote_compute_dashboard::{Aggregator, DashboardState};
    use include_dir::{Dir, include_dir};
    use tokio::sync::watch;

    static VIEWER: Dir<'static> = include_dir!("$OUT_DIR/dashboard-dist");

    type Snapshots = Arc<watch::Sender<DashboardState>>;

    #[derive(Clone)]
    struct AppState {
        snapshots: Snapshots,
    }

    /// Drain telemetry into an [`Aggregator`] and publish the current [`DashboardState`] a few
    /// times a second. The state lives here, so a viewer that connects or reconnects gets the
    /// running picture instead of a fresh start.
    pub async fn aggregate(probe: TelemetryProbe, node: RemoteNode, snapshots: Snapshots) {
        let Some(mut subscription) = probe.subscribe() else {
            return;
        };
        let mut aggregator = Aggregator::new(node.id().to_string());
        let start = Instant::now();
        let mut ticker = tokio::time::interval(Duration::from_millis(150));
        loop {
            tokio::select! {
                event = subscription.recv() => match event {
                    Some(event) => aggregator.apply(&event),
                    None => break,
                },
                _ = ticker.tick() => {
                    let now_ms = start.elapsed().as_secs_f64() * 1000.0;
                    aggregator.set_peers(&node.peer_snapshot().await, now_ms);
                    aggregator.tick(now_ms);
                    let _ = snapshots.send_replace(aggregator.snapshot(now_ms));
                }
            }
        }
    }

    pub async fn serve(snapshots: Snapshots, addr: String) {
        let app = axum::Router::new()
            .route("/events", get(events))
            .fallback(get(viewer))
            .with_state(AppState { snapshots });

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .expect("Failed to bind the dashboard HTTP listener");

        // Don't use graceful shutdown: the SSE stream never completes, so waiting for in-flight
        // requests to drain would hang on Ctrl-C. Abort the server when the signal fires instead.
        let server = std::future::IntoFuture::into_future(axum::serve(listener, app));
        tokio::select! {
            result = server => result.expect("Dashboard HTTP server failed"),
            _ = shutdown_signal() => tracing::info!("Ctrl-C received, stopping dashboard"),
        }
    }

    async fn viewer(uri: Uri) -> Response {
        let path = match uri.path().trim_start_matches('/') {
            "" => "index.html",
            path => path,
        };
        match VIEWER.get_file(path) {
            Some(file) => (
                [(CONTENT_TYPE, content_type(path))],
                axum::body::Bytes::from_static(file.contents()),
            )
                .into_response(),
            None => axum::http::StatusCode::NOT_FOUND.into_response(),
        }
    }

    fn content_type(path: &str) -> &'static str {
        if path.ends_with(".html") {
            "text/html; charset=utf-8"
        } else if path.ends_with(".js") {
            "text/javascript; charset=utf-8"
        } else if path.ends_with(".wasm") {
            "application/wasm"
        } else {
            "application/octet-stream"
        }
    }

    async fn events(
        State(state): State<AppState>,
    ) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let mut rx = state.snapshots.subscribe();
        let stream = async_stream::stream! {
            loop {
                let data = serde_json::to_string(&*rx.borrow_and_update()).unwrap_or_default();
                yield Ok(Event::default().data(data));
                if rx.changed().await.is_err() {
                    break;
                }
            }
        };
        Sse::new(stream).keep_alive(KeepAlive::default())
    }

    async fn shutdown_signal() {
        let _ = tokio::signal::ctrl_c().await;
    }
}
