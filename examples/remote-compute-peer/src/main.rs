//! A Burn Remote compute peer reachable over Iroh.
//!
//! The peer derives its endpoint identity from a shared topic string, so a client that knows the
//! same topic can reach it without copying a node id around. Tensor operations submitted by
//! clients run on this process's backend (CPU `flex` by default, or `wgpu` with `--features wgpu`).

use burn::server::{BURN_REMOTE_ALPN, Channel, RemoteNode};
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

#[tokio::main]
async fn main() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,burn_remote=debug"));
    fmt().with_env_filter(filter).init();

    let topic = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "burn-web".to_string());
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(topic_key(topic.as_str()))
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
        .bind()
        .await
        .expect("Failed to bind Iroh endpoint");

    let node = RemoteNode::from_endpoint(endpoint);
    tracing::info!(topic, node_id = %node.id(), "compute peer ready");
    tracing::info!("clients reach this peer with the same topic string (Ctrl-C to stop)");

    burn::server::start_async(compute_device(), Channel::Iroh { node }).await;
}
