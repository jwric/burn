//! A Burn compute peer that runs in the browser.
//!
//! This is the server side of Burn Remote running in wasm: the tab brings up a WebGPU device, binds
//! an Iroh endpoint, and serves tensor operations submitted by remote clients. It is the mirror
//! image of the browser client examples — there the browser offloads work to a remote GPU; here the
//! browser *is* the GPU peer.
//!
//! It uses the umbrella [`burn::server::serve`] API, which returns immediately with a running server
//! (the accept loop runs on the JS event loop in the browser, a tokio runtime natively). The peer
//! derives its endpoint identity from a shared topic string, so a client that knows the same topic
//! addresses it directly.

use wasm_bindgen::prelude::*;

use burn::backend::remote::{Endpoint, RemoteNode, SecretKey};
use burn::server::{BURN_REMOTE_ALPN, Router, serve};
use burn::tensor::Device;
use iroh::endpoint::presets;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Derive a stable secret key from a topic, so the client can compute the matching endpoint id.
fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

/// A running compute peer. Dropping it stops serving.
#[wasm_bindgen]
pub struct ComputePeer {
    node: RemoteNode,
    _router: Router,
}

#[wasm_bindgen]
impl ComputePeer {
    /// Bring up a WebGPU device and start serving under `topic`.
    pub async fn start(topic: String) -> Result<ComputePeer, String> {
        console_error_panic_hook::set_once();

        let device = Device::wgpu_async(Default::default()).await;

        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(topic_key(&topic))
            .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
            .bind()
            .await
            .map_err(|err| err.to_string())?;

        let node = RemoteNode::from_endpoint(endpoint);
        let router = serve(device, node.clone());

        Ok(Self {
            node,
            _router: router,
        })
    }

    /// The peer's endpoint id, for clients that address it by id rather than topic.
    pub fn endpoint_id(&self) -> String {
        self.node.id().to_string()
    }
}
