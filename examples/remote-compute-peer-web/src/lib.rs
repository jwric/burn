//! A Burn compute peer that runs in the browser.
//!
//! This is the server side of Burn Remote running in wasm: the tab binds an Iroh endpoint and
//! serves tensor operations submitted by remote clients. It is the mirror image of the browser
//! client examples — there the browser offloads work to a remote peer; here the browser *is* the
//! peer.
//!
//! The peer derives its endpoint identity from a shared topic string, so a client that knows the
//! same topic addresses it directly (the same scheme the native `remote-compute-peer` uses).
//!
//! The served backend is Flex (CPU), which compiles for wasm today. Swapping to WebGPU is a
//! one-line change (serve `burn_wgpu::WebGpu` after `init_setup_async`), pending an upstream
//! `cubecl-runtime` wasm fix — see the README.

use wasm_bindgen::prelude::*;

use burn_flex::Flex;
use burn_remote::server::Router;
use burn_remote::{BURN_REMOTE_ALPN, Endpoint, RemoteNode, SecretKey};
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
    /// Start serving under `topic`.
    pub async fn start(topic: String) -> Result<ComputePeer, String> {
        console_error_panic_hook::set_once();

        let endpoint = Endpoint::builder(presets::N0)
            .secret_key(topic_key(&topic))
            .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
            .bind()
            .await
            .map_err(|err| err.to_string())?;

        let node = RemoteNode::from_endpoint(endpoint);
        let router = node.serve::<Flex>(vec![Default::default()]);

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
