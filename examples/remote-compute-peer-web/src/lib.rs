//! A Burn compute peer that runs in the browser: it brings up a WebGPU device and serves tensor
//! operations to remote clients over Iroh, through the umbrella `burn::server::serve` API. The
//! mirror of the browser client examples — here the browser is the GPU peer.

use wasm_bindgen::prelude::*;

use burn::backend::remote::{Endpoint, RemoteNode, SecretKey};
use burn::server::{BURN_REMOTE_ALPN, Router, serve};
use burn::tensor::Device;
use iroh::endpoint::presets;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

/// Dropping it stops serving.
#[wasm_bindgen]
pub struct ComputePeer {
    node: RemoteNode,
    _router: Router,
}

#[wasm_bindgen]
impl ComputePeer {
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

    pub fn endpoint_id(&self) -> String {
        self.node.id().to_string()
    }
}
