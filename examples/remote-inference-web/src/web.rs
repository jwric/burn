//! Browser entry point: a MNIST classifier whose tensor operations run on a remote Iroh compute
//! peer rather than in the browser.
//!
//! The model is defined and its weights are loaded on the client, but every operation is shipped
//! to the compute peer and executed on its backend (CPU or GPU). Only the small input and the
//! output probabilities cross the wire as tensor data.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::{Bytes, Device, Tensor, activation::softmax};

use crate::model::Model;

/// Trained MNIST parameters in the burnpack format, produced by the `mnist` example.
static STATE_ENCODED: &[u8] = include_bytes!("../model.bpk");

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Derive the compute peer's endpoint identity from a shared topic string.
///
/// The peer binds its endpoint with the secret key derived from the same string, so the public
/// half computed here addresses it directly — no node id needs to be copied between the two sides.
fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    let secret = SecretKey::from_bytes(hash.as_bytes());
    EndpointAddr::from(secret.public())
}

/// A MNIST classifier bound to a remote compute peer.
#[wasm_bindgen]
pub struct RemoteMnist {
    device: Device,
    model: Model,
}

#[wasm_bindgen]
impl RemoteMnist {
    /// Connect to the compute peer reachable under `topic` and load the model onto it.
    ///
    /// Binds a browser-side Iroh endpoint, opens the session to the peer, then builds the model
    /// and loads its weights — which streams the parameters to the peer as tensor data.
    pub async fn connect(topic: String) -> Result<RemoteMnist, String> {
        console_error_panic_hook::set_once();

        let node = RemoteNode::bind().await.map_err(|err| err.to_string())?;
        let device = Device::remote_iroh_async(&node, server_endpoint(&topic), 0).await;

        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
            .map_err(|err| format!("Failed to decode model weights: {err}"))?;
        let model = Model::new(&device).load_record(record);

        Ok(Self { device, model })
    }

    /// Classify a 28x28 grayscale image, returning the 10 class probabilities.
    ///
    /// `input` is a row-major `f32` slice of length 784 with pixel values in `[0, 255]`.
    pub async fn inference(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let input = Tensor::<1>::from_floats(input, &self.device).reshape([1, 28, 28]);

        // Match the normalization used during training (mean/std from the PyTorch MNIST example).
        let input = ((input / 255) - 0.1307) / 0.3081;

        let output = softmax(self.model.forward(input), 1);
        let data = output
            .into_data_async()
            .await
            .map_err(|err| format!("Failed to read inference result: {err:?}"))?;

        Ok(data.iter::<f32>().collect())
    }
}
