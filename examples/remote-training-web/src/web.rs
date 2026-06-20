//! Browser entry point: train a small MLP whose forward pass, gradients and optimizer updates all
//! run on a remote Iroh compute peer.
//!
//! The autodiff graph is built on the client, but every tensor operation it records — including the
//! backward pass and the optimizer step — is executed on the peer's backend. Only scalar loss
//! values are read back, asynchronously, to drive the live chart.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{GradientsParams, ModuleOptimizer, SgdConfig};
use burn::tensor::{Device, Distribution, Tensor};

use crate::model::Mlp;

/// Number of training samples generated on the peer.
const SAMPLES: usize = 256;
/// Frequency of the target function `y = sin(FREQ * x)`.
const FREQ: f32 = 3.0;
/// Hidden width of the MLP.
const HIDDEN: usize = 32;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Derive the compute peer's endpoint identity from a shared topic string (see the inference demo
/// for the full explanation).
fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    let secret = SecretKey::from_bytes(hash.as_bytes());
    EndpointAddr::from(secret.public())
}

/// A regression model trained on a remote compute peer.
#[wasm_bindgen]
pub struct RemoteTrainer {
    model: Option<Mlp>,
    optim: ModuleOptimizer,
    inputs: Tensor<2>,
    targets: Tensor<2>,
    lr: f64,
}

#[wasm_bindgen]
impl RemoteTrainer {
    /// Connect to the compute peer reachable under `topic`, then build the model and training data
    /// on it. The model is wrapped for automatic differentiation so `backward` is available.
    pub async fn connect(topic: String, learning_rate: f64) -> Result<RemoteTrainer, String> {
        console_error_panic_hook::set_once();

        let node = RemoteNode::bind().await.map_err(|err| err.to_string())?;
        let device = Device::remote_iroh_async(&node, server_endpoint(&topic), 0)
            .await
            .autodiff();

        // Inputs in [-1, 1] and the target curve, both materialized on the peer.
        let inputs = Tensor::<2>::random([SAMPLES, 1], Distribution::Default, &device) * 2.0 - 1.0;
        let targets = (inputs.clone() * FREQ).sin();

        Ok(Self {
            model: Some(Mlp::new(HIDDEN, &device)),
            optim: SgdConfig::new().init(),
            inputs,
            targets,
            lr: learning_rate,
        })
    }

    /// Run `steps` optimizer steps and return the loss recorded after each one.
    pub async fn train(&mut self, steps: usize) -> Result<Vec<f32>, String> {
        let mut history = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut model = self.model.take().expect("model is always restored below");

            let prediction = model.forward(self.inputs.clone());
            let loss = MseLoss::new().forward(prediction, self.targets.clone(), Reduction::Mean);

            let value = loss
                .clone()
                .into_data_async()
                .await
                .map_err(|err| format!("Failed to read loss: {err:?}"))?;
            history.push(value.iter::<f32>().next().unwrap_or(f32::NAN));

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = self.optim.step(self.lr, model, grads);

            self.model = Some(model);
        }

        Ok(history)
    }
}
