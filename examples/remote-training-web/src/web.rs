//! Browser entry point: train a small MLP whose forward pass, gradients and optimizer updates all
//! run on a remote Iroh compute peer. The autodiff graph is built client-side; only scalar loss
//! values are read back to drive the live chart.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{GradientsParams, ModuleOptimizer, SgdConfig};
use burn::tensor::{Device, Distribution, Tensor};

use crate::model::Mlp;

const SAMPLES: usize = 256;
/// Target curve is `y = sin(FREQ * x)`.
const FREQ: f32 = 3.0;
const HIDDEN: usize = 32;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    let secret = SecretKey::from_bytes(hash.as_bytes());
    EndpointAddr::from(secret.public())
}

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
    /// Connect to the peer under `topic`, then build the autodiff model and training data on it.
    pub async fn connect(topic: String, learning_rate: f64) -> Result<RemoteTrainer, String> {
        console_error_panic_hook::set_once();

        let node = RemoteNode::bind().await.map_err(|err| err.to_string())?;
        let device = Device::remote_iroh_async(&node, server_endpoint(&topic), 0)
            .await
            .autodiff();

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
