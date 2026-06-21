//! Browser entry point: a small tensor REPL whose expressions run on a remote Iroh compute peer.
//!
//! Each submitted line is parsed and evaluated into tensor operations on the remote device;
//! only the value being displayed is read back, asynchronously. Variables persist across lines.

use alloc::format;
use alloc::string::{String, ToString};

use wasm_bindgen::prelude::*;

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::tensor::Device;

use crate::eval::{Env, Value, run_line};

/// Largest matrix slice rendered back to the page; bigger tensors are truncated with an ellipsis.
const MAX_SHOWN: usize = 8;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    let secret = SecretKey::from_bytes(hash.as_bytes());
    EndpointAddr::from(secret.public())
}

/// A REPL session bound to a remote compute peer.
#[wasm_bindgen]
pub struct Session {
    device: Device,
    env: Env,
}

#[wasm_bindgen]
impl Session {
    /// Connect to the compute peer reachable under `topic`.
    pub async fn connect(topic: String) -> Result<Session, String> {
        console_error_panic_hook::set_once();

        let node = RemoteNode::bind().await.map_err(|err| err.to_string())?;
        let device = Device::remote_iroh_async(&node, server_endpoint(&topic), 0).await;

        Ok(Self {
            device,
            env: Env::new(),
        })
    }

    /// Evaluate one line and return its rendered value (or an error message).
    pub async fn run(&mut self, line: String) -> Result<String, String> {
        let value = run_line(&line, &self.device, &mut self.env)?;
        render(value).await
    }
}

async fn render(value: Value) -> Result<String, String> {
    match value {
        Value::Scalar(s) => Ok(format_number(s)),
        Value::Tensor(tensor) => {
            let [rows, cols] = tensor.dims();
            let data = tensor
                .into_data_async()
                .await
                .map_err(|err| format!("failed to read tensor: {err:?}"))?;
            let values: alloc::vec::Vec<f32> = data.iter::<f32>().collect();
            Ok(format_matrix(&values, rows, cols))
        }
    }
}

fn format_number(value: f32) -> String {
    format!("{value:.4}")
}

fn format_matrix(values: &[f32], rows: usize, cols: usize) -> String {
    let shown_rows = rows.min(MAX_SHOWN);
    let shown_cols = cols.min(MAX_SHOWN);

    let mut out = format!("[{rows}x{cols}]\n");
    for r in 0..shown_rows {
        for c in 0..shown_cols {
            out.push_str(&format_number(values[r * cols + c]));
            out.push(' ');
        }
        if cols > shown_cols {
            out.push_str("...");
        }
        out.push('\n');
    }
    if rows > shown_rows {
        out.push_str("...\n");
    }
    out
}
