//! Notebook-style Burn scripting against a remote compute peer.
//!
//! Each block below maps to a notebook cell (see `notebook.ipynb`). The point is that none of this
//! needs `async` or `#[tokio::main]`: `RemoteNode::bind_blocking` owns its runtime, so tensor
//! operations run synchronously on the remote peer just like local ones — which is what makes Burn
//! usable from a Rust REPL or Jupyter (evcxr) kernel.
//!
//! Start a peer first, e.g. `cargo run -p remote-compute-peer -- burn-web`, then
//! `cargo run -p remote-notebook -- burn-web`.

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::tensor::{Device, Distribution, Tensor};

/// Derive the peer's endpoint identity from a shared topic string.
fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    EndpointAddr::from(SecretKey::from_bytes(hash.as_bytes()).public())
}

fn main() {
    let topic = std::env::args().nth(1).unwrap_or_else(|| "burn-web".to_string());

    // Cell 1 — connect. No runtime setup, no `.await`.
    let node = RemoteNode::bind_blocking().expect("failed to bind local endpoint");
    let device = Device::remote_iroh(&node, server_endpoint(&topic), 0);
    println!("connected to '{topic}'\n");

    // Cell 2 — create a tensor on the peer and read it back.
    let a = Tensor::<2>::random([3, 4], Distribution::Default, &device);
    println!("a =\n{a}\n");

    // Cell 3 — operations run on the peer; only the printed value is read back.
    let b = Tensor::<2>::random([4, 2], Distribution::Default, &device);
    let c = a.matmul(b);
    println!("a @ b =\n{c}\n");

    // Cell 4 — reductions, still on the peer.
    let mean = c.mean();
    println!("mean(a @ b) = {}", mean.into_data());
}
