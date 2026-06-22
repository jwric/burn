//! Notebook-style Burn scripting against a remote compute peer. Each block maps to a cell in
//! `notebook.ipynb`; `RemoteNode::bind_blocking` owns its runtime so none of it needs `async`.
//!
//! Start a peer first, e.g. `cargo run -p remote-compute-peer -- burn-web`, then
//! `cargo run -p remote-notebook -- burn-web`.

use burn::backend::remote::{EndpointAddr, RemoteNode, SecretKey};
use burn::tensor::{Device, Distribution, Tensor};

fn server_endpoint(topic: &str) -> EndpointAddr {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    EndpointAddr::from(SecretKey::from_bytes(hash.as_bytes()).public())
}

fn main() {
    let topic = std::env::args().nth(1).unwrap_or_else(|| "burn-web".to_string());

    let node = RemoteNode::bind_blocking().expect("failed to bind local endpoint");
    let device = Device::remote_iroh(&node, server_endpoint(&topic), 0);
    println!("connected to '{topic}'\n");

    let a = Tensor::<2>::random([3, 4], Distribution::Default, &device);
    println!("a =\n{a}\n");

    let b = Tensor::<2>::random([4, 2], Distribution::Default, &device);
    let c = a.matmul(b);
    println!("a @ b =\n{c}\n");

    let mean = c.mean();
    println!("mean(a @ b) = {}", mean.into_data());
}
