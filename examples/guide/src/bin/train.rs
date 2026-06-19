#![recursion_limit = "131"]
use burn::server::RemoteNode;
use burn::{data::dataset::Dataset, optim::AdamConfig, prelude::*};
use guide::{
    inference,
    model::ModelConfig,
    training::{self, TrainingConfig},
};
use iroh::{EndpointAddr, EndpointId, SecretKey};

fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

#[tokio::main]
async fn main() {
    let topic = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "marc".to_string());

    let server_id: EndpointId = topic_key(&topic).public();

    println!("topic     : {topic}");
    println!("server id : {server_id}");
    println!("connecting...");

    let node = RemoteNode::bind().await.expect("bind failed");
    let device = Device::remote_iroh(&node, EndpointAddr::from(server_id), 0);

    println!("connected\n");

    // All the training artifacts will be saved in this directory
    let artifact_dir = "target/guide";

    // Train the model
    training::train(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 4096 * 2), AdamConfig::new()),
        device.clone(),
    );

    // Infer the model
    inference::infer(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
