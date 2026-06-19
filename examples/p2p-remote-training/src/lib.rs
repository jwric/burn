use burn_remote::{BURN_REMOTE_ALPN, EndpointAddr, EndpointId, RemoteNode, SecretKey};
use burn_tensor::{Device, Distribution, Tensor};
use iroh::endpoint::presets;
use tracing_subscriber::{EnvFilter, fmt};

fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,burn_remote=debug"));

    fmt().with_env_filter(filter).init();
}

fn topic_key(topic: &str) -> SecretKey {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes())
}

// Bind a RemoteNode whose identity is fixed to the given topic string.
// The N0 preset auto-publishes the relay address to pkarr DNS so the client
// can discover it with only the NodeId.
async fn bind_with_topic(topic: &str) -> RemoteNode {
    use iroh::Endpoint;

    let secret = topic_key(topic);
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
        .bind()
        .await
        .expect("bind failed");
    RemoteNode::from_endpoint(endpoint)
}

pub fn run_server(topic: &str) {
    use burn_flex::Flex;
    use burn_remote::server::start_iroh_async;

    init_logging();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let node = bind_with_topic(topic).await;

        tracing::info!(topic, node_id = %node.id(), "server ready");
        tracing::info!("waiting for clients (press Ctrl-C to stop)");

        start_iroh_async::<Flex>(node, vec![Default::default()]).await;

        tracing::info!("server stopped");
    });
}

pub fn run_client(topic: &str) {
    let server_id: EndpointId = topic_key(topic).public();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        println!("topic     : {topic}");
        println!("server id : {server_id}");
        println!("connecting...");

        let node = RemoteNode::bind().await.expect("bind failed");
        let remote = node.device(EndpointAddr::from(server_id), 0);
        remote.connect();

        println!("connected\n");
        train(&Device::new(remote));
    });
}

fn train(device: &Device) {
    const N: usize = 512;
    const STEPS: usize = 80;
    const LR: f32 = 0.08;

    println!("target: y = 2.5 * x + 0.5");
    println!("steps : {STEPS}  samples: {N}\n");

    let x = Tensor::<1>::random([N], Distribution::Default, device) * 2.0 - 1.0;
    let y_true = x.clone() * 2.5 + 0.5;

    let mut w = Tensor::<1>::from_floats([0.0_f32], device);
    let mut b = Tensor::<1>::from_floats([0.0_f32], device);

    println!("{:>5}  {:>10}", "step", "loss");

    for step in 0..STEPS {
        let y_pred = x.clone() * w.clone().expand([N]) + b.clone().expand([N]);
        let error = y_pred - y_true.clone();
        let loss = (error.clone() * error.clone()).mean();
        let dw = (x.clone() * error.clone()).mean() * 2.0_f32;
        let db = error.mean() * 2.0_f32;

        w = w - dw * LR;
        b = b - db * LR;

        if step % 10 == 0 || step == STEPS - 1 {
            let loss_val = loss.to_data().to_vec::<f32>().unwrap()[0];
            println!("{:>5}  {:>10.6}", step + 1, loss_val);
        }
    }

    let w_val = w.to_data().to_vec::<f32>().unwrap()[0];
    let b_val = b.to_data().to_vec::<f32>().unwrap()[0];

    println!("\nlearned: y = {w_val:.4} * x + {b_val:.4}");
    println!("target : y = 2.5000 * x + 0.5000");
}
