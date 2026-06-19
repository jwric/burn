#![cfg(all(feature = "client", feature = "server", feature = "iroh"))]

use burn_flex::Flex;
use burn_remote::{BURN_REMOTE_ALPN, RemoteNode};
use burn_tensor::{Device, Tensor};
use iroh::{Endpoint, RelayMode, endpoint::presets, protocol::Router};

async fn local_node() -> RemoteNode {
    let endpoint = Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .clear_ip_transports()
        .bind_addr("127.0.0.1:0")
        .unwrap()
        .bind()
        .await
        .unwrap();
    RemoteNode::from_endpoint(endpoint)
}

#[tokio::test(flavor = "multi_thread")]
async fn executes_over_iroh_session_stream() {
    let server = local_node().await;
    let client = local_node().await;
    let router = server.serve::<Flex>(vec![Default::default()]);

    let remote = client.device(server.endpoint().addr(), 0);
    remote.connect();
    let device = Device::new(remote);

    let output = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device) * 2.0;
    assert_eq!(
        output.to_data().to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn transfers_tensor_directly_between_iroh_compute_peers() {
    let source_server = local_node().await;
    let target_server = local_node().await;
    let client = local_node().await;

    let source_router = source_server.serve::<Flex>(vec![Default::default()]);
    let target_router = target_server.serve::<Flex>(vec![Default::default()]);

    let source_remote = client.device(source_server.endpoint().addr(), 0);
    let target_remote = client.device(target_server.endpoint().addr(), 0);
    source_remote.connect();
    target_remote.connect();
    let source = Device::new(source_remote);
    let target = Device::new(target_remote);

    let tensor = Tensor::<1>::from_floats([3.0, 5.0, 7.0], &source);
    let transferred = tensor.to_device(&target);
    assert_eq!(
        transferred.to_data().to_vec::<f32>().unwrap(),
        vec![3.0, 5.0, 7.0]
    );

    source_router.shutdown().await.unwrap();
    target_router.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn passes_application_credentials_to_the_peer_authorizer() {
    let server = local_node().await;
    let client = local_node().await;
    let protocol = server
        .protocol::<Flex>(vec![Default::default()])
        .with_authorizer(|request: burn_remote::server::AuthorizationRequest<'_>| {
            (request.credential == b"fleet-ticket")
                .then_some(())
                .ok_or_else(|| "invalid fleet ticket".to_string())
        });
    let router = Router::builder(server.endpoint().clone())
        .accept(BURN_REMOTE_ALPN, protocol)
        .spawn();

    let remote = client.device_authorized(server.endpoint().addr(), 0, b"fleet-ticket".to_vec());
    remote.connect();
    let device = Device::new(remote);
    let data = Tensor::<1>::from_floats([4.0], &device).to_data();
    assert_eq!(data.to_vec::<f32>().unwrap(), vec![4.0]);

    router.shutdown().await.unwrap();
}
