//! A native Burn compute swarm in one binary: `serve` runs a compute peer that joins the swarm,
//! `client` discovers peers and fans a Mandelbrot across them. The native twin of the browser peer.
//!
//! ```sh
//! cargo run -p remote-swarm-cluster -- --local serve burn-web alice   # seed; prints a ticket
//! cargo run -p remote-swarm-cluster -- --local serve <ticket> bob
//! cargo run -p remote-swarm-cluster -- --local client <ticket>
//! ```
//!
//! A seed may take a landing URL (the browser-peer page); it then prints a `<url>#<ticket>` QR for
//! phones to scan and join as GPU peers:
//!
//! ```sh
//! cargo run -p remote-swarm-cluster -- serve burn-office laptop https://peer.example.com
//! ```

use std::time::Duration;

use anyhow::{Context, Result, bail};
use burn::server::{BURN_REMOTE_ALPN, RemoteNode, serve_builder};
use burn::tensor::{Device, Int, Tensor};
use iroh::endpoint::presets;
use iroh::{Endpoint, EndpointAddr};
use remote_swarm::{
    GOSSIP_ALPN, Gossip, JoinTicket, PeerAdvert, PeerCaps, RemoteTicket, RosterEntry, Swarm,
    SwarmConfig, TopicId, topic_from_label,
};
use tracing_subscriber::{EnvFilter, fmt};

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,swarm=info")),
        )
        .init();

    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let local = raw.iter().any(|a| a == "--local");
    raw.retain(|a| a != "--local");
    let mut args = raw.into_iter();

    match args.next().unwrap_or_default().as_str() {
        "serve" => {
            let target = args.next().context(
                "usage: swarm-cluster [--local] serve <label-or-ticket> [name] [landing-url]",
            )?;
            let name = args.next().unwrap_or_else(|| "peer".to_string());
            let landing = args.next();
            run_serve(&target, &name, landing.as_deref(), local).await
        }
        "client" => {
            let target = args
                .next()
                .context("usage: swarm-cluster [--local] client <label-or-ticket>")?;
            run_client(&target, local).await
        }
        other => bail!(
            "unknown role {other:?}\n\nusage (add --local for a relay-free same-host run):\n  swarm-cluster [--local] serve <label-or-ticket> [name]\n  swarm-cluster [--local] client <label-or-ticket>"
        ),
    }
}

/// `--local` uses iroh's relay-free `Minimal` preset (direct paths only); otherwise N0 + relays.
async fn build_endpoint(local: bool, alpns: Vec<Vec<u8>>) -> Result<Endpoint> {
    if local {
        Ok(Endpoint::builder(presets::Minimal)
            .alpns(alpns)
            .bind()
            .await?)
    } else {
        let endpoint = Endpoint::builder(presets::N0).alpns(alpns).bind().await?;
        endpoint.online().await;
        Ok(endpoint)
    }
}

fn parse_target(input: &str) -> (TopicId, Vec<EndpointAddr>) {
    match JoinTicket::decode(input) {
        Ok(ticket) => (ticket.topic(), ticket.bootstrap().to_vec()),
        Err(_) => (topic_from_label(input), Vec::new()),
    }
}

async fn warm_bootstrap(endpoint: &Endpoint, bootstrap: &[EndpointAddr]) {
    for addr in bootstrap {
        let _ = endpoint.connect(addr.clone(), GOSSIP_ALPN).await;
    }
}

async fn run_serve(target: &str, name: &str, landing: Option<&str>, local: bool) -> Result<()> {
    let (topic, bootstrap) = parse_target(target);
    let endpoint =
        build_endpoint(local, vec![BURN_REMOTE_ALPN.to_vec(), GOSSIP_ALPN.to_vec()]).await?;
    let node = RemoteNode::from_endpoint(endpoint.clone());

    let gossip = Gossip::builder().spawn(endpoint.clone());
    let _router = serve_builder(Device::flex(), node.clone())
        .accept(GOSSIP_ALPN, gossip.clone())
        .spawn();

    warm_bootstrap(&endpoint, &bootstrap).await;

    let advert = PeerAdvert::new(
        RemoteTicket::new(endpoint.addr(), Vec::new()),
        Some(name.to_string()),
        PeerCaps {
            backend: "flex".to_string(),
            device: None,
            devices: 1,
            browser: false,
        },
    );
    let bootstrap_ids = bootstrap.iter().map(|addr| addr.id).collect();
    let config = SwarmConfig::new(topic)
        .bootstrap(bootstrap_ids)
        .advert(advert);
    let swarm = Swarm::join(endpoint.clone(), &gossip, config).await?;

    if bootstrap.is_empty() {
        let ticket = JoinTicket::new(topic, vec![endpoint.addr()]).encode();
        println!("\n[{name}] seed peer serving on flex.");
        println!("JOIN TICKET (give to other peers and the client):\n{ticket}\n");
        if let Some(landing) = landing {
            let link = format!("{}#{ticket}", landing.trim_end_matches('#'));
            println!("SCAN to donate a phone GPU ({link}):\n");
            if let Err(err) = qr2term::print_qr(&link) {
                tracing::warn!("could not render QR: {err}");
            }
            println!();
        }
    } else {
        println!("[{name}] serving on flex, joined the swarm.");
    }

    let mut interval = tokio::time::interval(Duration::from_secs(3));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                tracing::info!(target: "swarm", "[{name}] {} other peer(s) in swarm", swarm.peer_count());
            }
            _ = tokio::signal::ctrl_c() => {
                swarm.leave();
                tokio::time::sleep(Duration::from_millis(300)).await;
                return Ok(());
            }
        }
    }
}

async fn run_client(target: &str, local: bool) -> Result<()> {
    let (topic, bootstrap) = parse_target(target);
    let endpoint = build_endpoint(local, vec![GOSSIP_ALPN.to_vec()]).await?;
    warm_bootstrap(&endpoint, &bootstrap).await;
    let node = RemoteNode::from_endpoint(endpoint.clone());

    let bootstrap_ids = bootstrap.iter().map(|addr| addr.id).collect();
    let (swarm, _router) = Swarm::spawn(
        endpoint.clone(),
        SwarmConfig::new(topic).bootstrap(bootstrap_ids),
    )
    .await?;

    println!("[client] joined swarm, discovering compute peers…");
    let peers = discover_peers(&swarm, Duration::from_secs(15)).await;
    if peers.is_empty() {
        bail!("no compute peers found in the swarm");
    }
    println!("[client] {} compute peer(s):", peers.len());
    for peer in &peers {
        println!(
            "    - {:<10} [{}]",
            peer.advert.name.clone().unwrap_or_default(),
            peer.advert.caps.backend
        );
    }

    // Blocking remote tensor ops are fine on the multi-threaded runtime (iroh I/O uses other workers).
    render_swarm_mandelbrot(&node, &peers)
}

async fn discover_peers(swarm: &Swarm, timeout: Duration) -> Vec<RosterEntry> {
    let start = std::time::Instant::now();
    while swarm.peer_count() == 0 {
        if start.elapsed() > timeout {
            return Vec::new();
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    let mut roster = swarm.roster();
    roster.sort_by_key(|entry| entry.advert.caps.backend != "wgpu"); // GPU peers first
    roster
}

const VIEW: (f32, f32, f32, f32) = (-2.6, 1.0, -1.2, 1.2); // xmin, xmax, ymin, ymax
const WIDTH: usize = 100;
const BAND_H: usize = 5;
const MAX_ITER: usize = 60;

/// Fan a Mandelbrot across the swarm — each band on a different peer — and render it as ASCII,
/// verifying band 0 against a local recompute.
fn render_swarm_mandelbrot(node: &RemoteNode, peers: &[RosterEntry]) -> Result<()> {
    let (xmin, xmax, ymin, ymax) = VIEW;
    let bands = peers.len() * 3;
    let height = bands * BAND_H;

    let devices: Vec<Device> = peers
        .iter()
        .map(|peer| Device::remote_ticket(node, &peer.advert.ticket, 0))
        .collect();

    let mut image = vec![0f32; WIDTH * height];
    let mut who: Vec<&str> = Vec::with_capacity(bands);

    for band in 0..bands {
        let peer = band % peers.len();
        let (y0, y1) = band_bounds(ymin, ymax, band, bands);
        let tile = mandelbrot_tile(&devices[peer], xmin, xmax, y0, y1, WIDTH, BAND_H);
        let offset = band * BAND_H * WIDTH;
        image[offset..offset + tile.len()].copy_from_slice(&tile);
        who.push(peers[peer].advert.name.as_deref().unwrap_or("?"));
    }

    let (y0, y1) = band_bounds(ymin, ymax, 0, bands);
    let local = mandelbrot_tile(&Device::flex(), xmin, xmax, y0, y1, WIDTH, BAND_H);
    let max_diff = local
        .iter()
        .zip(&image[..local.len()])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    println!("\n  Mandelbrot, computed by the swarm:\n");
    let ramp = b" .:-=+*#%@";
    for row in 0..height {
        let mut line = String::with_capacity(WIDTH);
        for col in 0..WIDTH {
            let escaped = image[row * WIDTH + col] / MAX_ITER as f32;
            let idx = (escaped * (ramp.len() - 1) as f32).round() as usize;
            line.push(ramp[idx.min(ramp.len() - 1)] as char);
        }
        println!("  {line}");
    }

    println!("\n  bands (top → bottom), each computed on a peer:");
    for (band, name) in who.iter().enumerate() {
        println!("    band {band:>2}: {name}");
    }
    println!("\n  verify: band 0 recomputed locally — max abs diff vs remote = {max_diff:.3}");
    if max_diff < 1.0 {
        println!("  ✓ the swarm's compute matches a local run");
    } else {
        println!("  ✗ unexpected mismatch");
    }
    Ok(())
}

fn band_bounds(ymin: f32, ymax: f32, band: usize, bands: usize) -> (f32, f32) {
    let span = ymax - ymin;
    let y0 = ymin + span * band as f32 / bands as f32;
    let y1 = ymin + span * (band + 1) as f32 / bands as f32;
    (y0, y1)
}

/// Mandelbrot escape counts for a `w × h` tile, as Burn tensor ops on `device`.
fn mandelbrot_tile(
    device: &Device,
    xmin: f32,
    xmax: f32,
    y0: f32,
    y1: f32,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let step_x = (xmax - xmin) / (w as f32 - 1.0);
    let step_y = (y1 - y0) / (h as f32 - 1.0);

    let xs = Tensor::<1, Int>::arange(0..w as i64, device)
        .float()
        .mul_scalar(step_x)
        .add_scalar(xmin);
    let ys = Tensor::<1, Int>::arange(0..h as i64, device)
        .float()
        .mul_scalar(step_y)
        .add_scalar(y0);
    let cx = xs.reshape([1, w]).expand([h, w]);
    let cy = ys.reshape([h, 1]).expand([h, w]);

    let mut zx = Tensor::<2>::zeros([h, w], device);
    let mut zy = Tensor::<2>::zeros([h, w], device);
    let mut count = Tensor::<2>::zeros([h, w], device);

    for _ in 0..MAX_ITER {
        let zx2 = zx.clone() * zx.clone();
        let zy2 = zy.clone() * zy.clone();
        // inside |z| <= 2 (|z|^2 <= 4): 1.0 while still iterating, 0.0 once escaped
        let inside = (zx2.clone() + zy2.clone()).lower_equal_elem(4.0).float();
        count = count + inside;
        let next_zx = zx2 - zy2 + cx.clone();
        let next_zy = (zx.clone() * zy.clone()).mul_scalar(2.0) + cy.clone();
        zx = next_zx;
        zy = next_zy;
    }

    count.into_data().to_vec::<f32>().expect("f32 tile data")
}
