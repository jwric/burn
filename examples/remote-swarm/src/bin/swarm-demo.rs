//! A runnable demo of the gossip swarm primitive.
//!
//! ```sh
//! cargo run -p remote-swarm --bin swarm-demo -- seed burn-web
//! cargo run -p remote-swarm --bin swarm-demo -- peer burnswarm... "phone-A"
//! cargo run -p remote-swarm --bin swarm-demo -- watch burnswarm...
//! ```
//!
//! Add `--local` (before the role) for a relay-free same-host/LAN run.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use burn_remote::RemoteTicket;
use iroh::endpoint::presets;
use iroh::{Endpoint, EndpointAddr};
use remote_swarm::{
    GOSSIP_ALPN, JoinTicket, PeerAdvert, PeerCaps, Swarm, SwarmConfig, topic_from_label,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,remote_swarm=info".into()),
        )
        .init();

    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let local = raw.iter().any(|arg| arg == "--local");
    raw.retain(|arg| arg != "--local");

    let mut args = raw.into_iter();
    let role = args.next().unwrap_or_default();
    match role.as_str() {
        "seed" => {
            let label = args
                .next()
                .context("usage: swarm-demo [--local] seed <label> [landing-url]")?;
            let landing = args.next();
            run_seed(&label, landing, local).await
        }
        "peer" => {
            let ticket = args
                .next()
                .context("usage: swarm-demo [--local] peer <ticket> [name]")?;
            let name = args.next().unwrap_or_else(|| "peer".to_string());
            run_peer(&ticket, &name, local).await
        }
        "watch" => {
            let ticket = args
                .next()
                .context("usage: swarm-demo [--local] watch <ticket>")?;
            run_watch(&ticket, local).await
        }
        other => {
            bail!(
                "unknown role {other:?}\n\nusage (add --local for a relay-free same-host/LAN run):\n  swarm-demo [--local] seed <label>\n  swarm-demo [--local] peer <ticket> [name]\n  swarm-demo [--local] watch <ticket>"
            );
        }
    }
}

/// `--local` uses iroh's relay-free `Minimal` preset (direct paths only); otherwise N0 + relays.
async fn build_endpoint(local: bool) -> Result<Endpoint> {
    let alpns = vec![GOSSIP_ALPN.to_vec()];
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

fn local_advert(endpoint: &Endpoint, name: &str, caps: PeerCaps) -> PeerAdvert {
    let ticket = RemoteTicket::new(endpoint.addr(), Vec::new());
    PeerAdvert::new(ticket, Some(name.to_string()), caps)
}

fn caps(backend: &str, browser: bool) -> PeerCaps {
    PeerCaps {
        backend: backend.to_string(),
        device: None,
        devices: 1,
        browser,
    }
}

async fn run_seed(label: &str, landing: Option<String>, local: bool) -> Result<()> {
    let endpoint = build_endpoint(local).await?;
    let topic = topic_from_label(label);
    let advert = local_advert(&endpoint, "seed", caps("flex", false));
    let ticket = JoinTicket::new(topic, vec![endpoint.addr()]).encode();

    let link = match &landing {
        Some(base) => format!("{}#{ticket}", base.trim_end_matches('#')),
        None => ticket.clone(),
    };

    println!("\n=== Burn compute swarm — seed ===");
    println!("label : {label}");
    println!("node  : {}", endpoint.id());
    println!("\nJOIN TICKET:\n{ticket}\n");
    if landing.is_some() {
        println!("LAUNCH LINK (scan to open a browser compute peer):\n{link}\n");
    }
    if let Err(err) = qr2term::print_qr(link.as_str()) {
        tracing::debug!(?err, "could not render QR code");
    }

    let config = SwarmConfig::new(topic).advert(advert);
    let (swarm, _router) = Swarm::spawn(endpoint, config).await?;
    roster_loop(&swarm, "seed").await;
    Ok(())
}

async fn run_peer(ticket_str: &str, name: &str, local: bool) -> Result<()> {
    let ticket = JoinTicket::decode(ticket_str)?;
    let endpoint = build_endpoint(local).await?;
    warm_bootstrap(&endpoint, ticket.bootstrap()).await;

    let advert = local_advert(&endpoint, name, caps("wgpu", false));
    let config = SwarmConfig::new(ticket.topic())
        .bootstrap(ticket.bootstrap_ids())
        .advert(advert);
    let (swarm, _router) = Swarm::spawn(endpoint, config).await?;
    println!("[{name}] joined swarm as {}", swarm.endpoint_id());
    roster_loop(&swarm, name).await;
    Ok(())
}

async fn run_watch(ticket_str: &str, local: bool) -> Result<()> {
    let ticket = JoinTicket::decode(ticket_str)?;
    let endpoint = build_endpoint(local).await?;
    warm_bootstrap(&endpoint, ticket.bootstrap()).await;

    let config = SwarmConfig::new(ticket.topic()).bootstrap(ticket.bootstrap_ids());
    let (swarm, _router) = Swarm::spawn(endpoint, config).await?;
    println!("[watch] observing swarm as {}", swarm.endpoint_id());
    roster_loop(&swarm, "watch").await;
    Ok(())
}

async fn warm_bootstrap(endpoint: &Endpoint, bootstrap: &[EndpointAddr]) {
    for addr in bootstrap {
        if let Err(err) = endpoint.connect(addr.clone(), GOSSIP_ALPN).await {
            tracing::debug!(?err, peer = %addr.id, "bootstrap warm-up failed");
        }
    }
}

async fn roster_loop(swarm: &Swarm, who: &str) {
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    loop {
        tokio::select! {
            _ = interval.tick() => print_roster(swarm, who),
            _ = tokio::signal::ctrl_c() => {
                println!("[{who}] leaving swarm…");
                swarm.leave();
                // Let the broadcaster flush the Bye before we exit.
                tokio::time::sleep(Duration::from_millis(300)).await;
                return;
            }
        }
    }
}

fn print_roster(swarm: &Swarm, who: &str) {
    let roster = swarm.roster();
    println!("[{who}] {} peer(s) in swarm:", roster.len());
    for entry in &roster {
        let id = entry.advert.endpoint_id().to_string();
        let short = &id[..id.len().min(8)];
        println!(
            "    - {name:<12} {short}  {backend:<5} sessions={sessions} ops/s={ops:.0}",
            name = entry.advert.name.clone().unwrap_or_default(),
            backend = entry.advert.caps.backend,
            sessions = entry.load.sessions,
            ops = entry.load.ops_per_sec,
        );
    }
}
