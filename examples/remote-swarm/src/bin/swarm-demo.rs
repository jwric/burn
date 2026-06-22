//! A runnable demo of the gossip swarm primitive.
//!
//! Start a seed, then point any number of peers and watchers at the ticket it prints:
//!
//! ```sh
//! # terminal 1 — the seed prints a JOIN TICKET
//! cargo run -p remote-swarm --bin swarm-demo -- seed burn-web
//!
//! # terminal 2+ — compute peers (paste the ticket from the seed)
//! cargo run -p remote-swarm --bin swarm-demo -- peer burnswarm... "phone-A"
//!
//! # a watcher just prints the live roster (the role a client/scheduler plays)
//! cargo run -p remote-swarm --bin swarm-demo -- watch burnswarm...
//! ```
//!
//! Each node prints its roster every couple of seconds; watch it grow as peers join and shrink when
//! one leaves (Ctrl+C sends a graceful `Bye`).

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

    // `--local` uses iroh's relay-free, discovery-free preset: peers connect over direct
    // (loopback / LAN) paths only, so the demo runs on one host or a LAN with no internet.
    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    let local = raw.iter().any(|arg| arg == "--local");
    raw.retain(|arg| arg != "--local");

    let mut args = raw.into_iter();
    let role = args.next().unwrap_or_default();
    match role.as_str() {
        "seed" => {
            let label = args
                .next()
                .context("usage: swarm-demo [--local] seed <label>")?;
            run_seed(&label, local).await
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

/// Build an Iroh endpoint that advertises the gossip protocol.
///
/// In `local` mode it uses the relay-free `Minimal` preset and skips waiting to come online — direct
/// addresses are known right after bind, which is all same-host/LAN peers need. Otherwise it uses
/// the production N0 preset (relays + discovery) and waits until online so the ticket carries a
/// reachable relay address.
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

/// Build this node's advert from its endpoint. Authorization is empty here (an open pool); a real
/// deployment would place a signed capability in it and validate it with a `PeerAuthorizer`.
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

async fn run_seed(label: &str, local: bool) -> Result<()> {
    let endpoint = build_endpoint(local).await?;
    let topic = topic_from_label(label);
    let advert = local_advert(&endpoint, "seed", caps("flex", false));
    let ticket = JoinTicket::new(topic, vec![endpoint.addr()]);

    println!("\n=== Burn compute swarm — seed ===");
    println!("label : {label}");
    println!("node  : {}", endpoint.id());
    println!(
        "\nJOIN TICKET (share out of band / encode in the QR):\n\n{}\n",
        ticket.encode()
    );

    // The seed has no bootstrap: it is the first node.
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

    // No advert: a watcher is an observer, exactly the role a client or scheduler plays.
    let config = SwarmConfig::new(ticket.topic()).bootstrap(ticket.bootstrap_ids());
    let (swarm, _router) = Swarm::spawn(endpoint, config).await?;
    println!("[watch] observing swarm as {}", swarm.endpoint_id());
    roster_loop(&swarm, "watch").await;
    Ok(())
}

/// Teach the endpoint the bootstrap addresses up front so the gossip join doesn't have to wait on
/// global discovery to resolve them — the QR ticket already carries the full address.
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
                // Give the broadcaster a moment to flush the Bye.
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
