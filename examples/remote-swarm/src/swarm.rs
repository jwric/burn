//! The swarm handle: join a gossip topic, announce this node as a compute peer, and keep a live
//! roster of everyone else.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use futures_util::StreamExt;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointId};
use iroh_gossip::api::{Event, GossipSender};
use iroh_gossip::net::{GOSSIP_ALPN, Gossip};
use iroh_gossip::proto::TopicId;
// Cross-target spawning and timers: tokio on native, spawn_local + browser timers in wasm.
use n0_future::task::{AbortOnDropHandle, spawn};
use n0_future::time::sleep;
use tokio::sync::mpsc;
// Portable monotonic clock (std on native, performance.now() in the browser).
use web_time::Instant;

use crate::message::{Load, PeerAdvert, SwarmMessage};
use crate::roster::{Roster, RosterEntry};

/// How a node participates in the swarm.
pub struct SwarmConfig {
    /// Gossip topic to join.
    pub topic: TopicId,
    /// Bootstrap peers to dial into the topic. Empty for the very first node (the seed).
    pub bootstrap: Vec<EndpointId>,
    /// What this node advertises as a compute peer. `None` for observers (clients, schedulers,
    /// dashboards) that only read the roster.
    pub advert: Option<PeerAdvert>,
    /// Heartbeat / prune interval. Default 3s.
    pub heartbeat: Duration,
    /// Forget a peer unheard from for longer than this. Default 10s (~3 missed heartbeats).
    pub peer_ttl: Duration,
    /// Re-broadcast a full `Announce` every Nth heartbeat, as a newcomer safety net. Default 5.
    pub announce_every: u32,
}

impl SwarmConfig {
    /// Default participation in `topic`: no bootstrap, observer-only, 3s heartbeat, 10s TTL.
    pub fn new(topic: TopicId) -> Self {
        Self {
            topic,
            bootstrap: Vec::new(),
            advert: None,
            heartbeat: Duration::from_secs(3),
            peer_ttl: Duration::from_secs(10),
            announce_every: 5,
        }
    }

    /// Set the bootstrap peers used to enter the topic.
    pub fn bootstrap(mut self, peers: Vec<EndpointId>) -> Self {
        self.bootstrap = peers;
        self
    }

    /// Advertise this node as a compute peer with the given record.
    pub fn advert(mut self, advert: PeerAdvert) -> Self {
        self.advert = Some(advert);
        self
    }
}

struct Inner {
    self_id: EndpointId,
    out: mpsc::UnboundedSender<SwarmMessage>,
    roster: Mutex<Roster>,
    advert: Mutex<Option<PeerAdvert>>,
    load: Mutex<Load>,
    // Background loops; dropping these handles aborts the tasks, so the swarm stops once the last
    // `Swarm` clone is gone.
    tasks: Mutex<Vec<AbortOnDropHandle<()>>>,
}

/// A live handle to the swarm. Clone it freely; dropping the last clone stops the background tasks.
#[derive(Clone)]
pub struct Swarm {
    inner: Arc<Inner>,
}

impl Swarm {
    /// Build a gossip instance and a [`Router`] on `endpoint`, then join.
    ///
    /// Use this for gossip-only nodes (clients, schedulers) or peers that don't also serve Burn
    /// Remote on the same endpoint. The endpoint must advertise [`GOSSIP_ALPN`]. Keep the returned
    /// [`Router`] alive for as long as you want to stay in the swarm.
    pub async fn spawn(endpoint: Endpoint, config: SwarmConfig) -> Result<(Self, Router)> {
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let router = Router::builder(endpoint.clone())
            .accept(GOSSIP_ALPN, gossip.clone())
            .spawn();
        let swarm = Self::join(endpoint, &gossip, config).await?;
        Ok((swarm, router))
    }

    /// Join using an already-spawned [`Gossip`]. Use this when you compose your own router — e.g. a
    /// compute peer that also accepts [`burn_remote::BURN_REMOTE_ALPN`] on the same endpoint:
    ///
    /// ```ignore
    /// let gossip = Gossip::builder().spawn(endpoint.clone());
    /// let router = Router::builder(endpoint.clone())
    ///     .accept(BURN_REMOTE_ALPN, burn_handler)
    ///     .accept(GOSSIP_ALPN, gossip.clone())
    ///     .spawn();
    /// let swarm = Swarm::join(endpoint, &gossip, config).await?;
    /// ```
    pub async fn join(endpoint: Endpoint, gossip: &Gossip, config: SwarmConfig) -> Result<Self> {
        let SwarmConfig {
            topic,
            bootstrap,
            advert,
            heartbeat,
            peer_ttl,
            announce_every,
        } = config;
        let announce_every = announce_every.max(1);
        let self_id = endpoint.id();

        // The seed (no bootstrap) subscribes without waiting; joiners wait until actually connected.
        let gossip_topic = if bootstrap.is_empty() {
            gossip.subscribe(topic, Vec::new()).await?
        } else {
            gossip.subscribe_and_join(topic, bootstrap).await?
        };
        let (sender, receiver) = gossip_topic.split();

        // A single broadcaster task owns the gossip sender, so the handle stays Send + Sync and we
        // never need GossipSender to be shareable. Everyone else queues outgoing messages here.
        let (out, out_rx) = mpsc::unbounded_channel::<SwarmMessage>();
        if let Some(advert) = &advert {
            let _ = out.send(SwarmMessage::Announce(advert.clone()));
        }

        let inner = Arc::new(Inner {
            self_id,
            out,
            roster: Mutex::new(Roster::new(peer_ttl)),
            advert: Mutex::new(advert),
            load: Mutex::new(Load::default()),
            tasks: Mutex::new(Vec::new()),
        });

        let broadcaster = AbortOnDropHandle::new(spawn(broadcast_loop(sender, out_rx)));
        let events = AbortOnDropHandle::new(spawn(event_loop(inner.clone(), receiver)));
        let heartbeats = AbortOnDropHandle::new(spawn(heartbeat_loop(
            inner.clone(),
            heartbeat,
            announce_every,
        )));
        inner
            .tasks
            .lock()
            .unwrap()
            .extend([broadcaster, events, heartbeats]);

        Ok(Self { inner })
    }

    /// This node's endpoint id.
    pub fn endpoint_id(&self) -> EndpointId {
        self.inner.self_id
    }

    /// Snapshot of all live peers (excluding this node).
    pub fn roster(&self) -> Vec<RosterEntry> {
        self.inner.roster.lock().unwrap().snapshot()
    }

    /// Number of live peers (excluding this node).
    pub fn peer_count(&self) -> usize {
        self.inner.roster.lock().unwrap().len()
    }

    /// The least-loaded peer, for simple scheduling.
    pub fn pick_least_loaded(&self) -> Option<RosterEntry> {
        self.inner.roster.lock().unwrap().pick_least_loaded()
    }

    /// Update the load this node reports in its heartbeats.
    pub fn report_load(&self, load: Load) {
        *self.inner.load.lock().unwrap() = load;
    }

    /// Replace this node's advert and broadcast it immediately.
    pub fn announce(&self, advert: PeerAdvert) {
        *self.inner.advert.lock().unwrap() = Some(advert.clone());
        let _ = self.inner.out.send(SwarmMessage::Announce(advert));
    }

    /// Leave the swarm gracefully: broadcast `Bye` so peers drop us without waiting for the TTL.
    /// Delivery is best-effort; give the broadcaster a moment before exiting the process.
    pub fn leave(&self) {
        let _ = self.inner.out.send(SwarmMessage::Bye {
            peer: self.inner.self_id,
        });
    }
}

async fn broadcast_loop(sender: GossipSender, mut out_rx: mpsc::UnboundedReceiver<SwarmMessage>) {
    while let Some(msg) = out_rx.recv().await {
        if let Err(err) = sender.broadcast(msg.encode()).await {
            tracing::debug!(?err, "gossip broadcast failed");
        }
    }
}

async fn event_loop(
    inner: Arc<Inner>,
    receiver: impl futures_util::Stream<Item = Result<Event, iroh_gossip::api::ApiError>>,
) {
    let mut receiver = std::pin::pin!(receiver);
    // `None` means our subscription ended (we're shutting down). A transient `Err` can surface when
    // a neighbour drops during churn — skip it and keep maintaining the roster; breaking here would
    // freeze the roster and let the TTL silently prune everyone.
    while let Some(item) = receiver.next().await {
        let event = match item {
            Ok(event) => event,
            Err(err) => {
                tracing::debug!(?err, "gossip event stream error");
                continue;
            }
        };
        match event {
            Event::Received(msg) => {
                let Ok(message) = SwarmMessage::decode(&msg.content) else {
                    continue;
                };
                let now = Instant::now();
                let mut roster = inner.roster.lock().unwrap();
                match message {
                    // Ignore our own announcements echoed back by the mesh.
                    SwarmMessage::Announce(advert) if advert.endpoint_id() == inner.self_id => {}
                    SwarmMessage::Announce(advert) => roster.observe_announce(advert, now),
                    SwarmMessage::Heartbeat { peer, load } if peer != inner.self_id => {
                        roster.observe_heartbeat(&peer, load, now);
                    }
                    SwarmMessage::Heartbeat { .. } => {}
                    SwarmMessage::Bye { peer } => roster.remove(&peer),
                }
            }
            // A new neighbour joined: re-announce so they learn the full roster right away.
            Event::NeighborUp(_) => {
                if let Some(advert) = inner.advert.lock().unwrap().clone() {
                    let _ = inner.out.send(SwarmMessage::Announce(advert));
                }
            }
            Event::NeighborDown(_) | Event::Lagged => {}
        }
    }
}

async fn heartbeat_loop(inner: Arc<Inner>, period: Duration, announce_every: u32) {
    let mut ticks: u32 = 0;
    loop {
        sleep(period).await;
        ticks = ticks.wrapping_add(1);

        let evicted = inner.roster.lock().unwrap().prune(Instant::now());
        if !evicted.is_empty() {
            tracing::debug!(count = evicted.len(), "pruned stale peers");
        }

        // Observers (no advert) stay silent — they only listen.
        let Some(advert) = inner.advert.lock().unwrap().clone() else {
            continue;
        };
        let msg = if ticks.is_multiple_of(announce_every) {
            SwarmMessage::Announce(advert)
        } else {
            SwarmMessage::Heartbeat {
                peer: inner.self_id,
                load: *inner.load.lock().unwrap(),
            }
        };
        let _ = inner.out.send(msg);
    }
}
