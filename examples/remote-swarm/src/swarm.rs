//! The swarm handle: join a gossip topic, advertise this node, and keep a live roster.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use futures_util::StreamExt;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointId};
use iroh_gossip::api::{Event, GossipSender};
use iroh_gossip::net::{GOSSIP_ALPN, Gossip};
use iroh_gossip::proto::TopicId;
use n0_future::task::{AbortOnDropHandle, spawn};
use n0_future::time::sleep;
use tokio::sync::mpsc;
use web_time::Instant;

use crate::message::{Load, PeerAdvert, SwarmMessage};
use crate::roster::{Roster, RosterEntry};

/// How a node participates in the swarm.
pub struct SwarmConfig {
    pub topic: TopicId,
    pub bootstrap: Vec<EndpointId>,
    pub advert: Option<PeerAdvert>,
    pub heartbeat: Duration,
    pub peer_ttl: Duration,
    /// Re-broadcast a full `Announce` every Nth heartbeat, as a newcomer safety net.
    pub announce_every: u32,
}

impl SwarmConfig {
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

    pub fn bootstrap(mut self, peers: Vec<EndpointId>) -> Self {
        self.bootstrap = peers;
        self
    }

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
    tasks: Mutex<Vec<AbortOnDropHandle<()>>>,
}

/// A live handle to the swarm; dropping the last clone stops its background tasks.
#[derive(Clone)]
pub struct Swarm {
    inner: Arc<Inner>,
}

impl Swarm {
    /// Build a gossip instance and [`Router`] on `endpoint`, then join. For gossip-only nodes, or
    /// peers that don't also serve Burn Remote on this endpoint.
    pub async fn spawn(endpoint: Endpoint, config: SwarmConfig) -> Result<(Self, Router)> {
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let router = Router::builder(endpoint.clone())
            .accept(GOSSIP_ALPN, gossip.clone())
            .spawn();
        let swarm = Self::join(endpoint, &gossip, config).await?;
        Ok((swarm, router))
    }

    /// Join using an already-spawned [`Gossip`], for callers composing their own router (e.g. serving
    /// [`burn_remote::BURN_REMOTE_ALPN`] and [`GOSSIP_ALPN`] on one endpoint).
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

        let gossip_topic = if bootstrap.is_empty() {
            gossip.subscribe(topic, Vec::new()).await?
        } else {
            gossip.subscribe_and_join(topic, bootstrap).await?
        };
        let (sender, receiver) = gossip_topic.split();

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

    pub fn endpoint_id(&self) -> EndpointId {
        self.inner.self_id
    }

    pub fn roster(&self) -> Vec<RosterEntry> {
        self.inner.roster.lock().unwrap().snapshot()
    }

    pub fn peer_count(&self) -> usize {
        self.inner.roster.lock().unwrap().len()
    }

    pub fn pick_least_loaded(&self) -> Option<RosterEntry> {
        self.inner.roster.lock().unwrap().pick_least_loaded()
    }

    pub fn report_load(&self, load: Load) {
        *self.inner.load.lock().unwrap() = load;
    }

    pub fn announce(&self, advert: PeerAdvert) {
        *self.inner.advert.lock().unwrap() = Some(advert.clone());
        let _ = self.inner.out.send(SwarmMessage::Announce(advert));
    }

    /// Broadcast `Bye` so peers drop us without waiting for the TTL. Best-effort; give the
    /// broadcaster a moment before exiting the process.
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
    while let Some(item) = receiver.next().await {
        // Skip transient stream errors; only `None` (subscription ended) stops roster upkeep.
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
                    SwarmMessage::Announce(advert) if advert.endpoint_id() == inner.self_id => {}
                    SwarmMessage::Announce(advert) => roster.observe_announce(advert, now),
                    SwarmMessage::Heartbeat { peer, load } if peer != inner.self_id => {
                        roster.observe_heartbeat(&peer, load, now);
                    }
                    SwarmMessage::Heartbeat { .. } => {}
                    SwarmMessage::Bye { peer } => roster.remove(&peer),
                }
            }
            // Re-announce when a neighbour joins so it learns the roster promptly.
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
