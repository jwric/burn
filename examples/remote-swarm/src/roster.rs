//! The live membership table a swarm node maintains by listening to the gossip topic.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use iroh::EndpointId;

use crate::message::{Load, PeerAdvert};

/// A compute peer currently known to this node.
#[derive(Clone, Debug)]
pub struct RosterEntry {
    /// The peer's advertised connection material and capabilities.
    pub advert: PeerAdvert,
    /// Most recent load report.
    pub load: Load,
    /// When we last heard from this peer (any message).
    pub last_seen: Instant,
}

/// A set of live compute peers, aged out by a time-to-live.
///
/// The roster is pure bookkeeping: feed it the messages and neighbour events the gossip task
/// observes, prune it on a timer, and read snapshots for scheduling. It does no I/O, so it is
/// trivially testable.
#[derive(Debug)]
pub struct Roster {
    peers: HashMap<EndpointId, RosterEntry>,
    ttl: Duration,
}

impl Roster {
    /// Create an empty roster that forgets peers unheard from for longer than `ttl`.
    pub fn new(ttl: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            ttl,
        }
    }

    /// Record an `Announce`: insert a new peer or refresh an existing one's advert.
    pub fn observe_announce(&mut self, advert: PeerAdvert, now: Instant) {
        let id = advert.endpoint_id();
        let entry = self.peers.entry(id).or_insert_with(|| RosterEntry {
            advert: advert.clone(),
            load: Load::default(),
            last_seen: now,
        });
        entry.advert = advert;
        entry.last_seen = now;
    }

    /// Record a `Heartbeat`. Returns `false` if the peer is unknown (no advert seen yet), so the
    /// caller can decide to request a re-announce.
    pub fn observe_heartbeat(&mut self, peer: &EndpointId, load: Load, now: Instant) -> bool {
        match self.peers.get_mut(peer) {
            Some(entry) => {
                entry.load = load;
                entry.last_seen = now;
                true
            }
            None => false,
        }
    }

    /// Drop a peer immediately (a `Bye`, or an explicit eviction).
    pub fn remove(&mut self, peer: &EndpointId) {
        self.peers.remove(peer);
    }

    /// Forget every peer unheard from for longer than the TTL. Returns the evicted ids.
    pub fn prune(&mut self, now: Instant) -> Vec<EndpointId> {
        let ttl = self.ttl;
        let expired: Vec<EndpointId> = self
            .peers
            .iter()
            .filter(|(_, e)| now.saturating_duration_since(e.last_seen) > ttl)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            self.peers.remove(id);
        }
        expired
    }

    /// Number of live peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// True when no peers are known.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// True if `peer` is currently in the roster.
    pub fn contains(&self, peer: &EndpointId) -> bool {
        self.peers.contains_key(peer)
    }

    /// Snapshot of every live peer, for dashboards or scheduling.
    pub fn snapshot(&self) -> Vec<RosterEntry> {
        self.peers.values().cloned().collect()
    }

    /// Pick the least-loaded peer (fewest sessions, then lowest ops/sec). `None` if empty.
    ///
    /// A minimal scheduling primitive — for a true swarm, clients can instead map work to peers by
    /// rendezvous-hashing over [`snapshot`](Self::snapshot) so assignment needs no coordinator.
    pub fn pick_least_loaded(&self) -> Option<RosterEntry> {
        self.peers
            .values()
            .min_by(|a, b| {
                a.load
                    .sessions
                    .cmp(&b.load.sessions)
                    .then(a.load.ops_per_sec.total_cmp(&b.load.ops_per_sec))
            })
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ticket::tests::advert;

    #[test]
    fn announce_inserts_and_refreshes() {
        let mut roster = Roster::new(Duration::from_secs(10));
        let now = Instant::now();

        roster.observe_announce(advert(1, "a"), now);
        roster.observe_announce(advert(2, "b"), now);
        assert_eq!(roster.len(), 2);

        // Re-announcing the same peer refreshes rather than duplicates.
        roster.observe_announce(advert(1, "a-renamed"), now);
        assert_eq!(roster.len(), 2);
        let entry = roster
            .snapshot()
            .into_iter()
            .find(|e| e.advert.endpoint_id() == advert(1, "a").endpoint_id())
            .unwrap();
        assert_eq!(entry.advert.name.as_deref(), Some("a-renamed"));
    }

    #[test]
    fn heartbeat_updates_load_only_for_known_peers() {
        let mut roster = Roster::new(Duration::from_secs(10));
        let now = Instant::now();
        let a = advert(1, "a");
        let id = a.endpoint_id();

        let load = Load {
            sessions: 7,
            ops_per_sec: 1.0,
        };
        assert!(!roster.observe_heartbeat(&id, load, now)); // unknown -> false

        roster.observe_announce(a, now);
        assert!(roster.observe_heartbeat(&id, load, now)); // known -> true
        assert_eq!(roster.snapshot()[0].load.sessions, 7);
    }

    #[test]
    fn prune_evicts_only_expired_entries() {
        let ttl = Duration::from_secs(10);
        let mut roster = Roster::new(ttl);
        let start = Instant::now();

        roster.observe_announce(advert(1, "old"), start);
        // A fresh heartbeat keeps the second peer alive past the first's expiry.
        let later = start + Duration::from_secs(8);
        roster.observe_announce(advert(2, "fresh"), later);

        let prune_at = start + Duration::from_secs(11);
        let evicted = roster.prune(prune_at);

        assert_eq!(evicted, vec![advert(1, "old").endpoint_id()]);
        assert_eq!(roster.len(), 1);
        assert!(roster.contains(&advert(2, "fresh").endpoint_id()));
    }

    #[test]
    fn bye_drops_immediately() {
        let mut roster = Roster::new(Duration::from_secs(10));
        let now = Instant::now();
        let a = advert(1, "a");
        let id = a.endpoint_id();
        roster.observe_announce(a, now);
        assert_eq!(roster.len(), 1);
        roster.remove(&id);
        assert!(roster.is_empty());
    }

    #[test]
    fn pick_least_loaded_prefers_fewer_sessions() {
        let mut roster = Roster::new(Duration::from_secs(10));
        let now = Instant::now();
        roster.observe_announce(advert(1, "busy"), now);
        roster.observe_announce(advert(2, "idle"), now);
        roster.observe_heartbeat(
            &advert(1, "busy").endpoint_id(),
            Load {
                sessions: 5,
                ops_per_sec: 0.0,
            },
            now,
        );
        roster.observe_heartbeat(
            &advert(2, "idle").endpoint_id(),
            Load {
                sessions: 1,
                ops_per_sec: 0.0,
            },
            now,
        );

        let pick = roster.pick_least_loaded().unwrap();
        assert_eq!(pick.advert.name.as_deref(), Some("idle"));
    }
}
