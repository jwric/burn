//! The live membership table a swarm node maintains from gossip.

use std::collections::HashMap;
use std::time::Duration;

use iroh::EndpointId;
use web_time::Instant;

use crate::message::{Load, PeerAdvert};

/// A compute peer currently known to this node.
#[derive(Clone, Debug)]
pub struct RosterEntry {
    pub advert: PeerAdvert,
    pub load: Load,
    pub last_seen: Instant,
}

/// A set of live compute peers, aged out by a time-to-live.
#[derive(Debug)]
pub struct Roster {
    peers: HashMap<EndpointId, RosterEntry>,
    ttl: Duration,
}

impl Roster {
    pub fn new(ttl: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            ttl,
        }
    }

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

    /// Returns `false` if the peer is unknown (no advert seen yet).
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

    pub fn remove(&mut self, peer: &EndpointId) {
        self.peers.remove(peer);
    }

    /// Forget peers unheard from for longer than the TTL; returns the evicted ids.
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

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn contains(&self, peer: &EndpointId) -> bool {
        self.peers.contains_key(peer)
    }

    pub fn snapshot(&self) -> Vec<RosterEntry> {
        self.peers.values().cloned().collect()
    }

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
        assert!(!roster.observe_heartbeat(&id, load, now));

        roster.observe_announce(a, now);
        assert!(roster.observe_heartbeat(&id, load, now));
        assert_eq!(roster.snapshot()[0].load.sessions, 7);
    }

    #[test]
    fn prune_evicts_only_expired_entries() {
        let ttl = Duration::from_secs(10);
        let mut roster = Roster::new(ttl);
        let start = Instant::now();

        roster.observe_announce(advert(1, "old"), start);
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
