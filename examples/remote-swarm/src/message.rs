//! Wire types flooded over the swarm's gossip topic.
//!
//! Control plane only: every message is broadcast to all subscribers, so keep them small. Tensors
//! and compute traffic travel over the dialed Burn Remote connection, never here.

use burn_remote::RemoteTicket;
use bytes::Bytes;
use iroh::EndpointId;
use serde::{Deserialize, Serialize};

/// Static capabilities a compute peer advertises in its [`PeerAdvert`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PeerCaps {
    /// Backend label, e.g. `"wgpu"`, `"cuda"`, `"flex"`.
    pub backend: String,
    /// GPU / adapter name when known, e.g. `"Apple M2"`.
    pub device: Option<String>,
    /// Number of compute devices this peer hosts on its endpoint.
    pub devices: u32,
    /// True when the peer runs in a browser tab (relay-only, suspends in the background).
    pub browser: bool,
}

/// A compute peer's changing load, refreshed with every heartbeat so a scheduler can balance work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Load {
    /// Clients currently connected to the peer.
    pub sessions: u32,
    /// Smoothed operations per second the peer is sustaining (`0.0` when unknown).
    pub ops_per_sec: f32,
}

/// Everything a client needs to discover and dial a compute peer.
///
/// The [`RemoteTicket`] is exactly what [`burn_remote::RemoteNode::device_from_ticket`] consumes,
/// so turning a roster entry into a usable device is a one-liner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeerAdvert {
    /// Connection material: the peer's Iroh address plus opaque authorization bytes.
    pub ticket: RemoteTicket,
    /// Friendly label for dashboards, e.g. `"seat 12 · Pixel 8"`.
    pub name: Option<String>,
    /// Static capabilities.
    pub caps: PeerCaps,
}

impl PeerAdvert {
    /// Build an advert from connection material and capabilities.
    pub fn new(ticket: RemoteTicket, name: Option<String>, caps: PeerCaps) -> Self {
        Self { ticket, name, caps }
    }

    /// Stable identity of the advertised peer (its Iroh endpoint id).
    pub fn endpoint_id(&self) -> EndpointId {
        self.ticket.peer().id
    }
}

/// A message broadcast to the whole swarm over the gossip topic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SwarmMessage {
    /// "I'm here, reach me like this, here's what I can do." Sent when a peer joins, whenever a new
    /// neighbour appears (so late joiners learn the full roster), and periodically as a fallback.
    Announce(PeerAdvert),
    /// Lightweight liveness plus current load — cheaper than re-announcing the whole record.
    Heartbeat { peer: EndpointId, load: Load },
    /// Graceful departure, so peers drop us immediately instead of waiting out the TTL.
    Bye { peer: EndpointId },
}

impl SwarmMessage {
    /// Encode for [`broadcast`](iroh_gossip::api::GossipSender::broadcast). Uses msgpack, matching
    /// the rest of the Burn Remote wire format.
    pub fn encode(&self) -> Bytes {
        rmp_serde::to_vec(self)
            .expect("a SwarmMessage always serializes")
            .into()
    }

    /// Decode a message received from the topic.
    pub fn decode(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ticket::tests::advert;

    #[test]
    fn round_trips_every_variant() {
        let a = advert(1, "alpha");
        let id = a.endpoint_id();

        let cases = [
            SwarmMessage::Announce(a),
            SwarmMessage::Heartbeat {
                peer: id,
                load: Load {
                    sessions: 3,
                    ops_per_sec: 42.5,
                },
            },
            SwarmMessage::Bye { peer: id },
        ];

        for msg in cases {
            let decoded = SwarmMessage::decode(&msg.encode()).expect("decodes");
            assert_eq!(msg, decoded);
        }
    }
}
