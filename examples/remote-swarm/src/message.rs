//! Wire types broadcast over the swarm's gossip topic — control plane only, kept small.

use burn_remote::RemoteTicket;
use bytes::Bytes;
use iroh::EndpointId;
use serde::{Deserialize, Serialize};

/// Static capabilities a compute peer advertises.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PeerCaps {
    pub backend: String,
    pub device: Option<String>,
    pub devices: u32,
    pub browser: bool,
}

/// A compute peer's changing load, refreshed with every heartbeat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Load {
    pub sessions: u32,
    pub ops_per_sec: f32,
}

/// Everything a client needs to discover and dial a compute peer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeerAdvert {
    pub ticket: RemoteTicket,
    pub name: Option<String>,
    pub caps: PeerCaps,
}

impl PeerAdvert {
    pub fn new(ticket: RemoteTicket, name: Option<String>, caps: PeerCaps) -> Self {
        Self { ticket, name, caps }
    }

    pub fn endpoint_id(&self) -> EndpointId {
        self.ticket.peer().id
    }
}

/// A message broadcast to the whole swarm over the gossip topic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SwarmMessage {
    Announce(PeerAdvert),
    Heartbeat { peer: EndpointId, load: Load },
    Bye { peer: EndpointId },
}

impl SwarmMessage {
    pub fn encode(&self) -> Bytes {
        rmp_serde::to_vec(self)
            .expect("a SwarmMessage always serializes")
            .into()
    }

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
