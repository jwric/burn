//! Gossip-based peer discovery for a Burn Remote compute swarm.
//!
//! `burn-remote` is the data plane (dial a peer, stream tensors); this crate is the control plane:
//! nodes join a gossip topic and exchange [`SwarmMessage`]s to maintain a live [`Roster`] of compute
//! peers. Gossip carries coordination only — tensors stay on the direct connection a peer's
//! [`RemoteTicket`] points at.

mod message;
mod roster;
mod swarm;
mod ticket;

pub use message::{Load, PeerAdvert, PeerCaps, SwarmMessage};
pub use roster::{Roster, RosterEntry};
pub use swarm::{Swarm, SwarmConfig};
pub use ticket::{JoinTicket, TicketError, topic_from_label};

pub use burn_remote::RemoteTicket;
pub use iroh_gossip::net::{GOSSIP_ALPN, Gossip};
pub use iroh_gossip::proto::TopicId;
