//! Gossip-based peer discovery for a Burn Remote compute swarm.
//!
//! `burn-remote` already gives you the **data plane**: a process binds an Iroh [`Endpoint`], a
//! client dials a compute peer with a [`RemoteTicket`], and tensor operations stream over a direct
//! QUIC connection. What it deliberately leaves to the application is *discovery* — "who is out
//! there to dial?".
//!
//! This crate is that missing **control plane**, built on [`iroh_gossip`]. Nodes join a shared
//! gossip topic and flood three tiny messages — [`SwarmMessage::Announce`],
//! [`SwarmMessage::Heartbeat`], and [`SwarmMessage::Bye`] — to maintain a live [`Roster`] of
//! reachable compute peers. A client then picks a peer from the roster and dials it over the
//! ordinary Burn Remote connection.
//!
//! Gossip is used **only** for coordination. Announcements are small and flooded to every
//! subscriber; tensors never travel through gossip — they go over the direct connection the
//! announced [`RemoteTicket`] points at.
//!
//! ```text
//! QR / out-of-band  ──▶  JoinTicket { topic, bootstrap }
//!                              │
//!         join gossip topic ──┤
//!                              ├─ broadcast Announce{ ticket, caps }   (control plane: gossip)
//!                              └─ maintain Roster from peers' Announce/Heartbeat/Bye
//!                              │
//!  pick a RosterEntry ──▶ RemoteNode::device_from_ticket(entry.advert.ticket)
//!                              └─ tensors stream here                  (data plane: direct QUIC)
//! ```
//!
//! [`Endpoint`]: iroh::Endpoint
//! [`RemoteTicket`]: burn_remote::RemoteTicket

mod message;
mod roster;
mod swarm;
mod ticket;

pub use message::{Load, PeerAdvert, PeerCaps, SwarmMessage};
pub use roster::{Roster, RosterEntry};
pub use swarm::{Swarm, SwarmConfig};
pub use ticket::{JoinTicket, TicketError, topic_from_label};

/// Re-exported so a compute peer composing its own router can accept the gossip protocol alongside
/// [`burn_remote::BURN_REMOTE_ALPN`] on one endpoint.
pub use iroh_gossip::net::GOSSIP_ALPN;
/// Re-exported so callers can name the gossip topic without an explicit `iroh-gossip` dependency.
pub use iroh_gossip::proto::TopicId;
