//! The join ticket carried by a QR code (or any out-of-band channel).
//!
//! It names the gossip topic and one or more bootstrap peers a fresh node dials to enter the swarm,
//! encoded as a compact, URL/QR-safe base32 string.

use core::fmt;

use iroh::{EndpointAddr, EndpointId};
use iroh_gossip::proto::TopicId;
use serde::{Deserialize, Serialize};

const PREFIX: &str = "burnswarm";

/// Derive a gossip [`TopicId`] from a human-readable label, mirroring the topic-string convention
/// used elsewhere in the Burn Remote examples.
pub fn topic_from_label(label: &str) -> TopicId {
    let hash = blake3::hash(format!("burn-swarm:{label}").as_bytes());
    TopicId::from_bytes(*hash.as_bytes())
}

/// Everything needed to enter a swarm: the topic, and bootstrap peers to dial into it.
///
/// Bootstrap entries are full [`EndpointAddr`]s (id + relay + direct paths) so a cold joiner can
/// reach the seed without waiting on global discovery — ideal for a freshly scanned QR code.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JoinTicket {
    topic: [u8; 32],
    bootstrap: Vec<EndpointAddr>,
}

impl JoinTicket {
    /// Build a ticket for `topic`, bootstrapping through `bootstrap`.
    pub fn new(topic: TopicId, bootstrap: Vec<EndpointAddr>) -> Self {
        Self {
            topic: *topic.as_bytes(),
            bootstrap,
        }
    }

    /// Build a ticket whose topic is derived from a human-readable label.
    pub fn from_label(label: &str, bootstrap: Vec<EndpointAddr>) -> Self {
        Self::new(topic_from_label(label), bootstrap)
    }

    /// The gossip topic to subscribe to.
    pub fn topic(&self) -> TopicId {
        TopicId::from_bytes(self.topic)
    }

    /// Full bootstrap addresses (id + paths).
    pub fn bootstrap(&self) -> &[EndpointAddr] {
        &self.bootstrap
    }

    /// Bootstrap peer ids, as [`iroh_gossip`] subscription expects.
    pub fn bootstrap_ids(&self) -> Vec<EndpointId> {
        self.bootstrap.iter().map(|addr| addr.id).collect()
    }

    /// Encode to a compact base32 string suitable for a URL fragment or QR code.
    pub fn encode(&self) -> String {
        let bytes = rmp_serde::to_vec(self).expect("a JoinTicket always serializes");
        format!("{PREFIX}{}", data_encoding::BASE32_NOPAD.encode(&bytes))
    }

    /// Decode a ticket produced by [`encode`](Self::encode).
    pub fn decode(s: &str) -> Result<Self, TicketError> {
        let body = s.strip_prefix(PREFIX).ok_or(TicketError::Prefix)?;
        let bytes = data_encoding::BASE32_NOPAD
            .decode(body.as_bytes())
            .map_err(|_| TicketError::Base32)?;
        rmp_serde::from_slice(&bytes).map_err(|_| TicketError::Decode)
    }
}

/// Failure decoding a [`JoinTicket`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TicketError {
    /// The string did not start with the expected `burnswarm` prefix.
    Prefix,
    /// The body was not valid base32.
    Base32,
    /// The decoded bytes were not a valid ticket.
    Decode,
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self {
            Self::Prefix => "missing 'burnswarm' prefix",
            Self::Base32 => "invalid base32 body",
            Self::Decode => "malformed ticket payload",
        };
        write!(f, "invalid join ticket: {reason}")
    }
}

impl std::error::Error for TicketError {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::message::{PeerAdvert, PeerCaps};
    use burn_remote::RemoteTicket;
    use iroh::SecretKey;

    /// A deterministic advert for tests: `seed` fixes the peer identity so the same id is reproducible.
    pub(crate) fn advert(seed: u8, name: &str) -> PeerAdvert {
        let key = SecretKey::from_bytes(&[seed; 32]);
        let addr = EndpointAddr::from(key.public());
        PeerAdvert::new(
            RemoteTicket::new(addr, Vec::new()),
            Some(name.to_string()),
            PeerCaps::default(),
        )
    }

    #[test]
    fn topic_from_label_is_deterministic() {
        assert_eq!(topic_from_label("burn-web"), topic_from_label("burn-web"));
        assert_ne!(topic_from_label("burn-web"), topic_from_label("other"));
    }

    #[test]
    fn ticket_round_trips_through_base32() {
        let addr = EndpointAddr::from(SecretKey::from_bytes(&[9u8; 32]).public());
        let ticket = JoinTicket::from_label("burn-web", vec![addr]);

        let encoded = ticket.encode();
        assert!(encoded.starts_with(PREFIX));

        let decoded = JoinTicket::decode(&encoded).expect("decodes");
        assert_eq!(ticket, decoded);
        assert_eq!(decoded.topic(), topic_from_label("burn-web"));
        assert_eq!(decoded.bootstrap_ids().len(), 1);
    }

    #[test]
    fn decode_rejects_garbage() {
        assert_eq!(JoinTicket::decode("nope"), Err(TicketError::Prefix));
        assert_eq!(JoinTicket::decode("burnswarm!!!"), Err(TicketError::Base32));
    }
}
