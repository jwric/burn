use core::fmt;

#[cfg(feature = "websocket")]
use burn_communication::Address;
use serde::{Deserialize, Serialize};

/// Stable identity of a Burn Remote peer.
///
/// Network paths are deliberately not part of the identity. An Iroh peer keeps the same
/// identity while moving between direct addresses and relays.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum PeerId {
    /// An Iroh endpoint, authenticated by its public key.
    #[cfg(feature = "iroh")]
    Iroh(iroh::EndpointId),
    /// A legacy WebSocket endpoint.
    #[cfg(feature = "websocket")]
    WebSocket(Address),
}

impl fmt::Display for PeerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(id) => write!(f, "iroh://{id}"),
            #[cfg(feature = "websocket")]
            Self::WebSocket(address) => address.fmt(f),
        }
    }
}

/// A peer identity plus the mutable network paths that may be used to reach it.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum PeerAddr {
    /// An Iroh endpoint address. It may contain direct and relay paths, or only an endpoint id
    /// when the configured Iroh address lookup can resolve it.
    #[cfg(feature = "iroh")]
    Iroh(iroh::EndpointAddr),
    /// A legacy WebSocket address.
    #[cfg(feature = "websocket")]
    WebSocket(Address),
}

/// Serializable connection material issued by an application control plane.
///
/// Burn treats `authorization` as opaque bytes. A fleet platform can place a signed capability,
/// expiry, tenant, or resource policy inside and validate it with a server-side
/// `PeerAuthorizer`.
#[cfg(feature = "iroh")]
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct RemoteTicket {
    peer: iroh::EndpointAddr,
    #[serde(with = "serde_bytes")]
    authorization: Vec<u8>,
}

#[cfg(feature = "iroh")]
impl RemoteTicket {
    /// Create a ticket from an Iroh endpoint address and application credential.
    pub fn new(peer: iroh::EndpointAddr, authorization: impl Into<Vec<u8>>) -> Self {
        Self {
            peer,
            authorization: authorization.into(),
        }
    }

    /// Iroh peer address carried by the ticket.
    pub fn peer(&self) -> &iroh::EndpointAddr {
        &self.peer
    }

    /// Opaque application credential carried by the ticket.
    pub fn authorization(&self) -> &[u8] {
        &self.authorization
    }
}

impl PeerAddr {
    /// Return the stable peer identity, excluding dialing hints.
    pub fn id(&self) -> PeerId {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(address) => PeerId::Iroh(address.id),
            #[cfg(feature = "websocket")]
            Self::WebSocket(address) => PeerId::WebSocket(address.clone()),
        }
    }

    /// Return true when this is an Iroh peer.
    pub fn is_iroh(&self) -> bool {
        match self {
            #[cfg(feature = "iroh")]
            Self::Iroh(_) => true,
            #[cfg(feature = "websocket")]
            Self::WebSocket(_) => false,
        }
    }
}

impl fmt::Display for PeerAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id().fmt(f)
    }
}

#[cfg(feature = "iroh")]
impl From<iroh::EndpointAddr> for PeerAddr {
    fn from(value: iroh::EndpointAddr) -> Self {
        Self::Iroh(value)
    }
}

#[cfg(feature = "iroh")]
impl From<iroh::EndpointId> for PeerAddr {
    fn from(value: iroh::EndpointId) -> Self {
        Self::Iroh(value.into())
    }
}

#[cfg(feature = "websocket")]
impl From<Address> for PeerAddr {
    fn from(value: Address) -> Self {
        Self::WebSocket(value)
    }
}
