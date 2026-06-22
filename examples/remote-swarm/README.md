# remote-swarm

Gossip-based **peer discovery** for a Burn Remote compute swarm — the control plane that turns the
existing point-to-point remote backend into a scan-to-join GPU pool.

`burn-remote` already provides the **data plane**: bind an Iroh endpoint, hand a client a
`RemoteTicket`, and tensor operations stream over a direct QUIC connection. What it deliberately
leaves to the application is *discovery*: "who is out there to dial?". This crate fills that gap with
[`iroh-gossip`](https://crates.io/crates/iroh-gossip).

## How it works

Nodes join a shared gossip topic and flood three small messages to maintain a live **roster** of
reachable compute peers:

| message     | when                                              | carries                          |
| ----------- | ------------------------------------------------- | -------------------------------- |
| `Announce`  | on join, on a new neighbour, periodically         | `RemoteTicket` + caps (the full record) |
| `Heartbeat` | every interval                                    | peer id + current load           |
| `Bye`       | on graceful leave                                 | peer id                          |

A client (or scheduler) reads the roster, picks a peer, and dials it with the **ordinary** Burn
Remote connection:

```rust
let entry = swarm.pick_least_loaded().unwrap();
let device = node.device_from_ticket(&entry.advert.ticket, 0); // normal burn-remote data plane
```

**Gossip is coordination only.** Announcements are small and flooded to every subscriber; tensors
never travel through gossip — they go over the direct connection the announced `RemoteTicket` points
at. Heartbeat liveness + a TTL make the roster self-healing under churn, which is exactly what a pool
of phones (background tabs suspend, screens lock) needs.

## Run the demo

```sh
# 1. a seed — prints a JOIN TICKET (and a scannable QR) and acts as the gossip bootstrap node.
#    Pass a landing URL to get a browser-peer launch link: <landing-url>#<ticket>
cargo run -p remote-swarm --bin swarm-demo -- seed burn-web
cargo run -p remote-swarm --bin swarm-demo -- seed burn-web http://localhost:8000  # QR opens the browser peer

# 2. compute peers — paste the ticket the seed printed
cargo run -p remote-swarm --bin swarm-demo -- peer burnswarm... "phone-A"
cargo run -p remote-swarm --bin swarm-demo -- peer burnswarm... "phone-B"

# 3. a watcher — the role a client/scheduler plays: just read the roster
cargo run -p remote-swarm --bin swarm-demo -- watch burnswarm...
```

Each node prints its roster every couple of seconds. Watch it grow as peers join and shrink when one
leaves (Ctrl+C sends a graceful `Bye`; killing a process hard lets the TTL evict it).

Add `--local` (before the role) for a **relay-free** run where peers connect over direct
loopback/LAN paths only — handy for a same-host smoke test or an offline LAN with no internet:

```sh
cargo run -p remote-swarm --bin swarm-demo -- --local seed burn-web
cargo run -p remote-swarm --bin swarm-demo -- --local peer burnswarm... "phone-A"
```

> Local mode has no relay to fall back on, so direct paths can decay and the mesh may not survive
> churn for long — it's a convenience for quick local testing. Real multi-device runs use the
> default path (iroh's relays); a busy demo should point at a self-hosted relay.

## How this maps to the QR / scan-to-join vision

- The `JoinTicket` (`topic` + bootstrap `EndpointAddr`s) is the **QR payload**: base32, URL-safe.
  A landing page reads it from the URL fragment (`https://host/#burnswarm...`) and joins.
- A **browser peer** (see `examples/remote-compute-peer-web`) would `Swarm::join` on the *same*
  endpoint it serves compute on — one endpoint, two ALPNs: `BURN_REMOTE_ALPN` for the data plane and
  `GOSSIP_ALPN` (re-exported here) for the control plane — then advertise its `RemoteTicket`.
- A native or browser **client** joins as an observer, reads the roster, and dials peers.

## Status / next steps

- **Native and browser.** The networked `Swarm` spawns its tasks and times its heartbeats through
  `n0-future`, on a portable `web-time` clock, so the library compiles for `wasm32-unknown-unknown`
  as well as native. The browser compute peer
  ([`remote-compute-peer-web`](../remote-compute-peer-web)) joins the swarm this way, serving compute
  and gossip on one endpoint. The pure pieces (`message`, `roster`, `ticket`) are unit tested.
- **Trust.** Adverts are currently taken at face value. Production should sign them (gossip's
  `delivered_from` is the forwarding neighbour, not the author) and/or gate access through the
  `authorization` bytes in the `RemoteTicket`, validated by a `PeerAuthorizer`.
- **Scheduling.** `pick_least_loaded` is a starting point; because every node holds the full roster,
  a coordinator-free scheduler (rendezvous-hash work → peer) is a natural extension.
