# remote-swarm-cluster

A complete Burn compute **swarm** in one native binary — the proof of the whole
discover → dial → compute → collect loop, and the native twin of the browser peer
([`remote-compute-peer-web`](../remote-compute-peer-web)).

- **`serve`** brings up a compute peer: it serves tensor ops on the Flex CPU backend *and* joins the
  gossip swarm, advertising its `RemoteTicket` — compute and discovery on one Iroh endpoint (via
  `burn::server::serve_builder` + `remote-swarm`).
- **`client`** joins the swarm as an observer, discovers the serving peers from the roster
  ([`remote-swarm`](../remote-swarm)), and **fans a batch of work across them**: it renders a
  Mandelbrot set split into horizontal bands, each band computed on a different peer
  (`device_from_ticket` → ordinary Burn Remote), then stitched back into one ASCII image.

## Run it (one host, relay-free)

```sh
# 1. a seed peer (label, no ticket) — prints a JOIN TICKET
cargo run -p remote-swarm-cluster -- --local serve burn-web alice

# 2. another peer — paste the ticket
cargo run -p remote-swarm-cluster -- --local serve <ticket> bob

# 3. the client — discovers alice + bob and fans the Mandelbrot across them
cargo run -p remote-swarm-cluster -- --local client <ticket>
```

The client prints the fractal, a per-band legend of which peer computed which band, and a
correctness check (it recomputes band 0 locally and compares). Drop `--local` to run across machines
over iroh's relays; peers and the client can be on different hosts.

## Why it matters

This is the swarm's *consumer* — the half that turns "peers advertise GPUs" into "peers do your
compute". The client ranks the roster GPU-first (`PeerCaps.backend`), so when browser peers join
([`remote-compute-peer-web`](../remote-compute-peer-web)), their WebGPU devices are preferred and CPU
tabs are overflow. The Mandelbrot fan-out is embarrassingly parallel and churn-tolerant — the same
shape as the "crowd render wall": each scanned phone is a peer, each tile lands on whoever's
available.

## Notes

- The client's remote tensor ops are blocking; they run fine on the multi-threaded Tokio runtime
  (iroh I/O uses other worker threads), matching the native client pattern in
  [`p2p-remote-training`](../p2p-remote-training).
- `--local` uses iroh's relay-free preset (direct/loopback paths only). With no relay to fall back
  on, a long-lived mesh can decay under churn — fine for a quick fan-out, but real multi-device runs
  should use the default (relayed) path.
