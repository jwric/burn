# Browser Compute Peer

A Burn Remote **compute peer that runs in the browser**. The tab binds an Iroh endpoint and serves
tensor operations submitted by remote clients — the mirror image of the browser *client* examples
([`remote-inference-web`](../remote-inference-web), [`remote-playground-web`](../remote-playground-web)):
there the browser offloads work to a remote peer; here the browser **is** the peer.

This is the demonstration for the wasm server port (see
[`crates/burn-remote/docs/wasm-server-feasibility.md`](../../crates/burn-remote/docs/wasm-server-feasibility.md)).
It proves the server half of Burn Remote runs in wasm: a browser endpoint accepts inbound Iroh
connections (brokered by a relay) and executes tensor ops on its own backend.

## Joining the swarm

This peer doesn't just serve — it **joins a gossip swarm** and advertises itself, so clients can
discover it instead of needing its address up front. It runs two protocols on one Iroh endpoint:
Burn Remote (`BURN_REMOTE_ALPN`, the tensor data plane) and iroh-gossip (`GOSSIP_ALPN`, the
discovery control plane), composed into a single router. Discovery lives in the
[`remote-swarm`](../remote-swarm) example crate.

The tab launches from a join ticket in the URL fragment (`…/#burnswarm…`, e.g. from a scanned QR
code), giving it the gossip topic and a bootstrap peer; it then binds a fresh identity, announces its
`RemoteTicket`, and appears in every other node's roster. A top bar shows the live swarm membership.

## API: `burn::server::serve_builder_with_telemetry`

To share one endpoint between compute and gossip, the peer uses
[`burn::server::serve_builder_with_telemetry(device, node, probe)`], which returns a `RouterBuilder`
pre-loaded with the Burn Remote protocol instead of a spawned router. The peer registers the gossip
protocol on it and calls `.spawn()`:

```rust,ignore
let gossip = Gossip::builder().spawn(endpoint.clone());
let router = serve_builder_with_telemetry(device, node.clone(), probe)
    .accept(GOSSIP_ALPN, gossip.clone())
    .spawn();
let swarm = Swarm::join(endpoint.clone(), &gossip, config).await?;
```

Its accept loop runs on the JS event loop in the browser (and on a tokio runtime natively). The
plain `serve` / `serve_with_telemetry` entries (which spawn their own single-protocol router) remain
for peers that don't share the endpoint; the blocking `start` / `start_async` helpers and the
WebSocket transport stay native-only.

## Live dashboard

The whole page is an egui canvas (via `eframe`). It aggregates the telemetry probe in-process into
a current, windowed view: throughput stats (ops/sec, transfers/sec — not totals since boot), an
animated op-class flow graph showing how tensors transit between op categories, an animated peer
map, and a recent op stream. The shared dashboard lives in
[`remote-compute-dashboard`](../remote-compute-dashboard); the same UI, fed the same state over
SSE, backs the native peer's HTTP dashboard (see [`remote-dashboard-web`](../remote-dashboard-web)).

## Backend: WebGPU, with a CPU fallback

The peer serves on the **WebGPU backend** (`Device::wgpu_async`) when the browser exposes a usable
adapter, so the tab donates its GPU. WebGPU needs a one-line fix to `cubecl-runtime` (gating
`ComputeClient::read_lazy` to non-wasm, matching its async twin), carried on the `jwric/cubecl` fork
the workspace points at — without it the WebGPU backend does not build for `wasm32`.

When WebGPU is unavailable (older Safari/Firefox, or an insecure context), the peer falls back to the
portable **Flex CPU backend** (`Device::flex`) so the tab can still join and serve — slower, but it
works everywhere. It probes `navigator.gpu.requestAdapter()` before choosing, because `wgpu_async`
panics *unrecoverably* in wasm when there's no adapter. The advertised `PeerCaps.backend` reports
which backend a peer runs, so a client can prefer GPU peers.

## Limitations of a browser peer

A browser peer runs on the single JS event loop, so it serves **independent sessions** only — it
cannot host co-located collective or same-host-transfer participants (see the worker module docs in
`burn-remote`). Iroh relays remain in the connection path, and the peer sees the plaintext tensor
data and operations it computes on.

The page takes a [Screen Wake Lock](https://developer.mozilla.org/en-US/docs/Web/API/Screen_Wake_Lock_API)
so a foreground tab keeps serving without the screen dimming or locking (re-acquired after the tab is
hidden, and a no-op where the API is unavailable). Switching to another app still backgrounds and
suspends the tab — keep this one in front to keep contributing.

## Running it

1. Build the wasm bundle:

   ```sh
   ./build-for-web.sh
   ```

2. Serve the directory and open it:

   ```sh
   ./run-server.sh   # http://localhost:8000
   ```

3. Start a swarm seed and get a launch link + QR (serving the page at, say, `http://localhost:8000`):

   ```sh
   cargo run -p remote-swarm --bin swarm-demo -- seed burn-web http://localhost:8000
   ```

   Open the printed link (or scan the QR) — the page reads the ticket from its `#…` fragment, joins
   the swarm, and the canvas switches to the live dashboard. Open it on several devices to grow the
   swarm. (You can also just enter a topic or ticket in the UI and click **Start serving**.)

4. Point a client at the **same topic/swarm** — for example run the `remote-playground-web` example
   and connect to `burn-web`. Its tensor operations now execute in a serving tab.
