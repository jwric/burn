# Browser Compute Peer

A Burn Remote **compute peer that runs in the browser**. The tab binds an Iroh endpoint and serves
tensor operations submitted by remote clients — the mirror image of the browser *client* examples
([`remote-inference-web`](../remote-inference-web), [`remote-playground-web`](../remote-playground-web)):
there the browser offloads work to a remote peer; here the browser **is** the peer.

This is the demonstration for the wasm server port (see
[`crates/burn-remote/docs/wasm-server-feasibility.md`](../../crates/burn-remote/docs/wasm-server-feasibility.md)).
It proves the server half of Burn Remote runs in wasm: a browser endpoint accepts inbound Iroh
connections (brokered by a relay) and executes tensor ops on its own backend.

## API: `burn::server::serve_with_telemetry`

The peer uses [`burn::server::serve_with_telemetry(device, node, probe)`], which returns a running
server without blocking — its accept loop runs on the JS event loop in the browser (and on a tokio
runtime natively). It is the telemetry-emitting variant of `serve`; both are the wasm-capable
Iroh-server path, while the blocking `start` / `start_async` helpers and the WebSocket transport
remain native-only.

## Live dashboard

The whole page is an egui canvas (via `eframe`). It aggregates the telemetry probe in-process into
a current, windowed view: throughput stats (ops/sec, transfers/sec — not totals since boot), an
animated op-class flow graph showing how tensors transit between op categories, an animated peer
map, and a recent op stream. The shared dashboard lives in
[`remote-compute-dashboard`](../remote-compute-dashboard); the same UI, fed the same state over
SSE, backs the native peer's HTTP dashboard (see [`remote-dashboard-web`](../remote-dashboard-web)).

## Backend: WebGPU

The peer serves on the **WebGPU backend** (`Device::wgpu_async`), so the tab donates its GPU. This
needs a one-line fix to `cubecl-runtime` (gating `ComputeClient::read_lazy` to non-wasm, matching its
async twin), carried on the `jwric/cubecl` fork the workspace points at — without it the WebGPU
backend does not build for `wasm32`.

## Limitations of a browser peer

A browser peer runs on the single JS event loop, so it serves **independent sessions** only — it
cannot host co-located collective or same-host-transfer participants (see the worker module docs in
`burn-remote`). Iroh relays remain in the connection path, and the peer sees the plaintext tensor
data and operations it computes on.

## Running it

1. Build the wasm bundle:

   ```sh
   ./build-for-web.sh
   ```

2. Serve the directory and open it:

   ```sh
   ./run-server.sh   # http://localhost:8000
   ```

3. Enter a topic and click **Start serving**. The canvas switches to the live dashboard.

4. Point a client at the **same topic** — for example run the `remote-playground-web` example and
   connect to `burn-web`. Its tensor operations now execute in the serving tab.
