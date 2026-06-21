# Browser Compute Peer

A Burn Remote **compute peer that runs in the browser**. The tab binds an Iroh endpoint and serves
tensor operations submitted by remote clients — the mirror image of the browser *client* examples
([`remote-inference-web`](../remote-inference-web), [`remote-playground-web`](../remote-playground-web)):
there the browser offloads work to a remote peer; here the browser **is** the peer.

This is the demonstration for the wasm server port (see
[`crates/burn-remote/docs/wasm-server-feasibility.md`](../../crates/burn-remote/docs/wasm-server-feasibility.md)).
It proves the server half of Burn Remote runs in wasm: a browser endpoint accepts inbound Iroh
connections (brokered by a relay) and executes tensor ops on its own backend.

## Backend: CPU now, WebGPU pending an upstream fix

The peer serves on the **Flex CPU backend**, which compiles for wasm today. The intended backend is
**WebGPU** (`burn-wgpu`), so the tab donates its GPU — but that is currently blocked by an upstream
bug in the pinned `cubecl-runtime`: `client.rs` declares `mod lazy` as `#[cfg(not(target_family =
"wasm"))]` while `read_lazy`/`read_lazy_async` use it unconditionally, so the WebGPU backend fails
to build for `wasm32`. Once that is fixed (gate those methods for wasm too), switching is a one-line
change:

```rust
// after `init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, Default::default()).await;`
let router = node.serve::<burn_wgpu::WebGpu>(vec![device]);
```

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

3. Enter a topic and click **Start serving**. The page shows the peer's endpoint id.

4. Point a client at the **same topic** — for example run the `remote-playground-web` example and
   connect to `burn-web`. Its tensor operations now execute in the serving tab.
