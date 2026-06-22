# Remote Peer Dashboard (browser viewer)

The egui telemetry dashboard as a standalone browser page, fed over Server-Sent Events instead of
an in-process probe. A native compute peer serves this page and an `/events` stream; opening the
peer's HTTP address renders the live dashboard in a tab, with no native window.

It shares its UI with [`remote-compute-dashboard`](../remote-compute-dashboard) (current-rate stats,
animated op-class flow graph, animated peer map, recent op stream). The only difference is the data
source: here a [`StateSource`](../remote-compute-dashboard/src/lib.rs) backed by `EventSource`
decodes `DashboardState` snapshots; in the browser compute peer the source aggregates the in-process
probe directly. Because the server holds the state, the page shows connected / stale / lost and a
refresh resumes the live picture instead of resetting to zero.

## Use with the native peer

The native peer's `build.rs` builds this viewer with `wasm-pack` and embeds it into the binary, so
there is no separate build step:

```sh
cargo run -p remote-compute-peer --features dashboard -- my-topic
```

It serves the embedded viewer and the `/events` stream on `127.0.0.1:8080` (override with
`BURN_DASHBOARD_ADDR`). Open `http://127.0.0.1:8080` and point a client at `my-topic`; its tensor
operations stream in.

## Standalone

`build-for-web.sh` builds `pkg/` directly if you want to host the viewer yourself against a remote
peer's `/events` endpoint (it points at `/events` on whatever origin serves it).
