# remote-swarm-client-web

A **browser client** for the Burn compute swarm. It joins the gossip topic as an observer, discovers
the compute peers from the roster ([`remote-swarm`](../remote-swarm)), and continuously fans an
**animated Mandelbrot zoom** across them — each horizontal band computed on a different peer
(`device_from_ticket` → ordinary Burn Remote, read back async), re-dispatched every frame so the
peers never go idle, and stitched into the image on the canvas. The browser twin of the native
[`remote-swarm-cluster`](../remote-swarm-cluster) client.

Peers are ranked GPU-first (`PeerCaps.backend`), so WebGPU phones ([`remote-compute-peer-web`](../remote-compute-peer-web))
are preferred and CPU tabs are overflow. The roster is re-read each frame, so peers joining or
leaving are picked up live (a peer that drops just loses its band that frame). The header shows the
peer count, frame, sustained ~GFLOP/s, and tiles done; a side panel maps each band to its peer.

## Run it

```sh
# 1. build the wasm bundle and serve it
./build-for-web.sh
python3 -m http.server 8000   # then open http://localhost:8000

# 2. start a swarm (a seed + some peers) and copy the seed's ticket
cargo run -p remote-swarm-cluster -- serve burn-web alice
cargo run -p remote-swarm-cluster -- serve <ticket> bob
#    …or browser peers: cargo run -p remote-swarm -- seed burn-web http://<peer-host>
```

Open `http://localhost:8000/#<ticket>` (or enter the ticket/topic in the UI) and the client renders
the Mandelbrot on the swarm. The compute runs on the peers; only the tile data crosses the wire.

> Needs reachable iroh relays (the default path) to dial peers across machines; for a same-host smoke
> test, run the native `remote-swarm-cluster` end to end instead.
