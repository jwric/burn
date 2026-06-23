# Office demo — scan a QR, donate your phone's GPU

A live compute swarm. People scan a QR code with their phones; each phone opens a web page that
**donates its GPU** to the swarm. On the projector, an animated Mandelbrot zoom renders **across
everyone's phones at once** — each horizontal band computed on a different device and re-dispatched
every frame, so the phones stay busy and the picture speeds up as more people join.

Everything is peer-to-peer over iroh; no central server does the math. Three roles:

| role | crate | who runs it |
| --- | --- | --- |
| compute peer | [`remote-compute-peer-web`](../remote-compute-peer-web) | phones (scan the QR) + the laptop |
| client (the zoom) | [`remote-swarm-client-web`](../remote-swarm-client-web) | the laptop, on the projector |
| bootstrap seed + QR | [`remote-swarm-cluster serve`](../remote-swarm-cluster) | the laptop |

## Two things to know up front

- **Phones reach the swarm through iroh's public relays, not your LAN.** The office network must allow
  outbound HTTPS/QUIC to the internet. (Browser peers can't do LAN-direct, so there is no fully
  offline cross-device path.) Test this before the meeting — see the last step.
- **WebGPU only runs in a secure context (HTTPS or `localhost`).** Phones therefore need the page over
  **HTTPS** to donate their GPU — hence the `cloudflared` tunnel below. Over plain HTTP they still
  join, but fall back to the CPU backend.

## Before the meeting (laptop)

Prerequisites:

- Rust ≥ 1.95 — `rustup update stable`
- `python3` — serves the pages
- `wasm-pack` — the build scripts install it if it's missing
- `cloudflared` — puts the phone page on HTTPS (mac: `brew install cloudflared`; linux: grab the
  binary from Cloudflare's releases)
- A Chromium-based browser on the laptop for the client

Build both web bundles. **The first build compiles a lot — do it ahead of time, it can take several
minutes:**

```sh
cd examples/remote-compute-peer-web && ./build-for-web.sh
cd ../remote-swarm-client-web      && ./build-for-web.sh
```

## Run the demo

Four terminals on the laptop (or background the servers with `&`).

**1 — serve the phone page locally**

```sh
cd examples/remote-compute-peer-web
./run-server.sh                       # http://localhost:8001
```

**2 — put the phone page on HTTPS**

```sh
cloudflared tunnel --url http://localhost:8001
# copy the https://<random>.trycloudflare.com URL it prints
```

**3 — serve the client page (the zoom)**

```sh
cd examples/remote-swarm-client-web
./run-server.sh                       # http://localhost:8002
```

**4 — bootstrap the swarm + print the QR** (this terminal is also a CPU peer, so the zoom renders
even before anyone scans):

```sh
cargo run -p remote-swarm-cluster -- serve burn-office laptop https://<random>.trycloudflare.com
```

It prints a `JOIN TICKET` and a QR for `https://<random>.trycloudflare.com#<ticket>`.

**Then:**

- **Phones:** scan the QR from terminal 4. The phone opens the page over HTTPS, shows
  `browser · wgpu · …`, and is now a GPU peer in the swarm.
- **Laptop / projector:** open `http://localhost:8002/#<ticket>` (paste the `JOIN TICKET` after the
  `#`). The Mandelbrot zoom renders across the swarm. The header shows the peer count, frame,
  sustained ~GFLOP/s, and tiles done; the side panel maps each band to the peer that computed it.
  Watch the peer count climb and the throughput rise as the room scans in.

Keep each phone's tab in the foreground — switching apps suspends the tab and drops that peer.

## No-tunnel fallback (CPU only)

No `cloudflared`? Point the QR at the LAN page instead and skip terminal 2:

```sh
# terminal 4, using the laptop's LAN IP:
cargo run -p remote-swarm-cluster -- serve burn-office laptop http://<laptop-LAN-ip>:8001
```

Phones open it over plain HTTP (insecure context), so they serve on the **flex CPU** backend — no
GPU, slower, and the screen may dim. The swarm still works and renders. (Connectivity still goes
through relays, so internet is still required.)

## Smoke test (no phones, no browser, no internet)

Prove the swarm end-to-end on the laptop alone — `--local` uses iroh's relay-free preset:

```sh
# terminal A — seed
cargo run -p remote-swarm-cluster -- --local serve burn-test alice
# copy the ticket, then terminal B — client
cargo run -p remote-swarm-cluster -- --local client <ticket>
```

You get an ASCII Mandelbrot with each band labeled by the peer that computed it, plus a band-0
local re-verification.

To test the real (relayed) path before the meeting: run terminal 4 above, open the tunnel URL on
**one** phone, and watch terminal 4 report `1 other peer(s) in swarm`. If it never does, the network
is blocking the relays.

## Troubleshooting

- **`serve` hangs at startup** — it waits to come online via the relays; if it never proceeds, the
  network is blocking outbound to iroh's relays.
- **Phone shows `browser · flex · …` instead of `wgpu`** — the page wasn't loaded over HTTPS, or the
  phone has no WebGPU (need Chrome/Android or iOS 17+). The CPU fallback is expected on plain HTTP.
- **The trycloudflare URL is different every run** — yes; always run `cloudflared` first, then pass
  that URL to `serve`. Don't hard-code it.
- **A band goes blank for a frame** — a peer left mid-frame; it's picked up again next frame.
- **Smoother CPU peers** — rebuild the bundles with `--release` instead of `--dev` in
  `build-for-web.sh` (much longer build). Not needed for the GPU path.
