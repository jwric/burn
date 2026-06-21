# Burn in a Rust Notebook, on a Remote GPU Peer

Drive Burn from a Rust REPL or Jupyter notebook while the computation runs on a remote
[Iroh](https://iroh.computer/) compute peer. Notebook cells are synchronous Rust — no `async`, no
`#[tokio::main]` — because `RemoteNode::bind_blocking` builds and owns its runtime.

This complements the browser demos ([`remote-inference-web`](../remote-inference-web),
[`remote-training-web`](../remote-training-web), [`remote-playground-web`](../remote-playground-web)):
same remote backend, but a native notebook instead of WebAssembly.

## Quick start (script form)

The runnable script in `src/main.rs` mirrors the notebook cell-by-cell:

```sh
# 1. Start a compute peer (CPU, or `--features wgpu` for GPU)
cargo run -p remote-compute-peer -- burn-web

# 2. Run the script against it
cargo run -p remote-notebook -- burn-web
```

## Jupyter (evcxr)

1. Install the Rust kernel:

   ```sh
   cargo install evcxr_jupyter
   evcxr_jupyter --install
   ```

2. Start a compute peer (see above).

3. Launch Jupyter **from the repository root** (the notebook uses a `path` dependency on the local
   `burn` crate) and open `examples/remote-notebook/notebook.ipynb`:

   ```sh
   jupyter notebook
   ```

   Run the cells top to bottom. The first `:dep` cell compiles Burn once; later cells reuse the
   `device` binding, so each cell submits its operations to the peer and prints the result.

To depend on a published Burn instead of the local checkout, replace the `:dep burn = { path = ... }`
line with a version, e.g. `:dep burn = { version = "0.22", features = ["extension", "remote-iroh", "flex"] }`.

## Why this works

`bind_blocking` returns a node whose Iroh endpoint runs on a runtime the node keeps alive, so
`Device::remote_iroh(&node, peer, 0)` and every subsequent tensor operation are ordinary synchronous
calls. The synchronous client path is covered by the `synchronous_client_round_trip` integration test
in `burn-remote`.
