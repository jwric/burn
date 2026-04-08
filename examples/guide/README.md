# Basic Workflow: From Training to Inference

This example corresponds to the [book's guide](https://burn.dev/books/burn/basic-workflow/).

## Example Usage

### Training

```sh
cargo run --bin train --release
```

### Inference

```sh
cargo run --bin infer --release
```

### Print the model

```sh
cargo run --bin print --release
```

## Dynamic Backend (Burn Dispatch Plugin)

You can run this guide without statically linking a concrete backend by using Burn's `dylib` feature.

1. Build the dispatch plugin as a shared library (lightweight recommended):

```sh
cargo rustc -p burn-dispatch --crate-type cdylib --release --no-default-features --features std,plugin,dylib,ndarray
```

Optional CUDA plugin build:

```sh
cargo rustc -p burn-dispatch --crate-type cdylib --release --no-default-features --features std,plugin,dylib,cuda
```

Optional WebGPU plugin build:

```sh
cargo rustc -p burn-dispatch --crate-type cdylib --release --no-default-features --features std,plugin,dylib,webgpu
```

Important: keep `--no-default-features` and select explicit backend features for the plugin.
Building `burn-dispatch` with default features can pull fusion-enabled GPU stacks that may panic at runtime in dynamic plugin mode.

1. Run the guide with the dynamic backend:

```sh
cargo run --manifest-path examples/guide/Cargo.toml --bin train --release --no-default-features --features dylib
```

The guide build script sets a plugin path to `target/{profile}/libburn_dispatch.*` automatically when `dylib` is enabled.

By default, guide attempts dispatch backend type IDs in this order: `ndarray` (`60`), `cpu` (`0`), then `cuda` (`10`).
You can override this by setting `GUIDE_DYLIB_BACKEND_TYPE_ID`.
If the selected type id isn't available in the loaded plugin, guide now fails fast with a clear error.

Example forcing CUDA dispatch device type:

```sh
GUIDE_DYLIB_BACKEND_TYPE_ID=10 cargo run --manifest-path examples/guide/Cargo.toml --bin train --release --no-default-features --features dylib
```
