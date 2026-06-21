# Remote Tensor Playground on the Web

A small **tensor REPL in the browser** whose expressions are evaluated on a remote
[Iroh](https://iroh.computer/) compute peer. Type an expression, it runs on the peer's backend
(CPU or GPU), and the result is read back. Variables persist across lines, like notebook cells.

This is the scripting/notebook counterpart to the [`remote-inference-web`](../remote-inference-web)
and [`remote-training-web`](../remote-training-web) demos.

## What you can type

```text
a = randn(3, 4)      # a 3x4 tensor of normal samples
b = ones(4, 2)
a @ b                # matrix multiply (prints the 3x2 result)
relu(a) + 1          # elementwise op + scalar broadcast
mean(a * a)          # reduction to a 1x1 tensor
t(a)                 # transpose
```

Supported: scalars and 2-D tensors; `+ - * /` (elementwise, with scalar broadcast); `@` (matmul);
`zeros`, `ones`, `rand`, `randn`; `relu`, `sigmoid`, `tanh`, `exp`, `sin`, `cos`, `abs`,
`t`/`transpose`; `sum`, `mean`. Assignment with `name = expr`.

## Running it

```sh
# 1. Start a compute peer (CPU, or `--features wgpu` for GPU)
cargo run -p remote-compute-peer -- burn-web

# 2. Build the web client
cd examples/remote-playground-web
./build-for-web.sh

# 3. Serve and open http://localhost:8000
./run-server.sh
```

Enter the same topic, click **Connect**, then type expressions.

## How it works

The parser and evaluator (`src/eval.rs`) turn each line into Burn tensor operations on the session's
device. Because that device is a remote Iroh device, the operations execute on the peer; only the
value being displayed is read back, with `into_data_async().await`. The evaluator itself is
backend-agnostic and unit-tested against the local CPU backend (`cargo test -p remote-playground-web`).

This is the core of a browser notebook: a persistent session plus a small surface of operations
submitted to a GPU peer. A richer notebook would grow the grammar (more ops, control flow) or wire
these calls to a cell-based UI.
