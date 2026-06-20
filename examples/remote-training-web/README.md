# Remote Training on the Web

Train a Burn model **in the browser** while the forward pass, the **backward pass**, and every
**optimizer update** run on a remote [Iroh](https://iroh.computer/) compute peer. A small MLP is fit
to `y = sin(3x)` with SGD; the browser only reads back scalar loss values to draw the live curve.

This is the training counterpart to [`remote-inference-web`](../remote-inference-web). The autodiff
graph is recorded on the client, but the operations it generates — including gradients and the
weight updates — execute on the peer's backend (CPU or GPU).

## Running it

### 1. Start a compute peer (native)

```sh
cargo run -p remote-compute-peer -- burn-web              # CPU
cargo run -p remote-compute-peer --features wgpu -- burn-web   # GPU
```

### 2. Build the web client

```sh
cd examples/remote-training-web
./build-for-web.sh
```

### 3. Serve and open

```sh
./run-server.sh
```

Open <http://localhost:8000>, enter the same topic, click **Connect**, then **Train**. The loss
curve updates as the peer runs each batch of steps.

## How it differs from local training

The training loop is ordinary Burn:

```rust
let prediction = model.forward(inputs);
let loss = MseLoss::new().forward(prediction, targets, Reduction::Mean);
let grads = GradientsParams::from_grads(loss.backward(), &model);
model = optim.step(lr, model, grads);
```

The only change for the browser is that the loss is read with `into_data_async().await` (a wasm
target cannot block), and the device is an Iroh remote device opened with
`Device::remote_iroh_async(...).await.autodiff()`. Everything else is backend-agnostic.

## Toward browser notebooks

This pattern — connect once, then call exported Rust functions that submit tensor operations to a
peer — is the building block for a browser-side scripting/notebook environment: a page can expose
model construction, training, and inference as functions a JS REPL or notebook cell drives, with all
heavy computation staying on a GPU peer. The two web examples here are deliberately small versions of
exactly that loop.
