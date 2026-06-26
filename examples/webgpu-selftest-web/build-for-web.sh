#!/usr/bin/env bash
set -e

rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
fi

# WebGPU needs web-sys's unstable APIs.
export RUSTFLAGS='--cfg web_sys_unstable_apis --cfg getrandom_backend="wasm_js"'

mkdir -p pkg
wasm-pack build --out-dir pkg --dev --target web --no-typescript
