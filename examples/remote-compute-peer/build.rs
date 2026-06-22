//! With the `dashboard` feature on, build the egui dashboard viewer to wasm and stage it under
//! `OUT_DIR/dashboard-dist` so `main.rs` can embed it. The nested build uses its own target
//! directory to avoid contending on the workspace target lock held by this build.

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var_os("CARGO_FEATURE_DASHBOARD").is_none() {
        return;
    }

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let viewer = Path::new(&manifest).join("../remote-dashboard-web");
    let shared = Path::new(&manifest).join("../remote-compute-dashboard/src");
    let dist = Path::new(&out_dir).join("dashboard-dist");
    let pkg = dist.join("pkg");
    let wasm_target = Path::new(&out_dir).join("wasm-target");

    println!("cargo:rerun-if-changed={}", viewer.join("src").display());
    println!("cargo:rerun-if-changed={}", viewer.join("Cargo.toml").display());
    println!("cargo:rerun-if-changed={}", viewer.join("index.html").display());
    println!("cargo:rerun-if-changed={}", shared.display());

    std::fs::create_dir_all(&dist).expect("create dashboard dist dir");

    let status = Command::new("wasm-pack")
        .arg("build")
        .arg(&viewer)
        .args(["--target", "web", "--release", "--no-typescript", "--out-dir"])
        .arg(&pkg)
        .env("CARGO_TARGET_DIR", &wasm_target)
        .env("RUSTFLAGS", r#"--cfg getrandom_backend="wasm_js""#)
        .status();

    match status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("wasm-pack build of the dashboard viewer failed: {status}"),
        Err(err) => panic!(
            "could not run wasm-pack (required by the `dashboard` feature): {err}. \
             Install it with `cargo install wasm-pack`."
        ),
    }

    std::fs::copy(viewer.join("index.html"), dist.join("index.html"))
        .expect("copy dashboard index.html");
}
