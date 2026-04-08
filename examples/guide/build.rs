use std::env;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_DYLIB").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR should always be defined"),
    );
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("guide should be in examples/guide under workspace root");

    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target"));

    let profile = env::var("PROFILE").expect("PROFILE should always be defined");
    let plugin_name = format!(
        "{}burn_dispatch.{}",
        env::consts::DLL_PREFIX,
        env::consts::DLL_EXTENSION
    );
    let plugin_path = target_dir.join(profile).join(plugin_name);

    println!(
        "cargo:rustc-env=GUIDE_DYLIB_PLUGIN_PATH={}",
        plugin_path.display()
    );

    if !plugin_path.exists() {
        println!(
            "cargo:warning=guide dylib backend expects burn-dispatch plugin at {}",
            plugin_path.display()
        );
    }
}
