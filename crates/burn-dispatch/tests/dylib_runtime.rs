#![cfg(all(feature = "dylib", feature = "std"))]

use burn_backend::ops::FloatTensorOps;
use burn_backend::{Backend, Shape, TensorData};
use burn_dispatch::{Dispatch, DispatchDevice};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

static PLUGIN_A: OnceLock<PathBuf> = OnceLock::new();
static PLUGIN_B: OnceLock<PathBuf> = OnceLock::new();

fn fixture_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mock-plugin")
        .join("Cargo.toml")
}

fn target_lib_filename() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "libburn_dispatch_test_plugin.so"
    }

    #[cfg(target_os = "macos")]
    {
        "libburn_dispatch_test_plugin.dylib"
    }

    #[cfg(target_os = "windows")]
    {
        "burn_dispatch_test_plugin.dll"
    }
}

fn build_plugin(variant_b: bool) -> PathBuf {
    let fixture_manifest = fixture_manifest_path();
    let profile_dir = if variant_b {
        "dylib-plugin-b"
    } else {
        "dylib-plugin-a"
    };

    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(profile_dir);

    let mut command = Command::new("cargo");
    command
        .arg("build")
        .arg("--manifest-path")
        .arg(&fixture_manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .arg("--release");

    if variant_b {
        command.arg("--features").arg("variant-b");
    }

    let status = command.status().expect("fixture plugin build should run");
    assert!(
        status.success(),
        "fixture plugin build should succeed (variant_b={variant_b})"
    );

    let dylib_path = target_dir.join("release").join(target_lib_filename());
    assert!(
        dylib_path.exists(),
        "fixture plugin dylib should exist at {}",
        dylib_path.display()
    );

    dylib_path
}

fn plugin_path(variant_b: bool) -> &'static Path {
    if variant_b {
        PLUGIN_B.get_or_init(|| build_plugin(true)).as_path()
    } else {
        PLUGIN_A.get_or_init(|| build_plugin(false)).as_path()
    }
}

fn as_f32_vec(data: TensorData) -> Vec<f32> {
    data.into_vec::<f32>()
        .expect("tensor data should decode as f32")
}

#[test]
fn dylib_backend_runs_add() {
    let device =
        DispatchDevice::dylib(plugin_path(false), 0, 0).expect("dylib device should be created");

    assert_eq!(Dispatch::name(&device), "dispatch<dylib<mock-plugin-a>>");

    let lhs = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2])),
        &device,
    );
    let rhs = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2])),
        &device,
    );

    let add = <Dispatch as FloatTensorOps<Dispatch>>::float_add(lhs, rhs);
    let add_data =
        burn_backend::read_sync(<Dispatch as FloatTensorOps<Dispatch>>::float_into_data(add))
            .expect("add result should be readable");

    assert_eq!(as_f32_vec(add_data), vec![6.0, 8.0, 10.0, 12.0]);

}

#[test]
fn dylib_runtime_swap_loads_different_plugin_variants() {
    let device_a =
        DispatchDevice::dylib(plugin_path(false), 0, 0).expect("variant A device should load");
    let device_b =
        DispatchDevice::dylib(plugin_path(true), 0, 0).expect("variant B device should load");

    assert_eq!(Dispatch::name(&device_a), "dispatch<dylib<mock-plugin-a>>");
    assert_eq!(Dispatch::name(&device_b), "dispatch<dylib<mock-plugin-b>>");

    let lhs_a = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![1.0, 1.0], Shape::new([2])),
        &device_a,
    );
    let rhs_a = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![2.0, 3.0], Shape::new([2])),
        &device_a,
    );

    let lhs_b = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![1.0, 1.0], Shape::new([2])),
        &device_b,
    );
    let rhs_b = <Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
        TensorData::new(vec![2.0, 3.0], Shape::new([2])),
        &device_b,
    );

    let out_a = <Dispatch as FloatTensorOps<Dispatch>>::float_add(lhs_a, rhs_a);
    let out_b = <Dispatch as FloatTensorOps<Dispatch>>::float_add(lhs_b, rhs_b);

    let out_a_data = burn_backend::read_sync(
        <Dispatch as FloatTensorOps<Dispatch>>::float_into_data(out_a),
    )
    .expect("variant A output should be readable");
    let out_b_data = burn_backend::read_sync(
        <Dispatch as FloatTensorOps<Dispatch>>::float_into_data(out_b),
    )
    .expect("variant B output should be readable");

    assert_eq!(as_f32_vec(out_a_data), vec![3.0, 4.0]);
    assert_eq!(as_f32_vec(out_b_data), vec![4.0, 5.0]);
}
