//
// Note: If you are following the Burn Book guide this file can be ignored.
//
// This lib.rs file is added only for convenience so that the code in this
// guide can be reused.
//
#[cfg(all(feature = "webgpu", feature = "vulkan"))]
compile_error!("Features `webgpu` and `vulkan` are mutually exclusive.");

#[cfg(all(feature = "dylib", any(feature = "webgpu", feature = "vulkan")))]
compile_error!(
    "Feature `dylib` is mutually exclusive with `webgpu`/`vulkan`. Use `--no-default-features --features dylib`."
);

#[cfg(not(any(feature = "dylib", feature = "webgpu", feature = "vulkan")))]
compile_error!("Enable one backend feature: `webgpu`, `vulkan`, or `dylib`.");

pub mod backend;
pub mod data;
pub mod inference;
pub mod model;
pub mod training;
