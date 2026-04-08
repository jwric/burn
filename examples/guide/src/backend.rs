use burn::backend::Autodiff;

#[cfg(feature = "dylib")]
pub type GuideBackend = burn::backend::dylib::Dylib<f32>;

#[cfg(feature = "vulkan")]
pub type GuideBackend = burn::backend::Vulkan<f32, i32>;

#[cfg(all(feature = "webgpu", not(feature = "vulkan")))]
pub type GuideBackend = burn::backend::WebGpu<f32, i32>;

pub type GuideAutodiffBackend = Autodiff<GuideBackend>;

#[cfg(feature = "dylib")]
fn configured_backend_type_id() -> Option<u16> {
    if let Ok(raw) = std::env::var("GUIDE_DYLIB_BACKEND_TYPE_ID") {
        return Some(raw.parse::<u16>().unwrap_or_else(|_| {
            panic!("GUIDE_DYLIB_BACKEND_TYPE_ID must be a valid u16, got '{raw}'")
        }));
    }

    option_env!("GUIDE_DYLIB_BACKEND_TYPE_ID").map(|raw| {
        raw.parse::<u16>().unwrap_or_else(|_| {
            panic!("GUIDE_DYLIB_BACKEND_TYPE_ID (compile-time) must be a valid u16, got '{raw}'")
        })
    })
}

#[cfg(feature = "dylib")]
fn try_dispatch_device_from_plugin(
    plugin_path: &std::path::Path,
) -> Option<<GuideBackend as burn::tensor::backend::Backend>::Device> {
    // Dispatch plugin backend ids are encoded as `backend_id * 10 + backend_type_id`.
    // We prefer stable CPU paths by default to avoid runtime issues in fused GPU init.
    const DISPATCH_TYPE_ID_CPU: u16 = 0;
    const DISPATCH_TYPE_ID_CUDA: u16 = 10;
    const DISPATCH_TYPE_ID_NDARRAY: u16 = 60;

    if let Some(type_id) = configured_backend_type_id() {
        let device = burn::backend::dylib::create_device_from_path(plugin_path, type_id, 0)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to create dispatch device type id {type_id} from {}: {err}",
                    plugin_path.display()
                )
            });
        return Some(device);
    }

    [
        DISPATCH_TYPE_ID_NDARRAY,
        DISPATCH_TYPE_ID_CPU,
        DISPATCH_TYPE_ID_CUDA,
    ]
    .into_iter()
    .find_map(|type_id| burn::backend::dylib::create_device_from_path(plugin_path, type_id, 0).ok())
}

#[cfg(feature = "dylib")]
fn plugin_candidates() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    if let Ok(path) = std::env::var("GUIDE_DYLIB_PLUGIN_PATH") {
        paths.push(std::path::PathBuf::from(path));
    }

    if let Some(path) = option_env!("GUIDE_DYLIB_PLUGIN_PATH") {
        paths.push(std::path::PathBuf::from(path));
    }

    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        let name = format!(
            "{}burn_dispatch.{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_EXTENSION
        );
        paths.push(exe_dir.join(name));
    }

    paths
}

pub fn create_device() -> <GuideBackend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "dylib")]
    {
        for plugin_path in plugin_candidates() {
            if !plugin_path.exists() {
                continue;
            }

            if let Some(device) = try_dispatch_device_from_plugin(&plugin_path) {
                return device;
            }

            if let Ok(device) = burn::backend::dylib::create_default_device_from_path(&plugin_path)
            {
                return device;
            }
        }

        return burn::backend::dylib::create_default_device().unwrap_or_else(|err| {
            panic!(
                "Failed to create default dylib device. Build burn-dispatch as a cdylib and place libburn_dispatch next to the executable. You can force backend and plugin paths with GUIDE_DYLIB_BACKEND_TYPE_ID and GUIDE_DYLIB_PLUGIN_PATH: {err}"
            )
        });
    }

    #[cfg(any(feature = "webgpu", feature = "vulkan"))]
    {
        burn::backend::wgpu::WgpuDevice::default()
    }
}
