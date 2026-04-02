#![cfg(feature = "dylib")]

use burn_backend::{DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape, TensorData};
use burn_dylib::TensorBinaryOp;
use burn_dylib::loader::{LoadError, LoadedBackendPlugin, PluginCallError};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use super::device::DylibDevice;
use super::tensor::DylibTensor;

/// Errors that can occur while interacting with a runtime-loaded backend.
#[derive(Debug, Clone)]
pub enum DylibError {
    /// Shared library loading error.
    Load(String),
    /// Runtime id is unknown to the current process registry.
    RuntimeNotFound(u64),
    /// Device id is unknown to the current process registry.
    DeviceNotFound(u32),
    /// Backend plugin operation failed.
    Plugin(String),
    /// Conversion between host tensor data and plugin tensor data failed.
    Data(String),
    /// Invalid input provided to a plugin operation.
    InvalidInput(String),
}

impl Display for DylibError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Load(reason) => write!(f, "Failed to load dylib backend: {reason}"),
            Self::RuntimeNotFound(id) => write!(f, "Unknown dylib runtime id: {id}"),
            Self::DeviceNotFound(id) => write!(f, "Unknown dylib dispatch device id: {id}"),
            Self::Plugin(reason) => write!(f, "Dylib plugin error: {reason}"),
            Self::Data(reason) => write!(f, "Tensor data conversion error: {reason}"),
            Self::InvalidInput(reason) => write!(f, "Invalid dylib input: {reason}"),
        }
    }
}

impl std::error::Error for DylibError {}

pub(crate) struct DylibRuntime {
    pub(crate) id: u64,
    pub(crate) path: String,
    pub(crate) plugin: Arc<LoadedBackendPlugin>,
}

impl core::fmt::Debug for DylibRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DylibRuntime")
            .field("id", &self.id)
            .field("path", &self.path)
            .finish()
    }
}

static NEXT_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_DEVICE_INDEX: AtomicU32 = AtomicU32::new(1);

static RUNTIMES: LazyLock<RwLock<HashMap<u64, Arc<DylibRuntime>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static RUNTIMES_BY_PATH: LazyLock<RwLock<HashMap<String, u64>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static DEVICES: LazyLock<RwLock<HashMap<u32, DylibDevice>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn normalize_path(path: &Path) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

fn map_load_error(err: LoadError) -> DylibError {
    DylibError::Load(err.to_string())
}

fn map_call_error(err: PluginCallError) -> DylibError {
    DylibError::Plugin(err.to_string())
}

pub(crate) fn to_execution_error(err: DylibError) -> ExecutionError {
    ExecutionError::WithContext {
        reason: format!("dylib dispatch error: {err}"),
    }
}

pub(crate) fn get_runtime(runtime_id: u64) -> Result<Arc<DylibRuntime>, DylibError> {
    RUNTIMES
        .read()
        .unwrap()
        .get(&runtime_id)
        .cloned()
        .ok_or(DylibError::RuntimeNotFound(runtime_id))
}

pub(crate) fn register_runtime(path: impl AsRef<Path>) -> Result<u64, DylibError> {
    let key = normalize_path(path.as_ref());

    if let Some(id) = RUNTIMES_BY_PATH.read().unwrap().get(&key).copied() {
        return Ok(id);
    }

    let mut path_map = RUNTIMES_BY_PATH.write().unwrap();
    if let Some(id) = path_map.get(&key).copied() {
        return Ok(id);
    }

    let plugin = unsafe { LoadedBackendPlugin::load(path.as_ref()) }.map_err(map_load_error)?;
    let id = NEXT_RUNTIME_ID.fetch_add(1, Ordering::Relaxed);

    let runtime = Arc::new(DylibRuntime {
        id,
        path: key.clone(),
        plugin: Arc::new(plugin),
    });

    RUNTIMES.write().unwrap().insert(id, runtime);
    path_map.insert(key, id);

    Ok(id)
}

pub(crate) fn create_device_from_runtime(
    runtime_id: u64,
    backend_type_id: u16,
    ordinal: usize,
) -> Result<DylibDevice, DylibError> {
    let ordinal_u32 = u32::try_from(ordinal)
        .map_err(|_| DylibError::InvalidInput(format!("Invalid device ordinal: {ordinal}")))?;

    let runtime = get_runtime(runtime_id)?;
    let handle = runtime
        .plugin
        .create_device(backend_type_id, ordinal)
        .map_err(map_call_error)?;

    let registry_index = NEXT_DEVICE_INDEX.fetch_add(1, Ordering::Relaxed);
    let device = DylibDevice {
        registry_index,
        runtime_id,
        backend_type_id,
        ordinal: ordinal_u32,
        handle,
    };

    DEVICES
        .write()
        .unwrap()
        .insert(device.registry_index, device.clone());

    Ok(device)
}

pub(crate) fn create_device_from_path(
    path: impl AsRef<Path>,
    backend_type_id: u16,
    ordinal: usize,
) -> Result<DylibDevice, DylibError> {
    let runtime_id = register_runtime(path)?;
    create_device_from_runtime(runtime_id, backend_type_id, ordinal)
}

pub(crate) fn device_from_registry(index_id: u32) -> Result<DylibDevice, DylibError> {
    DEVICES
        .read()
        .unwrap()
        .get(&index_id)
        .cloned()
        .ok_or(DylibError::DeviceNotFound(index_id))
}

pub(crate) fn backend_name(device: &DylibDevice) -> String {
    match get_runtime(device.runtime_id)
        .and_then(|runtime| runtime.plugin.name().map_err(map_call_error))
    {
        Ok(name) => format!("dylib<{name}>"),
        Err(err) => format!("dylib<error:{err}>"),
    }
}

pub(crate) fn backend_seed(device: &DylibDevice, seed: u64) {
    let _ = get_runtime(device.runtime_id)
        .and_then(|runtime| runtime.plugin.seed(seed).map_err(map_call_error));
}

pub(crate) fn backend_sync(device: &DylibDevice) -> Result<(), ExecutionError> {
    let runtime = get_runtime(device.runtime_id).map_err(to_execution_error)?;
    runtime
        .plugin
        .sync()
        .map_err(map_call_error)
        .map_err(to_execution_error)
}

pub(crate) fn dtype_usage(dtype: DType) -> DTypeUsageSet {
    match dtype {
        DType::F32 => DTypeUsage::general(),
        _ => DTypeUsageSet::default(),
    }
}

pub(crate) fn tensor_from_data(
    data: TensorData,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    let data_f32 = data.convert::<f32>();
    let requested_shape = data_f32.shape.clone();
    let values = data_f32
        .into_vec::<f32>()
        .map_err(|err| DylibError::Data(err.to_string()))?;

    let runtime = get_runtime(device.runtime_id)?;
    let handle = runtime
        .plugin
        .tensor_from_f32_data(device.handle, requested_shape.as_slice(), &values)
        .map_err(map_call_error)?;

    let shape = runtime
        .plugin
        .tensor_shape(handle)
        .map(|dims| Shape::new_raw(dims.into()))
        .unwrap_or(requested_shape);

    Ok(DylibTensor::new(
        device.runtime_id,
        device.clone(),
        handle,
        DType::F32,
        shape,
    ))
}

pub(crate) fn tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let values = runtime
        .plugin
        .tensor_into_f32_data(tensor.handle)
        .map_err(map_call_error)?;

    let shape = runtime
        .plugin
        .tensor_shape(tensor.handle)
        .map(|dims| Shape::new_raw(dims.into()))
        .unwrap_or_else(|_| tensor.shape.clone());

    Ok(TensorData::new(values, shape))
}

pub(crate) fn tensor_to_device(
    tensor: DylibTensor,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    if tensor.device == *device {
        return Ok(tensor);
    }

    let data = tensor_into_data(tensor)?;
    tensor_from_data(data, device)
}

pub(crate) fn tensor_binary(
    lhs: DylibTensor,
    rhs: DylibTensor,
    op: TensorBinaryOp,
) -> Result<DylibTensor, DylibError> {
    if lhs.runtime_id != rhs.runtime_id {
        return Err(DylibError::InvalidInput(format!(
            "Cross-runtime operations are not supported (lhs={}, rhs={})",
            lhs.runtime_id, rhs.runtime_id
        )));
    }

    let runtime = get_runtime(lhs.runtime_id)?;
    let handle = runtime
        .plugin
        .tensor_binary(op, lhs.handle, rhs.handle)
        .map_err(map_call_error)?;

    let shape = runtime
        .plugin
        .tensor_shape(handle)
        .map(|dims| Shape::new_raw(dims.into()))
        .unwrap_or_else(|_| lhs.shape.clone());

    Ok(DylibTensor::new(
        lhs.runtime_id,
        lhs.device,
        handle,
        DType::F32,
        shape,
    ))
}
