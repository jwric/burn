#![cfg(feature = "dylib")]

use burn_backend::{DType, DTypeUsage, DTypeUsageSet, ExecutionError, Shape, TensorData};
use burn_dylib::loader::{LoadError, LoadedBackendPlugin, PluginCallError};
use burn_dylib::{
    DenseTensorBinaryOp, DenseTensorKind, DeviceHandle, TensorBinaryOp, TensorHandle,
};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceSnapshot {
    pub(crate) runtime_id: u64,
    pub(crate) backend_type_id: u16,
    pub(crate) ordinal: u32,
    pub(crate) handle: DeviceHandle,
}

#[derive(Debug)]
struct DeviceEntry {
    snapshot: DeviceSnapshot,
    refs: AtomicUsize,
}

impl DeviceEntry {
    fn new(snapshot: DeviceSnapshot) -> Self {
        Self {
            snapshot,
            refs: AtomicUsize::new(1),
        }
    }

    fn snapshot(&self) -> DeviceSnapshot {
        self.snapshot
    }

    fn retain(&self) {
        self.refs.fetch_add(1, Ordering::Relaxed);
    }

    fn release(&self) -> bool {
        self.refs.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

pub(crate) struct DylibRuntime {
    id: u64,
    path: String,
    plugin: Arc<LoadedBackendPlugin>,
}

impl DylibRuntime {
    fn name(&self) -> Result<String, DylibError> {
        self.plugin.name().map_err(map_call_error)
    }

    fn seed(&self, seed: u64) -> Result<(), DylibError> {
        self.plugin.seed(seed).map_err(map_call_error)
    }

    fn sync(&self) -> Result<(), DylibError> {
        self.plugin.sync().map_err(map_call_error)
    }

    fn device_count(&self, backend_type_id: u16) -> usize {
        self.plugin.device_count(backend_type_id)
    }

    fn create_device_handle(
        &self,
        backend_type_id: u16,
        ordinal: usize,
    ) -> Result<DeviceHandle, DylibError> {
        self.plugin
            .create_device(backend_type_id, ordinal)
            .map_err(map_call_error)
    }

    fn tensor_from_f32_data(
        &self,
        device: DeviceHandle,
        shape: &Shape,
        values: &[f32],
    ) -> Result<TensorHandle, DylibError> {
        self.plugin
            .dense_float_tensor_from_f32_data(device, shape.as_slice(), values)
            .map_err(map_call_error)
    }

    fn tensor_into_f32_data(&self, tensor: TensorHandle) -> Result<(Vec<f32>, Shape), DylibError> {
        self.plugin
            .dense_float_tensor_into_f32_data(tensor)
            .map(|data| (data.values, Shape::new_raw(data.shape.into())))
            .map_err(map_call_error)
    }

    fn tensor_binary(
        &self,
        op: TensorBinaryOp,
        lhs: TensorHandle,
        rhs: TensorHandle,
    ) -> Result<TensorHandle, DylibError> {
        self.plugin
            .dense_tensor_binary(DenseTensorKind::Float, map_float_binary_op(op), lhs, rhs)
            .map_err(map_call_error)
    }

    fn tensor_shape_or(&self, tensor: TensorHandle, fallback: Shape) -> Shape {
        self.plugin
            .tensor_shape(tensor)
            .map(|dims| Shape::new_raw(dims.into()))
            .unwrap_or(fallback)
    }

    fn release_device(&self, device: DeviceHandle) {
        let _ = self.plugin.release_device(device);
    }

    fn release_tensor(&self, tensor: TensorHandle) {
        let _ = self.plugin.release_tensor(tensor);
    }
}

impl core::fmt::Debug for DylibRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DylibRuntime")
            .field("id", &self.id)
            .field("path", &self.path)
            .finish()
    }
}

#[derive(Debug)]
struct RuntimeRegistry {
    next_runtime_id: AtomicU64,
    next_device_index: AtomicU32,
    runtimes: RwLock<HashMap<u64, Arc<DylibRuntime>>>,
    runtimes_by_path: RwLock<HashMap<String, u64>>,
    devices: RwLock<HashMap<u32, Arc<DeviceEntry>>>,
}

impl RuntimeRegistry {
    fn new() -> Self {
        Self {
            next_runtime_id: AtomicU64::new(1),
            next_device_index: AtomicU32::new(1),
            runtimes: RwLock::new(HashMap::new()),
            runtimes_by_path: RwLock::new(HashMap::new()),
            devices: RwLock::new(HashMap::new()),
        }
    }

    fn get_runtime(&self, runtime_id: u64) -> Result<Arc<DylibRuntime>, DylibError> {
        self.runtimes
            .read()
            .unwrap()
            .get(&runtime_id)
            .cloned()
            .ok_or(DylibError::RuntimeNotFound(runtime_id))
    }

    fn register_runtime(&self, path: impl AsRef<Path>) -> Result<u64, DylibError> {
        let key = normalize_path(path.as_ref());

        if let Some(id) = self.runtimes_by_path.read().unwrap().get(&key).copied() {
            return Ok(id);
        }

        let mut path_map = self.runtimes_by_path.write().unwrap();
        if let Some(id) = path_map.get(&key).copied() {
            return Ok(id);
        }

        let plugin = unsafe { LoadedBackendPlugin::load(path.as_ref()) }.map_err(map_load_error)?;
        let id = self.next_runtime_id.fetch_add(1, Ordering::Relaxed);

        let runtime = Arc::new(DylibRuntime {
            id,
            path: key.clone(),
            plugin: Arc::new(plugin),
        });

        self.runtimes.write().unwrap().insert(id, runtime);
        path_map.insert(key, id);

        Ok(id)
    }

    fn insert_device(&self, snapshot: DeviceSnapshot) -> u32 {
        let registry_index = self.next_device_index.fetch_add(1, Ordering::Relaxed);
        self.devices
            .write()
            .unwrap()
            .insert(registry_index, Arc::new(DeviceEntry::new(snapshot)));
        registry_index
    }

    fn device_entry(&self, registry_index: u32) -> Result<Arc<DeviceEntry>, DylibError> {
        self.devices
            .read()
            .unwrap()
            .get(&registry_index)
            .cloned()
            .ok_or(DylibError::DeviceNotFound(registry_index))
    }

    fn device_snapshot(&self, registry_index: u32) -> Result<DeviceSnapshot, DylibError> {
        Ok(self.device_entry(registry_index)?.snapshot())
    }

    fn retain_device(&self, registry_index: u32) -> Result<(), DylibError> {
        self.device_entry(registry_index)?.retain();
        Ok(())
    }

    fn release_device(&self, registry_index: u32) {
        let entry = match self.devices.read().unwrap().get(&registry_index).cloned() {
            Some(entry) => entry,
            None => return,
        };

        if !entry.release() {
            return;
        }

        self.devices.write().unwrap().remove(&registry_index);

        if let Ok(runtime) = self.get_runtime(entry.snapshot.runtime_id) {
            runtime.release_device(entry.snapshot.handle);
        }
    }
}

static REGISTRY: LazyLock<RuntimeRegistry> = LazyLock::new(RuntimeRegistry::new);

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

fn lookup_device(device: &DylibDevice) -> Result<(Arc<DylibRuntime>, DeviceSnapshot), DylibError> {
    let snapshot = REGISTRY.device_snapshot(device.registry_index)?;
    let runtime = REGISTRY.get_runtime(snapshot.runtime_id)?;
    Ok((runtime, snapshot))
}

pub(crate) fn to_execution_error(err: DylibError) -> ExecutionError {
    ExecutionError::WithContext {
        reason: format!("dylib dispatch error: {err}"),
    }
}

pub(crate) fn get_runtime(runtime_id: u64) -> Result<Arc<DylibRuntime>, DylibError> {
    REGISTRY.get_runtime(runtime_id)
}

pub(crate) fn register_runtime(path: impl AsRef<Path>) -> Result<u64, DylibError> {
    REGISTRY.register_runtime(path)
}

pub(crate) fn device_snapshot(registry_index: u32) -> Result<DeviceSnapshot, DylibError> {
    REGISTRY.device_snapshot(registry_index)
}

pub(crate) fn retain_device(registry_index: u32) -> Result<(), DylibError> {
    REGISTRY.retain_device(registry_index)
}

pub(crate) fn release_device(registry_index: u32) {
    REGISTRY.release_device(registry_index);
}

pub(crate) fn release_tensor(runtime_id: u64, handle: TensorHandle) {
    if let Ok(runtime) = get_runtime(runtime_id) {
        runtime.release_tensor(handle);
    }
}

pub(crate) fn create_device_from_runtime(
    runtime_id: u64,
    backend_type_id: u16,
    ordinal: usize,
) -> Result<DylibDevice, DylibError> {
    let ordinal_u32 = u32::try_from(ordinal)
        .map_err(|_| DylibError::InvalidInput(format!("Invalid device ordinal: {ordinal}")))?;

    let runtime = get_runtime(runtime_id)?;
    let available = runtime.device_count(backend_type_id);
    if available != 0 && ordinal >= available {
        return Err(DylibError::InvalidInput(format!(
            "Invalid device ordinal {ordinal} for backend type {backend_type_id}"
        )));
    }

    let handle = runtime.create_device_handle(backend_type_id, ordinal)?;
    let registry_index = REGISTRY.insert_device(DeviceSnapshot {
        runtime_id,
        backend_type_id,
        ordinal: ordinal_u32,
        handle,
    });

    Ok(DylibDevice::from_registry_index(registry_index))
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
    retain_device(index_id)?;
    Ok(DylibDevice::from_registry_index(index_id))
}

pub(crate) fn backend_name(device: &DylibDevice) -> String {
    match lookup_device(device).and_then(|(runtime, _)| runtime.name()) {
        Ok(name) => format!("dylib<{name}>"),
        Err(err) => format!("dylib<error:{err}>"),
    }
}

pub(crate) fn backend_seed(device: &DylibDevice, seed: u64) {
    let _ = lookup_device(device).and_then(|(runtime, _)| runtime.seed(seed));
}

pub(crate) fn backend_sync(device: &DylibDevice) -> Result<(), ExecutionError> {
    let (runtime, _) = lookup_device(device).map_err(to_execution_error)?;
    runtime.sync().map_err(to_execution_error)
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

    let (runtime, snapshot) = lookup_device(device)?;
    let handle = runtime.tensor_from_f32_data(snapshot.handle, &requested_shape, &values)?;

    Ok(DylibTensor::new(
        snapshot.runtime_id,
        device.clone(),
        handle,
        DType::F32,
        requested_shape,
    ))
}

pub(crate) fn tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let (values, shape) = runtime.tensor_into_f32_data(tensor.handle)?;

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

    if lhs.device != rhs.device {
        return Err(DylibError::InvalidInput(format!(
            "Cross-device operations are not supported (lhs={}, rhs={})",
            lhs.device.registry_index, rhs.device.registry_index
        )));
    }

    let runtime = get_runtime(lhs.runtime_id)?;
    let handle = runtime.tensor_binary(op, lhs.handle, rhs.handle)?;
    let shape = runtime.tensor_shape_or(handle, lhs.shape.clone());

    Ok(DylibTensor::new(
        lhs.runtime_id,
        lhs.device,
        handle,
        DType::F32,
        shape,
    ))
}

fn map_float_binary_op(op: TensorBinaryOp) -> DenseTensorBinaryOp {
    match op {
        TensorBinaryOp::Add => DenseTensorBinaryOp::Add,
        TensorBinaryOp::Matmul => DenseTensorBinaryOp::Matmul,
    }
}
