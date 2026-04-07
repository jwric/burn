use crate::loader::{LoadError, LoadedBackendPlugin, PluginCallError};
use crate::{DeviceHandle, TensorHandle};
use burn_backend::{
    BoolDType, DType, DTypeUsage, DTypeUsageSet, Distribution, ExecutionError, FloatDType,
    IntDType, Scalar, Shape, Slice, TensorData,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
        DeformConvOptions, InterpolateOptions, MaxPool1dBackward, MaxPool1dWithIndices,
        MaxPool2dBackward, MaxPool2dWithIndices, UnfoldOptions,
    },
    quantization::QuantScheme,
};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use super::device::DylibDevice;
use super::tensor::DylibTensor;

// Runtime naming conventions:
// - `backend_*` functions forward backend metadata/control operations.
// - `float_tensor_*` functions forward float tensor operations.
// - `resolve_*` helpers provide registry context used by both paths.

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

fn same_backend_device(lhs: DeviceSnapshot, rhs: DeviceSnapshot) -> bool {
    lhs.runtime_id == rhs.runtime_id
        && lhs.backend_type_id == rhs.backend_type_id
        && lhs.ordinal == rhs.ordinal
}

struct RuntimeRegistry {
    next_runtime_id: AtomicU64,
    next_device_index: AtomicU32,
    runtimes: RwLock<HashMap<u64, Arc<LoadedBackendPlugin>>>,
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

    fn get_runtime(&self, runtime_id: u64) -> Result<Arc<LoadedBackendPlugin>, DylibError> {
        self.runtimes
            .read()
            .expect("runtime lock")
            .get(&runtime_id)
            .cloned()
            .ok_or(DylibError::RuntimeNotFound(runtime_id))
    }

    fn register_runtime(&self, path: impl AsRef<Path>) -> Result<u64, DylibError> {
        let key = normalize_path(path.as_ref());

        if let Some(id) = self
            .runtimes_by_path
            .read()
            .expect("runtime path lock")
            .get(&key)
            .copied()
        {
            return Ok(id);
        }

        let mut path_map = self.runtimes_by_path.write().expect("runtime path lock");
        if let Some(id) = path_map.get(&key).copied() {
            return Ok(id);
        }

        let plugin =
            Arc::new(unsafe { LoadedBackendPlugin::load(path.as_ref()) }.map_err(map_load_error)?);
        let id = self.next_runtime_id.fetch_add(1, Ordering::Relaxed);
        self.runtimes
            .write()
            .expect("runtime lock")
            .insert(id, plugin);
        path_map.insert(key, id);

        Ok(id)
    }

    fn insert_device(&self, snapshot: DeviceSnapshot) -> u32 {
        if let Some(entry) =
            self.devices
                .read()
                .expect("device lock")
                .iter()
                .find_map(|(registry_index, entry)| {
                    same_backend_device(entry.snapshot(), snapshot)
                        .then_some((*registry_index, entry))
                })
        {
            entry.1.retain();

            if entry.1.snapshot().handle != snapshot.handle {
                if let Ok(runtime) = self.get_runtime(snapshot.runtime_id) {
                    let _ = runtime.release_device(snapshot.handle);
                }
            }

            return entry.0;
        }

        let mut devices = self.devices.write().expect("device lock");

        if let Some(entry) = devices.iter().find_map(|(registry_index, entry)| {
            same_backend_device(entry.snapshot(), snapshot).then_some((*registry_index, entry))
        }) {
            entry.1.retain();

            if entry.1.snapshot().handle != snapshot.handle {
                if let Ok(runtime) = self.get_runtime(snapshot.runtime_id) {
                    let _ = runtime.release_device(snapshot.handle);
                }
            }

            return entry.0;
        }

        let registry_index = self.next_device_index.fetch_add(1, Ordering::Relaxed);
        devices.insert(registry_index, Arc::new(DeviceEntry::new(snapshot)));
        registry_index
    }

    fn device_entry(&self, registry_index: u32) -> Result<Arc<DeviceEntry>, DylibError> {
        self.devices
            .read()
            .expect("device lock")
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
        let entry = match self
            .devices
            .read()
            .expect("device lock")
            .get(&registry_index)
            .cloned()
        {
            Some(entry) => entry,
            None => return,
        };

        if !entry.release() {
            return;
        }

        self.devices
            .write()
            .expect("device lock")
            .remove(&registry_index);

        if let Ok(runtime) = self.get_runtime(entry.snapshot.runtime_id) {
            let _ = runtime.release_device(entry.snapshot.handle);
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

fn default_plugin_path() -> Result<PathBuf, DylibError> {
    let exe_path = std::env::current_exe()
        .map_err(|err| DylibError::Load(format!("Failed to resolve current executable: {err}")))?;
    let exe_dir = exe_path.parent().ok_or_else(|| {
        DylibError::Load(format!(
            "Failed to resolve executable directory for {}",
            exe_path.display()
        ))
    })?;

    let ext = std::env::consts::DLL_EXTENSION;
    let prefix = std::env::consts::DLL_PREFIX;

    let name = "burn_dispatch";
    let candidate = exe_dir.join(format!("{}{}.{ext}", prefix, name));

    if candidate.exists() {
        Ok(candidate)
    } else {
        Err(DylibError::Load(format!(
            "Default plugin not found at expected path: {}",
            candidate.display()
        )))
    }
}

fn map_load_error(err: LoadError) -> DylibError {
    DylibError::Load(err.to_string())
}

fn map_call_error(err: PluginCallError) -> DylibError {
    DylibError::Plugin(err.to_string())
}

fn resolve_device_context(
    device: &DylibDevice,
) -> Result<(Arc<LoadedBackendPlugin>, DeviceSnapshot), DylibError> {
    let snapshot = REGISTRY.device_snapshot(device.registry_index)?;
    let runtime = REGISTRY.get_runtime(snapshot.runtime_id)?;
    Ok((runtime, snapshot))
}

fn dtype_to_float_dtype(dtype: DType) -> Option<FloatDType> {
    match dtype {
        DType::F64 => Some(FloatDType::F64),
        DType::F32 => Some(FloatDType::F32),
        DType::Flex32 => Some(FloatDType::Flex32),
        DType::F16 => Some(FloatDType::F16),
        DType::BF16 => Some(FloatDType::BF16),
        _ => None,
    }
}

fn dtype_to_int_dtype(dtype: DType) -> Option<IntDType> {
    match dtype {
        DType::I64 => Some(IntDType::I64),
        DType::I32 => Some(IntDType::I32),
        DType::I16 => Some(IntDType::I16),
        DType::I8 => Some(IntDType::I8),
        DType::U64 => Some(IntDType::U64),
        DType::U32 => Some(IntDType::U32),
        DType::U16 => Some(IntDType::U16),
        DType::U8 => Some(IntDType::U8),
        _ => None,
    }
}

fn dtype_to_bool_dtype(dtype: DType) -> Option<BoolDType> {
    match dtype {
        DType::Bool(dtype) => Some(dtype),
        _ => None,
    }
}

fn int_data_to_u64(data: TensorData) -> Result<Vec<u64>, DylibError> {
    match data.dtype {
        DType::I64 => data
            .into_vec::<i64>()
            .map(|values| values.into_iter().map(|v| v as u64).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::I32 => data
            .into_vec::<i32>()
            .map(|values| values.into_iter().map(|v| v as u64).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::I16 => data
            .into_vec::<i16>()
            .map(|values| values.into_iter().map(|v| v as u64).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::I8 => data
            .into_vec::<i8>()
            .map(|values| values.into_iter().map(|v| v as u64).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::U64 => data
            .into_vec::<u64>()
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::U32 => data
            .into_vec::<u32>()
            .map(|values| values.into_iter().map(u64::from).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::U16 => data
            .into_vec::<u16>()
            .map(|values| values.into_iter().map(u64::from).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        DType::U8 => data
            .into_vec::<u8>()
            .map(|values| values.into_iter().map(u64::from).collect())
            .map_err(|err| DylibError::Data(err.to_string())),
        other => Err(DylibError::Data(format!(
            "Expected int tensor data, got {other:?}"
        ))),
    }
}

fn tensor_data_from_u64(values: Vec<u64>, shape: Shape, dtype: IntDType) -> TensorData {
    match dtype {
        IntDType::I64 => TensorData::new(
            values.into_iter().map(|v| v as i64).collect::<Vec<i64>>(),
            shape,
        ),
        IntDType::I32 => TensorData::new(
            values.into_iter().map(|v| v as i32).collect::<Vec<i32>>(),
            shape,
        ),
        IntDType::I16 => TensorData::new(
            values.into_iter().map(|v| v as i16).collect::<Vec<i16>>(),
            shape,
        ),
        IntDType::I8 => TensorData::new(
            values.into_iter().map(|v| v as i8).collect::<Vec<i8>>(),
            shape,
        ),
        IntDType::U64 => TensorData::new(values, shape),
        IntDType::U32 => TensorData::new(
            values.into_iter().map(|v| v as u32).collect::<Vec<u32>>(),
            shape,
        ),
        IntDType::U16 => TensorData::new(
            values.into_iter().map(|v| v as u16).collect::<Vec<u16>>(),
            shape,
        ),
        IntDType::U8 => TensorData::new(
            values.into_iter().map(|v| v as u8).collect::<Vec<u8>>(),
            shape,
        ),
    }
}

fn bool_data_to_u8(data: TensorData) -> Result<Vec<u8>, DylibError> {
    data.convert::<u8>()
        .into_vec::<u8>()
        .map_err(|err| DylibError::Data(err.to_string()))
}

fn shape_from_runtime(
    runtime: &LoadedBackendPlugin,
    handle: TensorHandle,
    fallback: &Shape,
) -> Shape {
    runtime
        .float_tensor_shape(handle)
        .map(|dims| Shape::new_raw(dims.into()))
        .unwrap_or_else(|_| fallback.clone())
}

fn tensor_from_output(
    runtime: &LoadedBackendPlugin,
    source: &DylibTensor,
    handle: TensorHandle,
    dtype: DType,
) -> DylibTensor {
    let shape = shape_from_runtime(runtime, handle, &source.shape);
    DylibTensor::new(
        source.runtime_id,
        source.device.clone(),
        handle,
        dtype,
        shape,
    )
}

fn tensor_from_output_with_fallback(
    runtime: &LoadedBackendPlugin,
    runtime_id: u64,
    device: &DylibDevice,
    fallback_shape: &Shape,
    handle: TensorHandle,
    dtype: DType,
) -> DylibTensor {
    let shape = shape_from_runtime(runtime, handle, fallback_shape);
    DylibTensor::new(runtime_id, device.clone(), handle, dtype, shape)
}

fn ensure_same_runtime_and_device(
    op_name: &str,
    tensors: &[&DylibTensor],
) -> Result<(), DylibError> {
    let Some(first) = tensors.first() else {
        return Ok(());
    };

    for tensor in tensors.iter().skip(1) {
        if tensor.runtime_id != first.runtime_id {
            return Err(DylibError::InvalidInput(format!(
                "Cross-runtime operations are not supported for {op_name} (first={}, other={})",
                first.runtime_id, tensor.runtime_id
            )));
        }

        if tensor.device != first.device {
            return Err(DylibError::InvalidInput(format!(
                "Cross-device operations are not supported for {op_name} (first={}, other={})",
                first.device.registry_index, tensor.device.registry_index
            )));
        }
    }

    Ok(())
}

fn forward_unary_op(
    tensor: DylibTensor,
    output_dtype: DType,
    call: impl FnOnce(&LoadedBackendPlugin, TensorHandle) -> Result<TensorHandle, PluginCallError>,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let handle = call(&runtime, tensor.handle).map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &tensor, handle, output_dtype))
}

fn forward_binary_op(
    lhs: DylibTensor,
    rhs: DylibTensor,
    output_dtype: DType,
    op_name: &str,
    call: impl FnOnce(
        &LoadedBackendPlugin,
        TensorHandle,
        TensorHandle,
    ) -> Result<TensorHandle, PluginCallError>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(op_name, &[&lhs, &rhs])?;
    let runtime = get_runtime(lhs.runtime_id)?;
    let handle = call(&runtime, lhs.handle, rhs.handle).map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &lhs, handle, output_dtype))
}

fn forward_ternary_op(
    a: DylibTensor,
    b: DylibTensor,
    c: DylibTensor,
    output_dtype: DType,
    op_name: &str,
    call: impl FnOnce(
        &LoadedBackendPlugin,
        TensorHandle,
        TensorHandle,
        TensorHandle,
    ) -> Result<TensorHandle, PluginCallError>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(op_name, &[&a, &b, &c])?;
    let runtime = get_runtime(a.runtime_id)?;
    let handle = call(&runtime, a.handle, b.handle, c.handle).map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &a, handle, output_dtype))
}

pub(crate) fn to_execution_error(err: DylibError) -> ExecutionError {
    ExecutionError::WithContext {
        reason: format!("dylib dispatch error: {err}"),
    }
}

pub(crate) fn get_runtime(runtime_id: u64) -> Result<Arc<LoadedBackendPlugin>, DylibError> {
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
        let _ = runtime.release_tensor(handle);
    }
}

fn create_device_from_runtime(
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

    let handle = runtime
        .create_device(backend_type_id, ordinal)
        .map_err(map_call_error)?;
    let registry_index = REGISTRY.insert_device(DeviceSnapshot {
        runtime_id,
        backend_type_id,
        ordinal: ordinal_u32,
        handle,
    });

    Ok(DylibDevice::from_registry_index(registry_index))
}

/// Creates a device using the plugin's default device.
pub fn create_default_device() -> Result<DylibDevice, DylibError> {
    let path = default_plugin_path()?;
    let runtime_id = register_runtime(&path)?;
    let runtime = get_runtime(runtime_id)?;
    let (type_id, ordinal, device) = runtime.create_default_device().map_err(map_call_error)?;
    let ordinal_u32 = u32::try_from(ordinal).map_err(|_| {
        DylibError::InvalidInput(format!(
            "Invalid device ordinal from plugin default device: {ordinal}"
        ))
    })?;

    let registry_index = REGISTRY.insert_device(DeviceSnapshot {
        runtime_id,
        backend_type_id: type_id,
        ordinal: ordinal_u32,
        handle: device,
    });

    Ok(DylibDevice::from_registry_index(registry_index))
}

/// Creates a device from a shared library path, backend type id, and device ordinal.
pub fn create_device_from_path(
    path: impl AsRef<Path>,
    backend_type_id: u16,
    ordinal: usize,
) -> Result<DylibDevice, DylibError> {
    let runtime_id = register_runtime(path)?;
    create_device_from_runtime(runtime_id, backend_type_id, ordinal)
}

/// Creates a device from a registry index previously returned by `device_from_registry`.
pub fn device_from_registry(index_id: u32) -> Result<DylibDevice, DylibError> {
    retain_device(index_id)?;
    Ok(DylibDevice::from_registry_index(index_id))
}

pub(crate) fn backend_name(device: &DylibDevice) -> String {
    match resolve_device_context(device)
        .and_then(|(runtime, _)| runtime.backend_name().map_err(map_call_error))
    {
        Ok(name) => format!("dylib<{name}>"),
        Err(err) => format!("dylib<error:{err}>"),
    }
}

pub(crate) fn backend_seed(device: &DylibDevice, seed: u64) {
    let _ = resolve_device_context(device)
        .and_then(|(runtime, _)| runtime.backend_seed(seed).map_err(map_call_error));
}

pub(crate) fn backend_sync(device: &DylibDevice) -> Result<(), ExecutionError> {
    let (runtime, _) = resolve_device_context(device).map_err(to_execution_error)?;
    runtime
        .backend_sync()
        .map_err(map_call_error)
        .map_err(to_execution_error)
}

pub(crate) fn dtype_usage(dtype: DType) -> DTypeUsageSet {
    match dtype {
        DType::F64
        | DType::F32
        | DType::Flex32
        | DType::F16
        | DType::BF16
        | DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8
        | DType::Bool(_)
        | DType::QFloat(_) => DTypeUsage::general(),
    }
}

pub(crate) fn q_tensor_from_data(
    data: TensorData,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    let scheme = match data.dtype {
        DType::QFloat(scheme) => scheme,
        dtype => {
            return Err(DylibError::InvalidInput(format!(
                "Expected quantized dtype for q_tensor_from_data, got {dtype:?}"
            )));
        }
    };

    let shape = data.shape.clone();
    let bytes = data.into_bytes().to_vec();

    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .q_tensor_from_u8_data(snapshot.handle, shape.as_slice(), &bytes, scheme)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        DType::QFloat(scheme),
    ))
}

pub(crate) fn q_tensor_quantize(
    tensor: DylibTensor,
    scheme: QuantScheme,
    scales: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("q_tensor_quantize", &[&tensor, &scales])?;

    let runtime = get_runtime(tensor.runtime_id)?;
    let handle = runtime
        .q_tensor_quantize(tensor.handle, scheme, scales.handle)
        .map_err(map_call_error)?;

    Ok(tensor_from_output(
        &runtime,
        &tensor,
        handle,
        DType::QFloat(scheme),
    ))
}

pub(crate) fn q_tensor_dequantize(
    tensor: DylibTensor,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let handle = runtime
        .q_tensor_dequantize(tensor.handle, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output(&runtime, &tensor, handle, dtype.into()))
}

pub(crate) fn q_tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let (bytes, scheme) = runtime
        .q_tensor_into_u8_data(tensor.handle)
        .map_err(map_call_error)?;
    let shape = shape_from_runtime(&runtime, tensor.handle, &tensor.shape);

    Ok(TensorData::from_bytes_vec(
        bytes,
        shape,
        DType::QFloat(scheme),
    ))
}

pub(crate) fn q_tensor_to_device(
    tensor: DylibTensor,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    if tensor.device == *device {
        return Ok(tensor);
    }

    let (target_runtime, target_snapshot) = resolve_device_context(device)?;

    if tensor.runtime_id == target_snapshot.runtime_id {
        let handle = target_runtime
            .q_tensor_to_device(tensor.handle, target_snapshot.handle)
            .map_err(map_call_error)?;
        return Ok(tensor_from_output_with_fallback(
            &target_runtime,
            tensor.runtime_id,
            device,
            &tensor.shape,
            handle,
            tensor.dtype,
        ));
    }

    let data = q_tensor_into_data(tensor)?;
    q_tensor_from_data(data, device)
}

pub(crate) fn q_tensor_reshape(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_reshape(handle, shape.as_slice())
    })
}

pub(crate) fn q_tensor_expand(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_expand(handle, shape.as_slice())
    })
}

pub(crate) fn q_tensor_swap_dims(
    tensor: DylibTensor,
    dim1: usize,
    dim2: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_swap_dims(handle, dim1, dim2)
    })
}

pub(crate) fn q_tensor_permute(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_permute(handle, axes)
    })
}

pub(crate) fn q_tensor_flip(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_flip(handle, axes)
    })
}

pub(crate) fn q_tensor_select(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "q_tensor_select",
        |runtime, tensor_handle, indices_handle| {
            runtime.q_tensor_select(tensor_handle, dim, indices_handle)
        },
    )
}

pub(crate) fn q_tensor_slice(
    tensor: DylibTensor,
    slices: &[Slice],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.q_tensor_slice(handle, slices)
    })
}

pub(crate) fn float_tensor_from_data(
    data: TensorData,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    let requested_dtype = dtype_to_float_dtype(data.dtype).unwrap_or(FloatDType::F32);

    let data_f32 = data.convert::<f32>();
    let requested_shape = data_f32.shape.clone();
    let values = data_f32
        .into_vec::<f32>()
        .map_err(|err| DylibError::Data(err.to_string()))?;

    let (runtime, snapshot) = resolve_device_context(device)?;
    let mut handle = runtime
        .float_tensor_from_f32_data(snapshot.handle, requested_shape.as_slice(), &values)
        .map_err(map_call_error)?;

    if requested_dtype != FloatDType::F32 {
        handle = runtime
            .float_tensor_cast(handle, requested_dtype)
            .map_err(map_call_error)?;
    }

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &requested_shape,
        handle,
        requested_dtype.into(),
    ))
}

pub(crate) fn float_tensor_random(
    shape: Shape,
    distribution: Distribution,
    device: &DylibDevice,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .float_tensor_random(snapshot.handle, shape.as_slice(), distribution, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn float_tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let values = runtime
        .float_tensor_into_f32_data(tensor.handle)
        .map_err(map_call_error)?;
    let shape = shape_from_runtime(&runtime, tensor.handle, &tensor.shape);

    Ok(TensorData::new(values, shape).convert_dtype(tensor.dtype))
}

pub(crate) fn float_tensor_to_device(
    tensor: DylibTensor,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    if tensor.device == *device {
        return Ok(tensor);
    }

    let (target_runtime, target_snapshot) = resolve_device_context(device)?;

    if tensor.runtime_id == target_snapshot.runtime_id {
        let handle = target_runtime
            .float_tensor_to_device(tensor.handle, target_snapshot.handle)
            .map_err(map_call_error)?;
        return Ok(tensor_from_output_with_fallback(
            &target_runtime,
            tensor.runtime_id,
            device,
            &tensor.shape,
            handle,
            tensor.dtype,
        ));
    }

    let data = float_tensor_into_data(tensor)?;
    float_tensor_from_data(data, device)
}

pub(crate) fn float_tensor_into_int(
    tensor: DylibTensor,
    out_dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_into_int(handle, out_dtype)
    })
}

pub(crate) fn float_tensor_empty(
    shape: Shape,
    device: &DylibDevice,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .float_tensor_empty(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn float_tensor_zeros(
    shape: Shape,
    device: &DylibDevice,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .float_tensor_zeros(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn float_tensor_ones(
    shape: Shape,
    device: &DylibDevice,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .float_tensor_ones(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn float_tensor_full(
    shape: Shape,
    fill_value: Scalar,
    device: &DylibDevice,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .float_tensor_full(snapshot.handle, shape.as_slice(), fill_value, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

macro_rules! define_binary_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            lhs: DylibTensor,
            rhs: DylibTensor,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype = lhs.dtype;
            forward_binary_op(
                lhs,
                rhs,
                output_dtype,
                stringify!($fn_name),
                |runtime, lhs_handle, rhs_handle| runtime.$loader_name(lhs_handle, rhs_handle),
            )
        }
    };
}

macro_rules! define_scalar_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(lhs: DylibTensor, rhs: Scalar) -> Result<DylibTensor, DylibError> {
            let output_dtype = lhs.dtype;
            forward_unary_op(lhs, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, rhs)
            })
        }
    };
}

macro_rules! define_unary_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(tensor: DylibTensor) -> Result<DylibTensor, DylibError> {
            let output_dtype = tensor.dtype;
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle)
            })
        }
    };
}

macro_rules! define_unary_dim_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(tensor: DylibTensor, dim: usize) -> Result<DylibTensor, DylibError> {
            let output_dtype = tensor.dtype;
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim)
            })
        }
    };
}

macro_rules! define_compare_binary {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            lhs: DylibTensor,
            rhs: DylibTensor,
            out_dtype: BoolDType,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype: DType = out_dtype.into();
            forward_binary_op(
                lhs,
                rhs,
                output_dtype,
                stringify!($fn_name),
                |runtime, lhs_handle, rhs_handle| {
                    runtime.$loader_name(lhs_handle, rhs_handle, out_dtype)
                },
            )
        }
    };
}

macro_rules! define_compare_scalar {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            lhs: DylibTensor,
            rhs: Scalar,
            out_dtype: BoolDType,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype: DType = out_dtype.into();
            forward_unary_op(lhs, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, rhs, out_dtype)
            })
        }
    };
}

macro_rules! define_repeat_dim_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            times: usize,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype = tensor.dtype;
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim, times)
            })
        }
    };
}

macro_rules! define_clamp_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            min: Scalar,
            max: Scalar,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype = tensor.dtype;
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, min, max)
            })
        }
    };
}

macro_rules! define_bool_reduce {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            out_dtype: BoolDType,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype: DType = out_dtype.into();
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, out_dtype)
            })
        }
    };
}

macro_rules! define_bool_reduce_dim {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            out_dtype: BoolDType,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype: DType = out_dtype.into();
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim, out_dtype)
            })
        }
    };
}

macro_rules! define_sort_same_dtype {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            descending: bool,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype = tensor.dtype;
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim, descending)
            })
        }
    };
}

macro_rules! define_with_indices {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            out_dtype: IntDType,
        ) -> Result<(DylibTensor, DylibTensor), DylibError> {
            let runtime = get_runtime(tensor.runtime_id)?;
            let handles = runtime
                .$loader_name(tensor.handle, dim, out_dtype)
                .map_err(map_call_error)?;
            let values = tensor_from_output(&runtime, &tensor, handles.values, tensor.dtype);
            let indices = tensor_from_output(&runtime, &tensor, handles.indices, out_dtype.into());
            Ok((values, indices))
        }
    };
}

macro_rules! define_with_indices_no_dtype {
    ($fn_name:ident, $loader_name:ident, $indices_dtype:expr) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
        ) -> Result<(DylibTensor, DylibTensor), DylibError> {
            let runtime = get_runtime(tensor.runtime_id)?;
            let handles = runtime
                .$loader_name(tensor.handle, dim)
                .map_err(map_call_error)?;
            let values = tensor_from_output(&runtime, &tensor, handles.values, tensor.dtype);
            let indices = tensor_from_output(&runtime, &tensor, handles.indices, $indices_dtype);
            Ok((values, indices))
        }
    };
}

macro_rules! define_sort_with_indices {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            descending: bool,
            out_dtype: IntDType,
        ) -> Result<(DylibTensor, DylibTensor), DylibError> {
            let runtime = get_runtime(tensor.runtime_id)?;
            let handles = runtime
                .$loader_name(tensor.handle, dim, descending, out_dtype)
                .map_err(map_call_error)?;
            let values = tensor_from_output(&runtime, &tensor, handles.values, tensor.dtype);
            let indices = tensor_from_output(&runtime, &tensor, handles.indices, out_dtype.into());
            Ok((values, indices))
        }
    };
}

macro_rules! define_sort_with_indices_no_dtype {
    ($fn_name:ident, $loader_name:ident, $indices_dtype:expr) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            descending: bool,
        ) -> Result<(DylibTensor, DylibTensor), DylibError> {
            let runtime = get_runtime(tensor.runtime_id)?;
            let handles = runtime
                .$loader_name(tensor.handle, dim, descending)
                .map_err(map_call_error)?;
            let values = tensor_from_output(&runtime, &tensor, handles.values, tensor.dtype);
            let indices = tensor_from_output(&runtime, &tensor, handles.indices, $indices_dtype);
            Ok((values, indices))
        }
    };
}

macro_rules! define_argsort {
    ($fn_name:ident, $loader_name:ident) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            descending: bool,
            out_dtype: IntDType,
        ) -> Result<DylibTensor, DylibError> {
            let output_dtype: DType = out_dtype.into();
            forward_unary_op(tensor, output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim, descending, out_dtype)
            })
        }
    };
}

macro_rules! define_argsort_no_dtype {
    ($fn_name:ident, $loader_name:ident, $output_dtype:expr) => {
        pub(crate) fn $fn_name(
            tensor: DylibTensor,
            dim: usize,
            descending: bool,
        ) -> Result<DylibTensor, DylibError> {
            forward_unary_op(tensor, $output_dtype, |runtime, handle| {
                runtime.$loader_name(handle, dim, descending)
            })
        }
    };
}

define_binary_same_dtype!(float_tensor_add, float_tensor_add);
define_scalar_same_dtype!(float_tensor_add_scalar, float_tensor_add_scalar);
define_binary_same_dtype!(float_tensor_sub, float_tensor_sub);
define_scalar_same_dtype!(float_tensor_sub_scalar, float_tensor_sub_scalar);
define_binary_same_dtype!(float_tensor_mul, float_tensor_mul);
define_scalar_same_dtype!(float_tensor_mul_scalar, float_tensor_mul_scalar);
define_binary_same_dtype!(float_tensor_div, float_tensor_div);
define_scalar_same_dtype!(float_tensor_div_scalar, float_tensor_div_scalar);
define_binary_same_dtype!(float_tensor_remainder, float_tensor_remainder);
define_scalar_same_dtype!(float_tensor_remainder_scalar, float_tensor_remainder_scalar);
define_binary_same_dtype!(float_tensor_matmul, float_tensor_matmul);
define_unary_same_dtype!(float_tensor_recip, float_tensor_recip);
define_unary_same_dtype!(float_tensor_sum, float_tensor_sum);
define_unary_dim_same_dtype!(float_tensor_sum_dim, float_tensor_sum_dim);
define_unary_same_dtype!(float_tensor_prod, float_tensor_prod);
define_unary_dim_same_dtype!(float_tensor_prod_dim, float_tensor_prod_dim);
define_unary_dim_same_dtype!(float_tensor_mean_dim, float_tensor_mean_dim);
define_unary_dim_same_dtype!(float_tensor_cumsum, float_tensor_cumsum);
define_unary_dim_same_dtype!(float_tensor_cumprod, float_tensor_cumprod);
define_unary_dim_same_dtype!(float_tensor_cummin, float_tensor_cummin);
define_unary_dim_same_dtype!(float_tensor_cummax, float_tensor_cummax);
define_unary_same_dtype!(float_tensor_exp, float_tensor_exp);
define_unary_same_dtype!(float_tensor_log, float_tensor_log);
define_unary_same_dtype!(float_tensor_log1p, float_tensor_log1p);
define_binary_same_dtype!(float_tensor_powf, float_tensor_powf);
define_scalar_same_dtype!(float_tensor_powf_scalar, float_tensor_powf_scalar);
define_unary_same_dtype!(float_tensor_sqrt, float_tensor_sqrt);
define_unary_same_dtype!(float_tensor_abs, float_tensor_abs);
define_unary_same_dtype!(float_tensor_cos, float_tensor_cos);
define_unary_same_dtype!(float_tensor_sin, float_tensor_sin);
define_unary_same_dtype!(float_tensor_tan, float_tensor_tan);
define_unary_same_dtype!(float_tensor_cosh, float_tensor_cosh);
define_unary_same_dtype!(float_tensor_sinh, float_tensor_sinh);
define_unary_same_dtype!(float_tensor_tanh, float_tensor_tanh);
define_unary_same_dtype!(float_tensor_acos, float_tensor_acos);
define_unary_same_dtype!(float_tensor_acosh, float_tensor_acosh);
define_unary_same_dtype!(float_tensor_asin, float_tensor_asin);
define_unary_same_dtype!(float_tensor_asinh, float_tensor_asinh);
define_unary_same_dtype!(float_tensor_atan, float_tensor_atan);
define_unary_same_dtype!(float_tensor_atanh, float_tensor_atanh);
define_binary_same_dtype!(float_tensor_atan2, float_tensor_atan2);
define_unary_same_dtype!(float_tensor_round, float_tensor_round);
define_unary_same_dtype!(float_tensor_floor, float_tensor_floor);
define_unary_same_dtype!(float_tensor_ceil, float_tensor_ceil);
define_unary_same_dtype!(float_tensor_trunc, float_tensor_trunc);
define_unary_same_dtype!(float_tensor_erf, float_tensor_erf);
define_compare_binary!(float_tensor_equal, float_tensor_equal);
define_compare_scalar!(float_tensor_equal_elem, float_tensor_equal_elem);
define_compare_binary!(float_tensor_greater, float_tensor_greater);
define_compare_scalar!(float_tensor_greater_elem, float_tensor_greater_elem);
define_compare_binary!(float_tensor_greater_equal, float_tensor_greater_equal);
define_compare_scalar!(
    float_tensor_greater_equal_elem,
    float_tensor_greater_equal_elem
);
define_compare_binary!(float_tensor_lower, float_tensor_lower);
define_compare_scalar!(float_tensor_lower_elem, float_tensor_lower_elem);
define_compare_binary!(float_tensor_lower_equal, float_tensor_lower_equal);
define_compare_scalar!(float_tensor_lower_equal_elem, float_tensor_lower_equal_elem);
define_repeat_dim_same_dtype!(float_tensor_repeat_dim, float_tensor_repeat_dim);
define_scalar_same_dtype!(float_tensor_clamp_min, float_tensor_clamp_min);
define_scalar_same_dtype!(float_tensor_clamp_max, float_tensor_clamp_max);
define_clamp_same_dtype!(float_tensor_clamp, float_tensor_clamp);
define_unary_same_dtype!(float_tensor_neg, float_tensor_neg);
define_unary_same_dtype!(float_tensor_transpose, float_tensor_transpose);
define_compare_binary!(float_tensor_not_equal, float_tensor_not_equal);
define_compare_scalar!(float_tensor_not_equal_elem, float_tensor_not_equal_elem);
define_unary_same_dtype!(float_tensor_mean, float_tensor_mean);
define_scalar_same_dtype!(float_tensor_powi_scalar, float_tensor_powi_scalar);
define_unary_same_dtype!(float_tensor_max, float_tensor_max);
define_unary_dim_same_dtype!(float_tensor_max_dim, float_tensor_max_dim);
define_with_indices!(
    float_tensor_max_dim_with_indices,
    float_tensor_max_dim_with_indices
);
define_unary_same_dtype!(float_tensor_min, float_tensor_min);
define_unary_dim_same_dtype!(float_tensor_min_dim, float_tensor_min_dim);
define_with_indices!(
    float_tensor_min_dim_with_indices,
    float_tensor_min_dim_with_indices
);
define_unary_same_dtype!(float_tensor_max_abs, float_tensor_max_abs);
define_unary_dim_same_dtype!(float_tensor_max_abs_dim, float_tensor_max_abs_dim);
define_bool_reduce!(float_tensor_any, float_tensor_any);
define_bool_reduce_dim!(float_tensor_any_dim, float_tensor_any_dim);
define_bool_reduce!(float_tensor_all, float_tensor_all);
define_bool_reduce_dim!(float_tensor_all_dim, float_tensor_all_dim);
define_unary_same_dtype!(float_tensor_sign, float_tensor_sign);
define_sort_same_dtype!(float_tensor_sort, float_tensor_sort);
define_sort_with_indices!(
    float_tensor_sort_with_indices,
    float_tensor_sort_with_indices
);
define_argsort!(float_tensor_argsort, float_tensor_argsort);
define_bool_reduce!(float_tensor_is_nan, float_tensor_is_nan);
define_bool_reduce!(float_tensor_is_inf, float_tensor_is_inf);

pub(crate) fn float_tensor_powi(
    lhs: DylibTensor,
    rhs: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = lhs.dtype;
    forward_binary_op(
        lhs,
        rhs,
        output_dtype,
        "float_tensor_powi",
        |runtime, lhs_handle, rhs_handle| runtime.float_tensor_powi(lhs_handle, rhs_handle),
    )
}

pub(crate) fn float_tensor_cat(
    tensors: Vec<DylibTensor>,
    dim: usize,
) -> Result<DylibTensor, DylibError> {
    let Some(first) = tensors.first().cloned() else {
        return Err(DylibError::InvalidInput(
            "float_tensor_cat requires at least one tensor".into(),
        ));
    };

    let refs = tensors.iter().collect::<Vec<_>>();
    ensure_same_runtime_and_device("float_tensor_cat", &refs)?;

    let runtime = get_runtime(first.runtime_id)?;
    let handles = tensors
        .iter()
        .map(|tensor| tensor.handle)
        .collect::<Vec<_>>();
    let handle = runtime
        .float_tensor_cat(&handles, dim)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &first, handle, first.dtype))
}

pub(crate) fn float_tensor_cross(
    lhs: DylibTensor,
    rhs: DylibTensor,
    dim: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = lhs.dtype;
    forward_binary_op(
        lhs,
        rhs,
        output_dtype,
        "float_tensor_cross",
        |runtime, lhs_handle, rhs_handle| runtime.float_tensor_cross(lhs_handle, rhs_handle, dim),
    )
}

pub(crate) fn float_tensor_swap_dims(
    tensor: DylibTensor,
    dim1: usize,
    dim2: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_swap_dims(handle, dim1, dim2)
    })
}

pub(crate) fn float_tensor_permute(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_permute(handle, axes)
    })
}

pub(crate) fn float_tensor_flip(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_flip(handle, axes)
    })
}

pub(crate) fn float_tensor_reshape(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_reshape(handle, shape.as_slice())
    })
}

pub(crate) fn float_tensor_gather(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "float_tensor_gather",
        |runtime, tensor_handle, indices_handle| {
            runtime.float_tensor_gather(dim, tensor_handle, indices_handle)
        },
    )
}

pub(crate) fn float_tensor_scatter_add(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "float_tensor_scatter_add",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.float_tensor_scatter_add(dim, tensor_handle, indices_handle, value_handle)
        },
    )
}

pub(crate) fn float_tensor_select(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "float_tensor_select",
        |runtime, tensor_handle, indices_handle| {
            runtime.float_tensor_select(tensor_handle, dim, indices_handle)
        },
    )
}

pub(crate) fn float_tensor_select_add(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "float_tensor_select_add",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.float_tensor_select_add(tensor_handle, dim, indices_handle, value_handle)
        },
    )
}

pub(crate) fn float_tensor_slice(
    tensor: DylibTensor,
    slices: &[Slice],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_slice(handle, slices)
    })
}

pub(crate) fn float_tensor_slice_assign(
    tensor: DylibTensor,
    slices: &[Slice],
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        value,
        output_dtype,
        "float_tensor_slice_assign",
        |runtime, tensor_handle, value_handle| {
            runtime.float_tensor_slice_assign(tensor_handle, slices, value_handle)
        },
    )
}

pub(crate) fn float_tensor_mask_where(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        mask,
        value,
        output_dtype,
        "float_tensor_mask_where",
        |runtime, tensor_handle, mask_handle, value_handle| {
            runtime.float_tensor_mask_where(tensor_handle, mask_handle, value_handle)
        },
    )
}

pub(crate) fn float_tensor_mask_fill(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: Scalar,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        mask,
        output_dtype,
        "float_tensor_mask_fill",
        |runtime, tensor_handle, mask_handle| {
            runtime.float_tensor_mask_fill(tensor_handle, mask_handle, value)
        },
    )
}

pub(crate) fn float_tensor_cast(
    tensor: DylibTensor,
    dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_cast(handle, dtype)
    })
}

pub(crate) fn float_tensor_argmax(
    tensor: DylibTensor,
    dim: usize,
    out_dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_argmax(handle, dim, out_dtype)
    })
}

pub(crate) fn float_tensor_argmin(
    tensor: DylibTensor,
    dim: usize,
    out_dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_argmin(handle, dim, out_dtype)
    })
}

pub(crate) fn float_tensor_expand(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_expand(handle, shape.as_slice())
    })
}

pub(crate) fn float_tensor_unfold(
    tensor: DylibTensor,
    dim: usize,
    size: usize,
    step: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.float_tensor_unfold(handle, dim, size, step)
    })
}

pub(crate) fn int_tensor_from_data(
    data: TensorData,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    let requested_dtype = dtype_to_int_dtype(data.dtype).unwrap_or(IntDType::I32);
    let data = data.convert_dtype(requested_dtype.into());
    let requested_shape = data.shape.clone();
    let values = int_data_to_u64(data)?;

    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_from_u64_data(
            snapshot.handle,
            requested_shape.as_slice(),
            &values,
            requested_dtype,
        )
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &requested_shape,
        handle,
        requested_dtype.into(),
    ))
}

pub(crate) fn int_tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let values = runtime
        .int_tensor_into_u64_data(tensor.handle)
        .map_err(map_call_error)?;
    let shape = shape_from_runtime(&runtime, tensor.handle, &tensor.shape);
    let dtype = dtype_to_int_dtype(tensor.dtype).unwrap_or(IntDType::I32);

    Ok(tensor_data_from_u64(values, shape, dtype).convert_dtype(tensor.dtype))
}

pub(crate) fn int_tensor_to_device(
    tensor: DylibTensor,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    if tensor.device == *device {
        return Ok(tensor);
    }

    let (target_runtime, target_snapshot) = resolve_device_context(device)?;

    if tensor.runtime_id == target_snapshot.runtime_id {
        let handle = target_runtime
            .int_tensor_to_device(tensor.handle, target_snapshot.handle)
            .map_err(map_call_error)?;
        return Ok(tensor_from_output_with_fallback(
            &target_runtime,
            tensor.runtime_id,
            device,
            &tensor.shape,
            handle,
            tensor.dtype,
        ));
    }

    let data = int_tensor_into_data(tensor)?;
    int_tensor_from_data(data, device)
}

pub(crate) fn int_tensor_into_float(
    tensor: DylibTensor,
    out_dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_into_float(handle, out_dtype)
    })
}

pub(crate) fn int_tensor_empty(
    shape: Shape,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_empty(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_zeros(
    shape: Shape,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_zeros(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_ones(
    shape: Shape,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_ones(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_full(
    shape: Shape,
    fill_value: Scalar,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_full(snapshot.handle, shape.as_slice(), fill_value, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_random(
    shape: Shape,
    distribution: Distribution,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_random(snapshot.handle, shape.as_slice(), distribution, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_arange_step(
    range: core::ops::Range<i64>,
    step: usize,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let fallback_len = if range.end <= range.start {
        0
    } else {
        let distance = (range.end - range.start) as usize;
        (distance + step.saturating_sub(1)) / step.max(1)
    };
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_arange_step(range.start, range.end, step, snapshot.handle, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &Shape::new([fallback_len]),
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_arange(
    range: core::ops::Range<i64>,
    device: &DylibDevice,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let fallback_len = if range.end <= range.start {
        0
    } else {
        (range.end - range.start) as usize
    };
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .int_tensor_arange(range.start, range.end, snapshot.handle, dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &Shape::new([fallback_len]),
        handle,
        dtype.into(),
    ))
}

pub(crate) fn int_tensor_cast(
    tensor: DylibTensor,
    dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_cast(handle, dtype)
    })
}

define_binary_same_dtype!(int_tensor_add, int_tensor_add);
define_scalar_same_dtype!(int_tensor_add_scalar, int_tensor_add_scalar);
define_binary_same_dtype!(int_tensor_sub, int_tensor_sub);
define_scalar_same_dtype!(int_tensor_sub_scalar, int_tensor_sub_scalar);
define_binary_same_dtype!(int_tensor_mul, int_tensor_mul);
define_scalar_same_dtype!(int_tensor_mul_scalar, int_tensor_mul_scalar);
define_binary_same_dtype!(int_tensor_div, int_tensor_div);
define_scalar_same_dtype!(int_tensor_div_scalar, int_tensor_div_scalar);
define_binary_same_dtype!(int_tensor_remainder, int_tensor_remainder);
define_scalar_same_dtype!(int_tensor_remainder_scalar, int_tensor_remainder_scalar);
define_binary_same_dtype!(int_tensor_matmul, int_tensor_matmul);
define_unary_same_dtype!(int_tensor_abs, int_tensor_abs);
define_unary_same_dtype!(int_tensor_sum, int_tensor_sum);
define_unary_dim_same_dtype!(int_tensor_sum_dim, int_tensor_sum_dim);
define_unary_same_dtype!(int_tensor_prod, int_tensor_prod);
define_unary_dim_same_dtype!(int_tensor_prod_dim, int_tensor_prod_dim);
define_unary_dim_same_dtype!(int_tensor_mean_dim, int_tensor_mean_dim);
define_unary_dim_same_dtype!(int_tensor_cumsum, int_tensor_cumsum);
define_unary_dim_same_dtype!(int_tensor_cumprod, int_tensor_cumprod);
define_unary_dim_same_dtype!(int_tensor_cummin, int_tensor_cummin);
define_unary_dim_same_dtype!(int_tensor_cummax, int_tensor_cummax);
define_unary_dim_same_dtype!(int_tensor_argmax, int_tensor_argmax);
define_unary_dim_same_dtype!(int_tensor_argmin, int_tensor_argmin);
define_compare_binary!(int_tensor_equal, int_tensor_equal);
define_compare_scalar!(int_tensor_equal_elem, int_tensor_equal_elem);
define_compare_binary!(int_tensor_greater, int_tensor_greater);
define_compare_scalar!(int_tensor_greater_elem, int_tensor_greater_elem);
define_compare_binary!(int_tensor_greater_equal, int_tensor_greater_equal);
define_compare_scalar!(int_tensor_greater_equal_elem, int_tensor_greater_equal_elem);
define_compare_binary!(int_tensor_lower, int_tensor_lower);
define_compare_scalar!(int_tensor_lower_elem, int_tensor_lower_elem);
define_compare_binary!(int_tensor_lower_equal, int_tensor_lower_equal);
define_compare_scalar!(int_tensor_lower_equal_elem, int_tensor_lower_equal_elem);
define_binary_same_dtype!(int_tensor_bitwise_and, int_tensor_bitwise_and);
define_scalar_same_dtype!(int_tensor_bitwise_and_scalar, int_tensor_bitwise_and_scalar);
define_binary_same_dtype!(int_tensor_bitwise_or, int_tensor_bitwise_or);
define_scalar_same_dtype!(int_tensor_bitwise_or_scalar, int_tensor_bitwise_or_scalar);
define_binary_same_dtype!(int_tensor_bitwise_xor, int_tensor_bitwise_xor);
define_scalar_same_dtype!(int_tensor_bitwise_xor_scalar, int_tensor_bitwise_xor_scalar);
define_unary_same_dtype!(int_tensor_bitwise_not, int_tensor_bitwise_not);
define_binary_same_dtype!(int_tensor_bitwise_left_shift, int_tensor_bitwise_left_shift);
define_scalar_same_dtype!(
    int_tensor_bitwise_left_shift_scalar,
    int_tensor_bitwise_left_shift_scalar
);
define_binary_same_dtype!(
    int_tensor_bitwise_right_shift,
    int_tensor_bitwise_right_shift
);
define_scalar_same_dtype!(
    int_tensor_bitwise_right_shift_scalar,
    int_tensor_bitwise_right_shift_scalar
);
define_repeat_dim_same_dtype!(int_tensor_repeat_dim, int_tensor_repeat_dim);
define_compare_binary!(int_tensor_not_equal, int_tensor_not_equal);
define_compare_scalar!(int_tensor_not_equal_elem, int_tensor_not_equal_elem);
define_binary_same_dtype!(int_tensor_powi, int_tensor_powi);
define_scalar_same_dtype!(int_tensor_powi_scalar, int_tensor_powi_scalar);
define_scalar_same_dtype!(int_tensor_clamp_min, int_tensor_clamp_min);
define_scalar_same_dtype!(int_tensor_clamp_max, int_tensor_clamp_max);
define_clamp_same_dtype!(int_tensor_clamp, int_tensor_clamp);
define_unary_same_dtype!(int_tensor_neg, int_tensor_neg);
define_unary_same_dtype!(int_tensor_mean, int_tensor_mean);
define_unary_same_dtype!(int_tensor_max, int_tensor_max);
define_unary_dim_same_dtype!(int_tensor_max_dim, int_tensor_max_dim);
define_with_indices_no_dtype!(
    int_tensor_max_dim_with_indices,
    int_tensor_max_dim_with_indices,
    IntDType::I64.into()
);
define_unary_same_dtype!(int_tensor_max_abs, int_tensor_max_abs);
define_unary_dim_same_dtype!(int_tensor_max_abs_dim, int_tensor_max_abs_dim);
define_unary_same_dtype!(int_tensor_min, int_tensor_min);
define_unary_dim_same_dtype!(int_tensor_min_dim, int_tensor_min_dim);
define_with_indices_no_dtype!(
    int_tensor_min_dim_with_indices,
    int_tensor_min_dim_with_indices,
    IntDType::I64.into()
);
define_unary_same_dtype!(int_tensor_transpose, int_tensor_transpose);
define_bool_reduce!(int_tensor_any, int_tensor_any);
define_bool_reduce_dim!(int_tensor_any_dim, int_tensor_any_dim);
define_bool_reduce!(int_tensor_all, int_tensor_all);
define_bool_reduce_dim!(int_tensor_all_dim, int_tensor_all_dim);
define_unary_same_dtype!(int_tensor_sign, int_tensor_sign);
define_sort_same_dtype!(int_tensor_sort, int_tensor_sort);
define_sort_with_indices_no_dtype!(
    int_tensor_sort_with_indices,
    int_tensor_sort_with_indices,
    IntDType::I64.into()
);
define_argsort_no_dtype!(int_tensor_argsort, int_tensor_argsort, IntDType::I64.into());

pub(crate) fn int_tensor_cat(
    tensors: Vec<DylibTensor>,
    dim: usize,
) -> Result<DylibTensor, DylibError> {
    let Some(first) = tensors.first().cloned() else {
        return Err(DylibError::InvalidInput(
            "int_tensor_cat requires at least one tensor".into(),
        ));
    };

    let refs = tensors.iter().collect::<Vec<_>>();
    ensure_same_runtime_and_device("int_tensor_cat", &refs)?;

    let runtime = get_runtime(first.runtime_id)?;
    let handles = tensors
        .iter()
        .map(|tensor| tensor.handle)
        .collect::<Vec<_>>();
    let handle = runtime
        .int_tensor_cat(&handles, dim)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &first, handle, first.dtype))
}

pub(crate) fn int_tensor_swap_dims(
    tensor: DylibTensor,
    dim1: usize,
    dim2: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_swap_dims(handle, dim1, dim2)
    })
}

pub(crate) fn int_tensor_permute(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_permute(handle, axes)
    })
}

pub(crate) fn int_tensor_flip(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_flip(handle, axes)
    })
}

pub(crate) fn int_tensor_reshape(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_reshape(handle, shape.as_slice())
    })
}

pub(crate) fn int_tensor_gather(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "int_tensor_gather",
        |runtime, tensor_handle, indices_handle| {
            runtime.int_tensor_gather(dim, tensor_handle, indices_handle)
        },
    )
}

pub(crate) fn int_tensor_scatter_add(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "int_tensor_scatter_add",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.int_tensor_scatter_add(dim, tensor_handle, indices_handle, value_handle)
        },
    )
}

pub(crate) fn int_tensor_select(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "int_tensor_select",
        |runtime, tensor_handle, indices_handle| {
            runtime.int_tensor_select(tensor_handle, dim, indices_handle)
        },
    )
}

pub(crate) fn int_tensor_select_add(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "int_tensor_select_add",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.int_tensor_select_add(tensor_handle, dim, indices_handle, value_handle)
        },
    )
}

pub(crate) fn int_tensor_slice(
    tensor: DylibTensor,
    slices: &[Slice],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_slice(handle, slices)
    })
}

pub(crate) fn int_tensor_slice_assign(
    tensor: DylibTensor,
    slices: &[Slice],
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        value,
        output_dtype,
        "int_tensor_slice_assign",
        |runtime, tensor_handle, value_handle| {
            runtime.int_tensor_slice_assign(tensor_handle, slices, value_handle)
        },
    )
}

pub(crate) fn int_tensor_mask_where(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        mask,
        value,
        output_dtype,
        "int_tensor_mask_where",
        |runtime, tensor_handle, mask_handle, value_handle| {
            runtime.int_tensor_mask_where(tensor_handle, mask_handle, value_handle)
        },
    )
}

pub(crate) fn int_tensor_mask_fill(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: Scalar,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        mask,
        output_dtype,
        "int_tensor_mask_fill",
        |runtime, tensor_handle, mask_handle| {
            runtime.int_tensor_mask_fill(tensor_handle, mask_handle, value)
        },
    )
}

pub(crate) fn int_tensor_expand(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_expand(handle, shape.as_slice())
    })
}

pub(crate) fn int_tensor_unfold(
    tensor: DylibTensor,
    dim: usize,
    size: usize,
    step: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.int_tensor_unfold(handle, dim, size, step)
    })
}

pub(crate) fn bool_tensor_from_data(
    data: TensorData,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    let requested_dtype = dtype_to_bool_dtype(data.dtype).unwrap_or(BoolDType::Native);
    let data = data.convert_dtype(requested_dtype.into());
    let requested_shape = data.shape.clone();
    let values = bool_data_to_u8(data)?;

    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .bool_tensor_from_u8_data(
            snapshot.handle,
            requested_shape.as_slice(),
            &values,
            requested_dtype,
        )
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &requested_shape,
        handle,
        requested_dtype.into(),
    ))
}

pub(crate) fn bool_tensor_into_data(tensor: DylibTensor) -> Result<TensorData, DylibError> {
    let runtime = get_runtime(tensor.runtime_id)?;
    let values = runtime
        .bool_tensor_into_u8_data(tensor.handle)
        .map_err(map_call_error)?;
    let shape = shape_from_runtime(&runtime, tensor.handle, &tensor.shape);
    Ok(TensorData::new(values, shape).convert_dtype(tensor.dtype))
}

pub(crate) fn bool_tensor_to_device(
    tensor: DylibTensor,
    device: &DylibDevice,
) -> Result<DylibTensor, DylibError> {
    if tensor.device == *device {
        return Ok(tensor);
    }

    let (target_runtime, target_snapshot) = resolve_device_context(device)?;

    if tensor.runtime_id == target_snapshot.runtime_id {
        let handle = target_runtime
            .bool_tensor_to_device(tensor.handle, target_snapshot.handle)
            .map_err(map_call_error)?;
        return Ok(tensor_from_output_with_fallback(
            &target_runtime,
            tensor.runtime_id,
            device,
            &tensor.shape,
            handle,
            tensor.dtype,
        ));
    }

    let data = bool_tensor_into_data(tensor)?;
    bool_tensor_from_data(data, device)
}

pub(crate) fn bool_tensor_into_int(
    tensor: DylibTensor,
    out_dtype: IntDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_into_int(handle, out_dtype)
    })
}

pub(crate) fn bool_tensor_into_float(
    tensor: DylibTensor,
    out_dtype: FloatDType,
) -> Result<DylibTensor, DylibError> {
    let output_dtype: DType = out_dtype.into();
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_into_float(handle, out_dtype)
    })
}

pub(crate) fn bool_tensor_empty(
    shape: Shape,
    device: &DylibDevice,
    dtype: BoolDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .bool_tensor_empty(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn bool_tensor_zeros(
    shape: Shape,
    device: &DylibDevice,
    dtype: BoolDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .bool_tensor_zeros(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

pub(crate) fn bool_tensor_ones(
    shape: Shape,
    device: &DylibDevice,
    dtype: BoolDType,
) -> Result<DylibTensor, DylibError> {
    let (runtime, snapshot) = resolve_device_context(device)?;
    let handle = runtime
        .bool_tensor_ones(snapshot.handle, shape.as_slice(), dtype)
        .map_err(map_call_error)?;

    Ok(tensor_from_output_with_fallback(
        &runtime,
        snapshot.runtime_id,
        device,
        &shape,
        handle,
        dtype.into(),
    ))
}

define_unary_same_dtype!(bool_tensor_not, bool_tensor_not);
define_binary_same_dtype!(bool_tensor_and, bool_tensor_and);
define_binary_same_dtype!(bool_tensor_or, bool_tensor_or);
define_binary_same_dtype!(bool_tensor_equal, bool_tensor_equal);
define_scalar_same_dtype!(bool_tensor_equal_elem, bool_tensor_equal_elem);
define_repeat_dim_same_dtype!(bool_tensor_repeat_dim, bool_tensor_repeat_dim);
define_binary_same_dtype!(bool_tensor_not_equal, bool_tensor_not_equal);
define_scalar_same_dtype!(bool_tensor_not_equal_elem, bool_tensor_not_equal_elem);
define_binary_same_dtype!(bool_tensor_xor, bool_tensor_xor);
define_unary_same_dtype!(bool_tensor_transpose, bool_tensor_transpose);
define_unary_same_dtype!(bool_tensor_any, bool_tensor_any);
define_unary_dim_same_dtype!(bool_tensor_any_dim, bool_tensor_any_dim);
define_unary_same_dtype!(bool_tensor_all, bool_tensor_all);
define_unary_dim_same_dtype!(bool_tensor_all_dim, bool_tensor_all_dim);

pub(crate) fn bool_tensor_cat(
    tensors: Vec<DylibTensor>,
    dim: usize,
) -> Result<DylibTensor, DylibError> {
    let Some(first) = tensors.first().cloned() else {
        return Err(DylibError::InvalidInput(
            "bool_tensor_cat requires at least one tensor".into(),
        ));
    };

    let refs = tensors.iter().collect::<Vec<_>>();
    ensure_same_runtime_and_device("bool_tensor_cat", &refs)?;

    let runtime = get_runtime(first.runtime_id)?;
    let handles = tensors
        .iter()
        .map(|tensor| tensor.handle)
        .collect::<Vec<_>>();
    let handle = runtime
        .bool_tensor_cat(&handles, dim)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &first, handle, first.dtype))
}

pub(crate) fn bool_tensor_reshape(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_reshape(handle, shape.as_slice())
    })
}

pub(crate) fn bool_tensor_gather(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "bool_tensor_gather",
        |runtime, tensor_handle, indices_handle| {
            runtime.bool_tensor_gather(dim, tensor_handle, indices_handle)
        },
    )
}

pub(crate) fn bool_tensor_scatter_or(
    dim: usize,
    tensor: DylibTensor,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "bool_tensor_scatter_or",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.bool_tensor_scatter_or(dim, tensor_handle, indices_handle, value_handle)
        },
    )
}

pub(crate) fn bool_tensor_select(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        indices,
        output_dtype,
        "bool_tensor_select",
        |runtime, tensor_handle, indices_handle| {
            runtime.bool_tensor_select(tensor_handle, dim, indices_handle)
        },
    )
}

pub(crate) fn bool_tensor_select_or(
    tensor: DylibTensor,
    dim: usize,
    indices: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        indices,
        value,
        output_dtype,
        "bool_tensor_select_or",
        |runtime, tensor_handle, indices_handle, value_handle| {
            runtime.bool_tensor_select_or(tensor_handle, dim, indices_handle, value_handle)
        },
    )
}

pub(crate) fn bool_tensor_slice(
    tensor: DylibTensor,
    slices: &[Slice],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_slice(handle, slices)
    })
}

pub(crate) fn bool_tensor_slice_assign(
    tensor: DylibTensor,
    slices: &[Slice],
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        value,
        output_dtype,
        "bool_tensor_slice_assign",
        |runtime, tensor_handle, value_handle| {
            runtime.bool_tensor_slice_assign(tensor_handle, slices, value_handle)
        },
    )
}

pub(crate) fn bool_tensor_mask_where(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_ternary_op(
        tensor,
        mask,
        value,
        output_dtype,
        "bool_tensor_mask_where",
        |runtime, tensor_handle, mask_handle, value_handle| {
            runtime.bool_tensor_mask_where(tensor_handle, mask_handle, value_handle)
        },
    )
}

pub(crate) fn bool_tensor_mask_fill(
    tensor: DylibTensor,
    mask: DylibTensor,
    value: Scalar,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_binary_op(
        tensor,
        mask,
        output_dtype,
        "bool_tensor_mask_fill",
        |runtime, tensor_handle, mask_handle| {
            runtime.bool_tensor_mask_fill(tensor_handle, mask_handle, value)
        },
    )
}

pub(crate) fn bool_tensor_swap_dims(
    tensor: DylibTensor,
    dim1: usize,
    dim2: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_swap_dims(handle, dim1, dim2)
    })
}

pub(crate) fn bool_tensor_permute(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_permute(handle, axes)
    })
}

pub(crate) fn bool_tensor_flip(
    tensor: DylibTensor,
    axes: &[usize],
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_flip(handle, axes)
    })
}

pub(crate) fn bool_tensor_expand(
    tensor: DylibTensor,
    shape: Shape,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_expand(handle, shape.as_slice())
    })
}

pub(crate) fn bool_tensor_unfold(
    tensor: DylibTensor,
    dim: usize,
    size: usize,
    step: usize,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.bool_tensor_unfold(handle, dim, size, step)
    })
}

pub(crate) fn module_embedding(
    weights: DylibTensor,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_embedding", &[&weights, &indices])?;

    let runtime = get_runtime(weights.runtime_id)?;
    let handle = runtime
        .module_embedding(weights.handle, indices.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(
        &runtime,
        &weights,
        handle,
        weights.dtype,
    ))
}

pub(crate) fn module_embedding_backward(
    weights: DylibTensor,
    output_grad: DylibTensor,
    indices: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_embedding_backward",
        &[&weights, &output_grad, &indices],
    )?;

    let runtime = get_runtime(weights.runtime_id)?;
    let handle = runtime
        .module_embedding_backward(weights.handle, output_grad.handle, indices.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(
        &runtime,
        &weights,
        handle,
        weights.dtype,
    ))
}

pub(crate) fn module_conv1d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvOptions<1>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv1d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv1d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv1d_x_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<1>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv1d_x_backward", &[&x, &weight, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv1d_x_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv1d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<1>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv1d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv1d_weight_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv1d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv1d_bias_backward", &[&x, &bias, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv1d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv2d_x_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<2>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv2d_x_backward", &[&x, &weight, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv2d_x_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv2d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<2>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv2d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv2d_weight_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv2d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv2d_bias_backward", &[&x, &bias, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv2d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv3d_x_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<3>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv3d_x_backward", &[&x, &weight, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv3d_x_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv3d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvOptions<3>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv3d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv3d_weight_backward(x.handle, weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv3d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_conv3d_bias_backward", &[&x, &bias, &output_grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv3d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose1d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvTransposeOptions<1>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv_transpose1d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose1d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose1d_x_backward(
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<1>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose1d_x_backward",
        &[&weight, &output_grad],
    )?;

    let runtime = get_runtime(weight.runtime_id)?;
    let handle = runtime
        .module_conv_transpose1d_x_backward(weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &weight, handle, weight.dtype))
}

pub(crate) fn module_conv_transpose1d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<1>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose1d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose1d_weight_backward(
            x.handle,
            weight.handle,
            output_grad.handle,
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose1d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose1d_bias_backward",
        &[&x, &bias, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose1d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose2d_x_backward(
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<2>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose2d_x_backward",
        &[&weight, &output_grad],
    )?;

    let runtime = get_runtime(weight.runtime_id)?;
    let handle = runtime
        .module_conv_transpose2d_x_backward(weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &weight, handle, weight.dtype))
}

pub(crate) fn module_conv_transpose2d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<2>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose2d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose2d_weight_backward(
            x.handle,
            weight.handle,
            output_grad.handle,
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose2d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose2d_bias_backward",
        &[&x, &bias, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose2d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose3d_x_backward(
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<3>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose3d_x_backward",
        &[&weight, &output_grad],
    )?;

    let runtime = get_runtime(weight.runtime_id)?;
    let handle = runtime
        .module_conv_transpose3d_x_backward(weight.handle, output_grad.handle, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &weight, handle, weight.dtype))
}

pub(crate) fn module_conv_transpose3d_weight_backward(
    x: DylibTensor,
    weight: DylibTensor,
    output_grad: DylibTensor,
    options: ConvTransposeOptions<3>,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose3d_weight_backward",
        &[&x, &weight, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose3d_weight_backward(
            x.handle,
            weight.handle,
            output_grad.handle,
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose3d_bias_backward(
    x: DylibTensor,
    bias: DylibTensor,
    output_grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device(
        "module_conv_transpose3d_bias_backward",
        &[&x, &bias, &output_grad],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose3d_bias_backward(x.handle, bias.handle, output_grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_unfold4d(
    x: DylibTensor,
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_unfold4d(x.handle, kernel_size, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_avg_pool1d(
    x: DylibTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_avg_pool1d(
            x.handle,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_avg_pool1d_backward(
    x: DylibTensor,
    grad: DylibTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_avg_pool1d_backward", &[&x, &grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_avg_pool1d_backward(
            x.handle,
            grad.handle,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_adaptive_avg_pool1d(
    x: DylibTensor,
    output_size: usize,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_adaptive_avg_pool1d(x.handle, output_size)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_adaptive_avg_pool1d_backward(
    x: DylibTensor,
    grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_adaptive_avg_pool1d_backward", &[&x, &grad])?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_adaptive_avg_pool1d_backward(x.handle, grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_max_pool1d(
    x: DylibTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_max_pool1d(x.handle, kernel_size, stride, padding, dilation, ceil_mode)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_max_pool1d_with_indices<E: Send + Sync + 'static>(
    x: DylibTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
) -> Result<MaxPool1dWithIndices<super::backend::Dylib<E>>, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handles = runtime
        .module_max_pool1d_with_indices(x.handle, kernel_size, stride, padding, dilation, ceil_mode)
        .map_err(map_call_error)?;

    let output = tensor_from_output(&runtime, &x, handles.output, x.dtype);
    let indices = tensor_from_output(&runtime, &x, handles.indices, IntDType::I64.into());
    Ok(MaxPool1dWithIndices::new(output, indices))
}

pub(crate) fn module_max_pool1d_with_indices_backward<E: Send + Sync + 'static>(
    x: DylibTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: bool,
    output_grad: DylibTensor,
    indices: DylibTensor,
) -> Result<MaxPool1dBackward<super::backend::Dylib<E>>, DylibError> {
    ensure_same_runtime_and_device(
        "module_max_pool1d_with_indices_backward",
        &[&x, &output_grad, &indices],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_max_pool1d_with_indices_backward(
            x.handle,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.handle,
            indices.handle,
        )
        .map_err(map_call_error)?;
    let x_grad = tensor_from_output(&runtime, &x, handle, x.dtype);
    Ok(MaxPool1dBackward::new(x_grad))
}

pub(crate) fn module_conv2d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvOptions<2>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv2d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv2d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_deform_conv2d(
    x: DylibTensor,
    offset: DylibTensor,
    weight: DylibTensor,
    mask: Option<DylibTensor>,
    bias: Option<DylibTensor>,
    options: DeformConvOptions<2>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &offset, &weight];
    if let Some(ref mask) = mask {
        tensors.push(mask);
    }
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_deform_conv2d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_deform_conv2d(
            x.handle,
            offset.handle,
            weight.handle,
            mask.as_ref().map(|value| value.handle),
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_deform_conv2d_backward<E: Send + Sync + 'static>(
    x: DylibTensor,
    offset: DylibTensor,
    weight: DylibTensor,
    mask: Option<DylibTensor>,
    bias: Option<DylibTensor>,
    output_grad: DylibTensor,
    options: DeformConvOptions<2>,
) -> Result<DeformConv2dBackward<super::backend::Dylib<E>>, DylibError> {
    let mut tensors = vec![&x, &offset, &weight, &output_grad];
    if let Some(ref mask) = mask {
        tensors.push(mask);
    }
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_deform_conv2d_backward", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handles = runtime
        .module_deform_conv2d_backward(
            x.handle,
            offset.handle,
            weight.handle,
            mask.as_ref().map(|value| value.handle),
            bias.as_ref().map(|value| value.handle),
            output_grad.handle,
            options,
        )
        .map_err(map_call_error)?;

    let x_grad = tensor_from_output(&runtime, &x, handles.x_grad, x.dtype);
    let offset_grad = tensor_from_output(&runtime, &offset, handles.offset_grad, offset.dtype);
    let weight_grad = tensor_from_output(&runtime, &weight, handles.weight_grad, weight.dtype);
    let mask_grad = handles.mask_grad.map(|handle| {
        let source = mask.as_ref().unwrap_or(&x);
        tensor_from_output(&runtime, source, handle, source.dtype)
    });
    let bias_grad = handles.bias_grad.map(|handle| {
        let source = bias.as_ref().unwrap_or(&x);
        tensor_from_output(&runtime, source, handle, source.dtype)
    });

    Ok(DeformConv2dBackward::new(
        x_grad,
        offset_grad,
        weight_grad,
        mask_grad,
        bias_grad,
    ))
}

pub(crate) fn module_conv3d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvOptions<3>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv3d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv3d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose2d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvTransposeOptions<2>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv_transpose2d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose2d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_conv_transpose3d(
    x: DylibTensor,
    weight: DylibTensor,
    bias: Option<DylibTensor>,
    options: ConvTransposeOptions<3>,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&x, &weight];
    if let Some(ref bias) = bias {
        tensors.push(bias);
    }
    ensure_same_runtime_and_device("module_conv_transpose3d", &tensors)?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_conv_transpose3d(
            x.handle,
            weight.handle,
            bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_avg_pool2d(
    x: DylibTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_avg_pool2d(
            x.handle,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_avg_pool2d_backward(
    x: DylibTensor,
    grad: DylibTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_avg_pool2d_backward", &[&x, &grad])?;
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_avg_pool2d_backward(
            x.handle,
            grad.handle,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_adaptive_avg_pool2d(
    x: DylibTensor,
    output_size: [usize; 2],
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_adaptive_avg_pool2d(x.handle, output_size)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_adaptive_avg_pool2d_backward(
    x: DylibTensor,
    grad: DylibTensor,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_adaptive_avg_pool2d_backward", &[&x, &grad])?;
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_adaptive_avg_pool2d_backward(x.handle, grad.handle)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_max_pool2d(
    x: DylibTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_max_pool2d(x.handle, kernel_size, stride, padding, dilation, ceil_mode)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_max_pool2d_with_indices<E: Send + Sync + 'static>(
    x: DylibTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> Result<MaxPool2dWithIndices<super::backend::Dylib<E>>, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handles = runtime
        .module_max_pool2d_with_indices(x.handle, kernel_size, stride, padding, dilation, ceil_mode)
        .map_err(map_call_error)?;

    let output = tensor_from_output(&runtime, &x, handles.output, x.dtype);
    let indices = tensor_from_output(&runtime, &x, handles.indices, IntDType::I64.into());
    Ok(MaxPool2dWithIndices::new(output, indices))
}

pub(crate) fn module_max_pool2d_with_indices_backward<E: Send + Sync + 'static>(
    x: DylibTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    output_grad: DylibTensor,
    indices: DylibTensor,
) -> Result<MaxPool2dBackward<super::backend::Dylib<E>>, DylibError> {
    ensure_same_runtime_and_device(
        "module_max_pool2d_with_indices_backward",
        &[&x, &output_grad, &indices],
    )?;

    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_max_pool2d_with_indices_backward(
            x.handle,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad.handle,
            indices.handle,
        )
        .map_err(map_call_error)?;
    let x_grad = tensor_from_output(&runtime, &x, handle, x.dtype);
    Ok(MaxPool2dBackward::new(x_grad))
}

pub(crate) fn module_interpolate(
    x: DylibTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> Result<DylibTensor, DylibError> {
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_interpolate(x.handle, output_size, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_interpolate_backward(
    x: DylibTensor,
    grad: DylibTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> Result<DylibTensor, DylibError> {
    ensure_same_runtime_and_device("module_interpolate_backward", &[&x, &grad])?;
    let runtime = get_runtime(x.runtime_id)?;
    let handle = runtime
        .module_interpolate_backward(x.handle, grad.handle, output_size, options)
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &x, handle, x.dtype))
}

pub(crate) fn module_attention(
    query: DylibTensor,
    key: DylibTensor,
    value: DylibTensor,
    mask: Option<DylibTensor>,
    attn_bias: Option<DylibTensor>,
    options: AttentionModuleOptions,
) -> Result<DylibTensor, DylibError> {
    let mut tensors = vec![&query, &key, &value];
    if let Some(ref mask) = mask {
        tensors.push(mask);
    }
    if let Some(ref attn_bias) = attn_bias {
        tensors.push(attn_bias);
    }
    ensure_same_runtime_and_device("module_attention", &tensors)?;

    let runtime = get_runtime(query.runtime_id)?;
    let handle = runtime
        .module_attention(
            query.handle,
            key.handle,
            value.handle,
            mask.as_ref().map(|value| value.handle),
            attn_bias.as_ref().map(|value| value.handle),
            options,
        )
        .map_err(map_call_error)?;
    Ok(tensor_from_output(&runtime, &query, handle, query.dtype))
}

pub(crate) fn module_rfft(
    signal: DylibTensor,
    dim: usize,
) -> Result<(DylibTensor, DylibTensor), DylibError> {
    let runtime = get_runtime(signal.runtime_id)?;
    let handles = runtime
        .module_rfft(signal.handle, dim)
        .map_err(map_call_error)?;

    let real = tensor_from_output(&runtime, &signal, handles.real, signal.dtype);
    let imag = tensor_from_output(&runtime, &signal, handles.imag, signal.dtype);
    Ok((real, imag))
}

define_scalar_same_dtype!(activation_leaky_relu, activation_leaky_relu);
define_unary_same_dtype!(activation_relu, activation_relu);
define_binary_same_dtype!(activation_relu_backward, activation_relu_backward);
define_unary_same_dtype!(activation_gelu, activation_gelu);
define_binary_same_dtype!(activation_prelu, activation_prelu);
define_binary_same_dtype!(activation_gelu_backward, activation_gelu_backward);
define_unary_same_dtype!(activation_sigmoid, activation_sigmoid);
define_binary_same_dtype!(activation_sigmoid_backward, activation_sigmoid_backward);
define_unary_same_dtype!(activation_log_sigmoid, activation_log_sigmoid);
define_binary_same_dtype!(
    activation_log_sigmoid_backward,
    activation_log_sigmoid_backward
);

pub(crate) fn activation_hard_sigmoid(
    tensor: DylibTensor,
    alpha: Scalar,
    beta: Scalar,
) -> Result<DylibTensor, DylibError> {
    let output_dtype = tensor.dtype;
    forward_unary_op(tensor, output_dtype, |runtime, handle| {
        runtime.activation_hard_sigmoid(handle, alpha, beta)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceHandle;

    #[test]
    fn insert_device_reuses_existing_backend_device() {
        let registry = RuntimeRegistry::new();
        let first = DeviceSnapshot {
            runtime_id: 7,
            backend_type_id: 3,
            ordinal: 0,
            handle: DeviceHandle(11),
        };
        let duplicate = DeviceSnapshot {
            handle: DeviceHandle(22),
            ..first
        };

        let first_index = registry.insert_device(first);
        let duplicate_index = registry.insert_device(duplicate);

        assert_eq!(first_index, duplicate_index);
        assert_eq!(registry.devices.read().expect("device lock").len(), 1);
        assert_eq!(
            registry
                .device_entry(first_index)
                .expect("device entry")
                .refs
                .load(Ordering::Relaxed),
            2
        );

        registry.release_device(first_index);
        assert!(registry.device_entry(first_index).is_ok());

        registry.release_device(first_index);
        assert!(matches!(
            registry.device_entry(first_index),
            Err(DylibError::DeviceNotFound(_))
        ));
    }

    #[test]
    fn insert_device_keeps_distinct_backend_devices_separate() {
        let registry = RuntimeRegistry::new();
        let first = DeviceSnapshot {
            runtime_id: 7,
            backend_type_id: 3,
            ordinal: 0,
            handle: DeviceHandle(11),
        };
        let second = DeviceSnapshot {
            ordinal: 1,
            handle: DeviceHandle(22),
            ..first
        };

        let first_index = registry.insert_device(first);
        let second_index = registry.insert_device(second);

        assert_ne!(first_index, second_index);
        assert_eq!(registry.devices.read().expect("device lock").len(), 2);
    }
}
