use crate::loader::{LoadError, LoadedBackendPlugin, PluginCallError};
use crate::{DeviceHandle, TensorHandle};
use burn_backend::{
    BoolDType, DType, DTypeUsage, DTypeUsageSet, Distribution, ExecutionError, FloatDType,
    IntDType, Scalar, Shape, Slice, TensorData,
};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::Path;
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
        let registry_index = self.next_device_index.fetch_add(1, Ordering::Relaxed);
        self.devices
            .write()
            .expect("device lock")
            .insert(registry_index, Arc::new(DeviceEntry::new(snapshot)));
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
        | DType::Bool(_) => DTypeUsage::general(),
        _ => DTypeUsageSet::default(),
    }
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
