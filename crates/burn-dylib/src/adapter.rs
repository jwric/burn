use burn_backend::{
    Backend, BoolDType, Device as BurnDevice, DeviceId, Distribution, FloatDType, IntDType, Scalar,
    Shape, Slice, TensorData, TensorMetadata,
};

use crate::{
    AbiBoolDType, AbiDistribution, AbiDistributionKind, AbiFloatDType, AbiIntDType, AbiScalar,
    AbiScalarKind, AbiSliceRef, BACKEND_PLUGIN_ABI_VERSION, BACKEND_TENSOR_OPS_ABI_VERSION,
    BackendNameFn, BackendPluginV1, BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer,
    OwnedUsizeBuffer, PluginStatus, PluginStatusCode, TensorHandle, TensorShapeRef,
};
use core::any::TypeId;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const ERR_PANIC: &[u8] = b"plugin panicked\0";
const ERR_EXECUTION: &[u8] = b"execution error\0";

/// Error returned by adapter helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginError {
    code: PluginStatusCode,
    message: &'static [u8],
}

impl PluginError {
    /// Creates a new plugin error.
    ///
    /// `message` must point to a static, null-terminated string.
    pub const fn new(code: PluginStatusCode, message: &'static [u8]) -> Self {
        Self { code, message }
    }

    /// Creates an `InvalidArgument` error.
    pub const fn invalid_argument(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::InvalidArgument, message)
    }

    /// Creates a `Failed` error.
    pub const fn failed(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::Failed, message)
    }

    fn into_status(self) -> PluginStatus {
        PluginStatus::error(self.code, self.message.as_ptr().cast())
    }
}

#[derive(Clone)]
struct TensorState<T> {
    device_handle: u64,
    tensor: T,
}

struct AdapterState<P: Backend> {
    next_device_id: AtomicU64,
    next_tensor_id: AtomicU64,
    devices: Mutex<HashMap<u64, P::Device>>,
    float_tensors: Mutex<HashMap<u64, TensorState<P::FloatTensorPrimitive>>>,
    int_tensors: Mutex<HashMap<u64, TensorState<P::IntTensorPrimitive>>>,
    bool_tensors: Mutex<HashMap<u64, TensorState<P::BoolTensorPrimitive>>>,
}

impl<P: Backend> AdapterState<P> {
    fn new() -> Self {
        Self {
            next_device_id: AtomicU64::new(1),
            next_tensor_id: AtomicU64::new(1),
            devices: Mutex::new(HashMap::new()),
            float_tensors: Mutex::new(HashMap::new()),
            int_tensors: Mutex::new(HashMap::new()),
            bool_tensors: Mutex::new(HashMap::new()),
        }
    }

    fn clear(&self) {
        self.next_device_id.store(1, Ordering::Relaxed);
        self.next_tensor_id.store(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").clear();
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .clear();
        self.int_tensors.lock().expect("int tensor lock").clear();
        self.bool_tensors.lock().expect("bool tensor lock").clear();
    }

    fn devices_snapshot(&self) -> Vec<P::Device> {
        self.devices
            .lock()
            .expect("device lock")
            .values()
            .cloned()
            .collect()
    }

    fn lookup_device(&self, handle: DeviceHandle) -> Result<P::Device, PluginStatus> {
        self.devices
            .lock()
            .expect("device lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_float(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::FloatTensorPrimitive>, PluginStatus> {
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_int(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::IntTensorPrimitive>, PluginStatus> {
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_bool(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::BoolTensorPrimitive>, PluginStatus> {
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_tensor_shape(&self, handle: TensorHandle) -> Result<Shape, PluginStatus> {
        if let Some(state) = self
            .float_tensors
            .lock()
            .expect("float tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(state.tensor.shape());
        }

        if let Some(state) = self
            .int_tensors
            .lock()
            .expect("int tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(state.tensor.shape());
        }

        if let Some(state) = self
            .bool_tensors
            .lock()
            .expect("bool tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(state.tensor.shape());
        }

        Err(invalid_argument())
    }

    fn insert_device(&self, device: P::Device) -> DeviceHandle {
        let id = self.next_device_id.fetch_add(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").insert(id, device);
        DeviceHandle(id)
    }

    fn insert_float(
        &self,
        device_handle: DeviceHandle,
        tensor: P::FloatTensorPrimitive,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .insert(
                id,
                TensorState {
                    device_handle: device_handle.0,
                    tensor,
                },
            );
        TensorHandle(id)
    }

    fn insert_int(
        &self,
        device_handle: DeviceHandle,
        tensor: P::IntTensorPrimitive,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.int_tensors.lock().expect("int tensor lock").insert(
            id,
            TensorState {
                device_handle: device_handle.0,
                tensor,
            },
        );
        TensorHandle(id)
    }

    fn insert_bool(
        &self,
        device_handle: DeviceHandle,
        tensor: P::BoolTensorPrimitive,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.bool_tensors.lock().expect("bool tensor lock").insert(
            id,
            TensorState {
                device_handle: device_handle.0,
                tensor,
            },
        );
        TensorHandle(id)
    }

    fn release_device(&self, device: DeviceHandle) {
        self.devices.lock().expect("device lock").remove(&device.0);
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
    }

    fn release_tensor(&self, tensor: TensorHandle) {
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .remove(&tensor.0);
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .remove(&tensor.0);
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .remove(&tensor.0);
    }
}

static ADAPTER_STATES: LazyLock<Mutex<HashMap<TypeId, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn adapter_state<P: Backend>() -> &'static AdapterState<P> {
    let mut states = ADAPTER_STATES.lock().expect("adapter state lock");
    let ptr = states
        .entry(TypeId::of::<P>())
        .or_insert_with(|| Box::into_raw(Box::new(AdapterState::<P>::new())) as usize);

    unsafe { &*(*ptr as *const AdapterState<P>) }
}

fn ok() -> PluginStatus {
    PluginStatus::ok()
}

fn invalid_argument() -> PluginStatus {
    PluginError::invalid_argument(ERR_INVALID_ARGUMENT).into_status()
}

fn execution_error() -> PluginStatus {
    PluginError::failed(ERR_EXECUTION).into_status()
}

fn with_boundary(f: impl FnOnce() -> PluginStatus) -> PluginStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => PluginError::failed(ERR_PANIC).into_status(),
    }
}

fn try_shape(shape: TensorShapeRef) -> Result<Shape, PluginStatus> {
    if shape.rank == 0 {
        return Ok(Shape::new([]));
    }
    if shape.dims.is_null() {
        return Err(invalid_argument());
    }

    let dims = unsafe { slice::from_raw_parts(shape.dims, shape.rank) };
    Ok(Shape::new_raw(dims.into()))
}

fn try_f32_data(data: F32SliceRef) -> Result<Vec<f32>, PluginStatus> {
    if data.len == 0 {
        return Ok(Vec::new());
    }
    if data.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(data.ptr, data.len) };
    Ok(values.to_vec())
}

fn try_slices(slices: AbiSliceRef) -> Result<Vec<Slice>, PluginStatus> {
    if slices.len == 0 {
        return Ok(Vec::new());
    }

    if slices.ptr.is_null() {
        return Err(invalid_argument());
    }

    let items = unsafe { slice::from_raw_parts(slices.ptr, slices.len) };
    Ok(items
        .iter()
        .map(|slice| Slice {
            start: slice.start,
            end: if slice.has_end == 0 {
                None
            } else {
                Some(slice.end)
            },
            step: slice.step,
        })
        .collect())
}

fn scalar_from_abi(value: AbiScalar) -> Scalar {
    match value.kind {
        AbiScalarKind::Float => Scalar::Float(f64::from_bits(value.payload)),
        AbiScalarKind::Int => Scalar::Int(value.payload as i64),
        AbiScalarKind::UInt => Scalar::UInt(value.payload),
        AbiScalarKind::Bool => Scalar::Bool(value.payload != 0),
    }
}

fn distribution_from_abi(value: AbiDistribution) -> Distribution {
    match value.kind {
        AbiDistributionKind::Default => Distribution::Default,
        AbiDistributionKind::Bernoulli => Distribution::Bernoulli(value.param0),
        AbiDistributionKind::Uniform => Distribution::Uniform(value.param0, value.param1),
        AbiDistributionKind::Normal => Distribution::Normal(value.param0, value.param1),
    }
}

fn float_dtype_from_abi(value: AbiFloatDType) -> FloatDType {
    match value {
        AbiFloatDType::F64 => FloatDType::F64,
        AbiFloatDType::F32 => FloatDType::F32,
        AbiFloatDType::Flex32 => FloatDType::Flex32,
        AbiFloatDType::F16 => FloatDType::F16,
        AbiFloatDType::BF16 => FloatDType::BF16,
    }
}

fn int_dtype_from_abi(value: AbiIntDType) -> IntDType {
    match value {
        AbiIntDType::I64 => IntDType::I64,
        AbiIntDType::I32 => IntDType::I32,
        AbiIntDType::I16 => IntDType::I16,
        AbiIntDType::I8 => IntDType::I8,
        AbiIntDType::U64 => IntDType::U64,
        AbiIntDType::U32 => IntDType::U32,
        AbiIntDType::U16 => IntDType::U16,
        AbiIntDType::U8 => IntDType::U8,
    }
}

fn bool_dtype_from_abi(value: AbiBoolDType) -> BoolDType {
    match value {
        AbiBoolDType::Native => BoolDType::Native,
        AbiBoolDType::U8 => BoolDType::U8,
        AbiBoolDType::U32 => BoolDType::U32,
    }
}

// Adapter naming conventions:
// - `abi_backend_*` shims back `BackendPluginV1` metadata/control callbacks.
// - `abi_float_tensor_*` shims back `BackendTensorOpsV1` float tensor callbacks.
// - `abi_release_*` shims release plugin-owned buffers/handles.

unsafe extern "C" fn abi_backend_seed<P: Backend>(seed: u64) -> PluginStatus {
    with_boundary(|| {
        for device in adapter_state::<P>().devices_snapshot() {
            P::seed(&device, seed);
        }

        ok()
    })
}

unsafe extern "C" fn abi_backend_sync<P: Backend>() -> PluginStatus {
    with_boundary(|| {
        for device in adapter_state::<P>().devices_snapshot() {
            if P::sync(&device).is_err() {
                return execution_error();
            }
        }

        ok()
    })
}

unsafe extern "C" fn abi_backend_device_count<P: Backend>(type_id: u16) -> usize {
    catch_unwind(AssertUnwindSafe(|| P::device_count(type_id))).unwrap_or(0)
}

unsafe extern "C" fn abi_create_device<P: Backend>(
    type_id: u16,
    ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_device.is_null() {
            return invalid_argument();
        }

        let available = P::device_count(type_id);
        if available == 0 || ordinal >= available {
            return invalid_argument();
        }

        let ordinal = match u32::try_from(ordinal) {
            Ok(ordinal) => ordinal,
            Err(_) => return invalid_argument(),
        };
        let device = P::Device::from_id(DeviceId::new(type_id, ordinal));
        let handle = adapter_state::<P>().insert_device(device);

        unsafe {
            *out_device = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_release_device<P: Backend>(device: DeviceHandle) -> PluginStatus {
    with_boundary(|| {
        adapter_state::<P>().release_device(device);
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_from_f32_data<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };
        let values = match try_f32_data(data) {
            Ok(values) => values,
            Err(status) => return status,
        };

        let tensor = P::float_from_data(TensorData::new(values, shape), &device_state);
        let handle = adapter_state::<P>().insert_float(device, tensor);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_into_f32_data<P: Backend>(
    tensor: TensorHandle,
    out_data: *mut OwnedF32Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let data = match burn_backend::read_sync(P::float_into_data(tensor_state.tensor)) {
            Ok(data) => data,
            Err(_) => return execution_error(),
        };
        let mut values = match data.into_vec::<f32>() {
            Ok(values) => values,
            Err(_) => return execution_error(),
        };
        let buffer = OwnedF32Buffer {
            ptr: values.as_mut_ptr(),
            len: values.len(),
        };
        std::mem::forget(values);

        unsafe {
            *out_data = buffer;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_shape<P: Backend>(
    tensor: TensorHandle,
    out_shape: *mut OwnedUsizeBuffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_shape.is_null() {
            return invalid_argument();
        }

        let shape = match adapter_state::<P>().lookup_tensor_shape(tensor) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let mut dims = shape.as_slice().to_vec();
        let buffer = OwnedUsizeBuffer {
            ptr: dims.as_mut_ptr(),
            len: dims.len(),
        };
        std::mem::forget(dims);

        unsafe {
            *out_shape = buffer;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_random<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    distribution: AbiDistribution,
    dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_random(
            shape,
            distribution_from_abi(distribution),
            &device_state,
            float_dtype_from_abi(dtype),
        );
        let handle = adapter_state::<P>().insert_float(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_to_device<P: Backend>(
    tensor: TensorHandle,
    device: DeviceHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };

        let out = P::float_to_device(tensor_state.tensor, &device_state);
        let handle = adapter_state::<P>().insert_float(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_empty<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_empty(shape, &device_state, float_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_float(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_into_int<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::float_into_int(tensor_state.tensor, int_dtype_from_abi(out_dtype));
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

macro_rules! abi_float_unary_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor);
                let handle = adapter_state::<P>()
                    .insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_binary_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            lhs: TensorHandle,
            rhs: TensorHandle,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let lhs_state = match adapter_state::<P>().lookup_float(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs_state = match adapter_state::<P>().lookup_float(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                if lhs_state.device_handle != rhs_state.device_handle {
                    return invalid_argument();
                }

                let out = P::$backend_fn(lhs_state.tensor, rhs_state.tensor);
                let handle =
                    adapter_state::<P>().insert_float(DeviceHandle(lhs_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_scalar_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            scalar: AbiScalar,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, scalar_from_abi(scalar));
                let handle = adapter_state::<P>()
                    .insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim);
                let handle = adapter_state::<P>()
                    .insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_compare_binary_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            lhs: TensorHandle,
            rhs: TensorHandle,
            out_dtype: AbiBoolDType,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let lhs_state = match adapter_state::<P>().lookup_float(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs_state = match adapter_state::<P>().lookup_float(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                if lhs_state.device_handle != rhs_state.device_handle {
                    return invalid_argument();
                }

                let out = P::$backend_fn(
                    lhs_state.tensor,
                    rhs_state.tensor,
                    bool_dtype_from_abi(out_dtype),
                );
                let handle =
                    adapter_state::<P>().insert_bool(DeviceHandle(lhs_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_compare_scalar_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            rhs: AbiScalar,
            out_dtype: AbiBoolDType,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(
                    tensor_state.tensor,
                    scalar_from_abi(rhs),
                    bool_dtype_from_abi(out_dtype),
                );
                let handle =
                    adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_arg_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            out_dtype: AbiIntDType,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim, int_dtype_from_abi(out_dtype));
                let handle =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

abi_float_binary_fn!(abi_float_tensor_add, float_add);
abi_float_scalar_fn!(abi_float_tensor_add_scalar, float_add_scalar);
abi_float_binary_fn!(abi_float_tensor_sub, float_sub);
abi_float_scalar_fn!(abi_float_tensor_sub_scalar, float_sub_scalar);
abi_float_binary_fn!(abi_float_tensor_mul, float_mul);
abi_float_scalar_fn!(abi_float_tensor_mul_scalar, float_mul_scalar);
abi_float_binary_fn!(abi_float_tensor_div, float_div);
abi_float_scalar_fn!(abi_float_tensor_div_scalar, float_div_scalar);
abi_float_binary_fn!(abi_float_tensor_remainder, float_remainder);
abi_float_scalar_fn!(abi_float_tensor_remainder_scalar, float_remainder_scalar);
abi_float_binary_fn!(abi_float_tensor_matmul, float_matmul);
abi_float_unary_fn!(abi_float_tensor_recip, float_recip);
abi_float_compare_binary_fn!(abi_float_tensor_equal, float_equal);
abi_float_compare_scalar_fn!(abi_float_tensor_equal_elem, float_equal_elem);
abi_float_compare_binary_fn!(abi_float_tensor_greater, float_greater);
abi_float_compare_scalar_fn!(abi_float_tensor_greater_elem, float_greater_elem);
abi_float_compare_binary_fn!(abi_float_tensor_greater_equal, float_greater_equal);
abi_float_compare_scalar_fn!(
    abi_float_tensor_greater_equal_elem,
    float_greater_equal_elem
);
abi_float_compare_binary_fn!(abi_float_tensor_lower, float_lower);
abi_float_compare_scalar_fn!(abi_float_tensor_lower_elem, float_lower_elem);
abi_float_compare_binary_fn!(abi_float_tensor_lower_equal, float_lower_equal);
abi_float_compare_scalar_fn!(abi_float_tensor_lower_equal_elem, float_lower_equal_elem);
abi_float_unary_fn!(abi_float_tensor_sum, float_sum);
abi_float_dim_fn!(abi_float_tensor_sum_dim, float_sum_dim);
abi_float_dim_fn!(abi_float_tensor_mean_dim, float_mean_dim);
abi_float_dim_fn!(abi_float_tensor_cumsum, float_cumsum);
abi_float_dim_fn!(abi_float_tensor_cumprod, float_cumprod);
abi_float_dim_fn!(abi_float_tensor_cummin, float_cummin);
abi_float_dim_fn!(abi_float_tensor_cummax, float_cummax);
abi_float_unary_fn!(abi_float_tensor_exp, float_exp);
abi_float_unary_fn!(abi_float_tensor_log, float_log);
abi_float_unary_fn!(abi_float_tensor_log1p, float_log1p);
abi_float_binary_fn!(abi_float_tensor_powf, float_powf);
abi_float_scalar_fn!(abi_float_tensor_powf_scalar, float_powf_scalar_impl);
abi_float_unary_fn!(abi_float_tensor_sqrt, float_sqrt);
abi_float_unary_fn!(abi_float_tensor_abs, float_abs);
abi_float_unary_fn!(abi_float_tensor_cos, float_cos);
abi_float_unary_fn!(abi_float_tensor_sin, float_sin);
abi_float_unary_fn!(abi_float_tensor_tan, float_tan);
abi_float_unary_fn!(abi_float_tensor_cosh, float_cosh);
abi_float_unary_fn!(abi_float_tensor_sinh, float_sinh);
abi_float_unary_fn!(abi_float_tensor_tanh, float_tanh);
abi_float_unary_fn!(abi_float_tensor_acos, float_acos);
abi_float_unary_fn!(abi_float_tensor_acosh, float_acosh);
abi_float_unary_fn!(abi_float_tensor_asin, float_asin);
abi_float_unary_fn!(abi_float_tensor_asinh, float_asinh);
abi_float_unary_fn!(abi_float_tensor_atan, float_atan);
abi_float_unary_fn!(abi_float_tensor_atanh, float_atanh);
abi_float_binary_fn!(abi_float_tensor_atan2, float_atan2);
abi_float_unary_fn!(abi_float_tensor_round, float_round);
abi_float_unary_fn!(abi_float_tensor_floor, float_floor);
abi_float_unary_fn!(abi_float_tensor_ceil, float_ceil);
abi_float_unary_fn!(abi_float_tensor_trunc, float_trunc);
abi_float_unary_fn!(abi_float_tensor_erf, float_erf);
abi_float_arg_fn!(abi_float_tensor_argmax, float_argmax);
abi_float_arg_fn!(abi_float_tensor_argmin, float_argmin);

unsafe extern "C" fn abi_float_tensor_cross<P: Backend>(
    lhs: TensorHandle,
    rhs: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let lhs_state = match adapter_state::<P>().lookup_float(lhs) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let rhs_state = match adapter_state::<P>().lookup_float(rhs) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if lhs_state.device_handle != rhs_state.device_handle {
            return invalid_argument();
        }

        let out = P::float_cross(lhs_state.tensor, rhs_state.tensor, dim);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(lhs_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_swap_dims<P: Backend>(
    tensor: TensorHandle,
    dim1: usize,
    dim2: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::float_swap_dims(tensor_state.tensor, dim1, dim2);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_permute<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_permute(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_flip<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_flip(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_reshape<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_reshape(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_gather<P: Backend>(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle {
            return invalid_argument();
        }

        let out = P::float_gather(dim, tensor_state.tensor, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_scatter_add<P: Backend>(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_float(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::float_scatter_add(
            dim,
            tensor_state.tensor,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_select<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle {
            return invalid_argument();
        }

        let out = P::float_select(tensor_state.tensor, dim, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_select_add<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_float(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::float_select_add(
            tensor_state.tensor,
            dim,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_slice<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };

        let out = P::float_slice(tensor_state.tensor, &slices);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_slice_assign<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_float(value) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if tensor_state.device_handle != value_state.device_handle {
            return invalid_argument();
        }

        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };

        let out = P::float_slice_assign(tensor_state.tensor, &slices, value_state.tensor);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_mask_where<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match adapter_state::<P>().lookup_bool(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_float(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != mask_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::float_mask_where(tensor_state.tensor, mask_state.tensor, value_state.tensor);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_mask_fill<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match adapter_state::<P>().lookup_bool(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != mask_state.device_handle {
            return invalid_argument();
        }

        let out = P::float_mask_fill(
            tensor_state.tensor,
            mask_state.tensor,
            scalar_from_abi(value),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_cast<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::float_cast(tensor_state.tensor, float_dtype_from_abi(out_dtype));
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_expand<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::float_expand(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_unfold<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    size: usize,
    step: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::float_unfold(tensor_state.tensor, dim, size, step);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_release_tensor<P: Backend>(tensor: TensorHandle) -> PluginStatus {
    with_boundary(|| {
        adapter_state::<P>().release_tensor(tensor);
        ok()
    })
}

unsafe extern "C" fn abi_release_f32_buffer(buffer: OwnedF32Buffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn abi_release_usize_buffer(buffer: OwnedUsizeBuffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

/// Builds the metadata table for a backend-backed plugin implementation.
pub const fn backend_plugin_v1<P: Backend>(backend_name: BackendNameFn) -> BackendPluginV1 {
    BackendPluginV1 {
        abi_version: BACKEND_PLUGIN_ABI_VERSION,
        backend_name,
        seed: abi_backend_seed::<P>,
        sync: abi_backend_sync::<P>,
        device_count: abi_backend_device_count::<P>,
    }
}

/// Builds the tensor operation table for a backend-backed plugin implementation.
pub const fn backend_tensor_ops_v1<P: Backend>() -> BackendTensorOpsV1 {
    BackendTensorOpsV1 {
        abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
        create_device: abi_create_device::<P>,
        release_device: abi_release_device::<P>,
        tensor_from_f32_data: abi_float_tensor_from_f32_data::<P>,
        tensor_into_f32_data: abi_float_tensor_into_f32_data::<P>,
        tensor_shape: abi_float_tensor_shape::<P>,
        tensor_random: abi_float_tensor_random::<P>,
        tensor_to_device: abi_float_tensor_to_device::<P>,
        tensor_empty: abi_float_tensor_empty::<P>,
        tensor_into_int: abi_float_tensor_into_int::<P>,
        tensor_add: abi_float_tensor_add::<P>,
        tensor_add_scalar: abi_float_tensor_add_scalar::<P>,
        tensor_sub: abi_float_tensor_sub::<P>,
        tensor_sub_scalar: abi_float_tensor_sub_scalar::<P>,
        tensor_mul: abi_float_tensor_mul::<P>,
        tensor_mul_scalar: abi_float_tensor_mul_scalar::<P>,
        tensor_div: abi_float_tensor_div::<P>,
        tensor_div_scalar: abi_float_tensor_div_scalar::<P>,
        tensor_remainder: abi_float_tensor_remainder::<P>,
        tensor_remainder_scalar: abi_float_tensor_remainder_scalar::<P>,
        tensor_matmul: abi_float_tensor_matmul::<P>,
        tensor_cross: abi_float_tensor_cross::<P>,
        tensor_recip: abi_float_tensor_recip::<P>,
        tensor_swap_dims: abi_float_tensor_swap_dims::<P>,
        tensor_permute: abi_float_tensor_permute::<P>,
        tensor_flip: abi_float_tensor_flip::<P>,
        tensor_reshape: abi_float_tensor_reshape::<P>,
        tensor_gather: abi_float_tensor_gather::<P>,
        tensor_scatter_add: abi_float_tensor_scatter_add::<P>,
        tensor_select: abi_float_tensor_select::<P>,
        tensor_select_add: abi_float_tensor_select_add::<P>,
        tensor_slice: abi_float_tensor_slice::<P>,
        tensor_slice_assign: abi_float_tensor_slice_assign::<P>,
        tensor_mask_where: abi_float_tensor_mask_where::<P>,
        tensor_mask_fill: abi_float_tensor_mask_fill::<P>,
        tensor_equal: abi_float_tensor_equal::<P>,
        tensor_equal_elem: abi_float_tensor_equal_elem::<P>,
        tensor_greater: abi_float_tensor_greater::<P>,
        tensor_greater_elem: abi_float_tensor_greater_elem::<P>,
        tensor_greater_equal: abi_float_tensor_greater_equal::<P>,
        tensor_greater_equal_elem: abi_float_tensor_greater_equal_elem::<P>,
        tensor_lower: abi_float_tensor_lower::<P>,
        tensor_lower_elem: abi_float_tensor_lower_elem::<P>,
        tensor_lower_equal: abi_float_tensor_lower_equal::<P>,
        tensor_lower_equal_elem: abi_float_tensor_lower_equal_elem::<P>,
        tensor_sum: abi_float_tensor_sum::<P>,
        tensor_sum_dim: abi_float_tensor_sum_dim::<P>,
        tensor_mean_dim: abi_float_tensor_mean_dim::<P>,
        tensor_cumsum: abi_float_tensor_cumsum::<P>,
        tensor_cumprod: abi_float_tensor_cumprod::<P>,
        tensor_cummin: abi_float_tensor_cummin::<P>,
        tensor_cummax: abi_float_tensor_cummax::<P>,
        tensor_cast: abi_float_tensor_cast::<P>,
        tensor_exp: abi_float_tensor_exp::<P>,
        tensor_log: abi_float_tensor_log::<P>,
        tensor_log1p: abi_float_tensor_log1p::<P>,
        tensor_powf: abi_float_tensor_powf::<P>,
        tensor_powf_scalar: abi_float_tensor_powf_scalar::<P>,
        tensor_sqrt: abi_float_tensor_sqrt::<P>,
        tensor_abs: abi_float_tensor_abs::<P>,
        tensor_cos: abi_float_tensor_cos::<P>,
        tensor_sin: abi_float_tensor_sin::<P>,
        tensor_tan: abi_float_tensor_tan::<P>,
        tensor_cosh: abi_float_tensor_cosh::<P>,
        tensor_sinh: abi_float_tensor_sinh::<P>,
        tensor_tanh: abi_float_tensor_tanh::<P>,
        tensor_acos: abi_float_tensor_acos::<P>,
        tensor_acosh: abi_float_tensor_acosh::<P>,
        tensor_asin: abi_float_tensor_asin::<P>,
        tensor_asinh: abi_float_tensor_asinh::<P>,
        tensor_atan: abi_float_tensor_atan::<P>,
        tensor_atanh: abi_float_tensor_atanh::<P>,
        tensor_atan2: abi_float_tensor_atan2::<P>,
        tensor_round: abi_float_tensor_round::<P>,
        tensor_floor: abi_float_tensor_floor::<P>,
        tensor_ceil: abi_float_tensor_ceil::<P>,
        tensor_trunc: abi_float_tensor_trunc::<P>,
        tensor_erf: abi_float_tensor_erf::<P>,
        tensor_argmax: abi_float_tensor_argmax::<P>,
        tensor_argmin: abi_float_tensor_argmin::<P>,
        tensor_expand: abi_float_tensor_expand::<P>,
        tensor_unfold: abi_float_tensor_unfold::<P>,
        release_tensor: abi_release_tensor::<P>,
        release_f32_buffer: abi_release_f32_buffer,
        release_usize_buffer: abi_release_usize_buffer,
    }
}

/// Clears the adapter state for a backend implementation.
///
/// This is primarily intended for tests.
#[doc(hidden)]
pub fn reset_state<P: Backend>() {
    adapter_state::<P>().clear();
}
