use crate::{
    BACKEND_PLUGIN_ABI_VERSION, BACKEND_TENSOR_OPS_ABI_VERSION, BackendPluginV1,
    BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer, OwnedUsizeBuffer, PluginStatus,
    PluginStatusCode, TensorBinaryOp, TensorHandle, TensorShapeRef,
};
use core::any::TypeId;
use core::ffi::c_char;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const ERR_PANIC: &[u8] = b"plugin panicked\0";

/// Result type used by the plugin adapter traits.
pub type PluginResult<T> = Result<T, PluginError>;

/// Error returned by trait-based plugin implementations.
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

    /// Creates an `Unsupported` error.
    pub const fn unsupported(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::Unsupported, message)
    }

    fn into_status(self) -> PluginStatus {
        PluginStatus::error(self.code, self.message.as_ptr().cast())
    }
}

/// Trait for plugin metadata and device management.
pub trait PluginMetadata: Send + Sync + 'static {
    /// Concrete device state stored behind plugin handles.
    type Device: Clone + Send + Sync + 'static;

    /// Returns the backend name.
    ///
    /// The returned bytes must be a static, null-terminated string.
    fn backend_name() -> &'static [u8];

    /// Seeds the plugin backend.
    ///
    /// The adapter provides all currently registered devices so backends that
    /// only expose per-device seed APIs can still implement the global plugin
    /// callback cleanly.
    fn seed(seed: u64, devices: &[Self::Device]) -> PluginResult<()> {
        let _ = (seed, devices);
        Ok(())
    }

    /// Synchronizes the plugin backend.
    ///
    /// The adapter provides all currently registered devices so backends that
    /// only expose per-device synchronization APIs can still implement the
    /// global plugin callback cleanly.
    fn sync(devices: &[Self::Device]) -> PluginResult<()> {
        let _ = devices;
        Ok(())
    }

    /// Returns how many devices are available for `type_id`.
    fn device_count(type_id: u16) -> usize;

    /// Creates a concrete device for `type_id` and `ordinal`.
    fn create_device(type_id: u16, ordinal: usize) -> PluginResult<Self::Device>;
}

/// Trait for the float tensor operations exposed by the current plugin ABI.
pub trait FloatTensorPlugin: PluginMetadata {
    /// Concrete float tensor state stored behind plugin handles.
    type FloatTensor: Clone + Send + Sync + 'static;

    /// Creates a tensor from host f32 data.
    fn tensor_from_f32_data(
        device: &Self::Device,
        shape: &[usize],
        data: &[f32],
    ) -> PluginResult<Self::FloatTensor>;

    /// Materializes a tensor into host f32 data.
    fn tensor_into_f32_data(tensor: &Self::FloatTensor) -> PluginResult<Vec<f32>>;

    /// Returns the tensor shape.
    fn tensor_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>>;

    /// Dispatches float binary tensor operations.
    fn tensor_binary(
        op: TensorBinaryOp,
        device: &Self::Device,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor>;
}

#[derive(Clone)]
struct TensorState<T> {
    device_handle: u64,
    tensor: T,
}

struct AdapterState<P: FloatTensorPlugin> {
    next_device_id: AtomicU64,
    next_tensor_id: AtomicU64,
    devices: Mutex<HashMap<u64, P::Device>>,
    tensors: Mutex<HashMap<u64, TensorState<P::FloatTensor>>>,
}

impl<P: FloatTensorPlugin> AdapterState<P> {
    fn new() -> Self {
        Self {
            next_device_id: AtomicU64::new(1),
            next_tensor_id: AtomicU64::new(1),
            devices: Mutex::new(HashMap::new()),
            tensors: Mutex::new(HashMap::new()),
        }
    }

    fn clear(&self) {
        self.next_device_id.store(1, Ordering::Relaxed);
        self.next_tensor_id.store(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").clear();
        self.tensors.lock().expect("tensor lock").clear();
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

    fn lookup_tensor(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::FloatTensor>, PluginStatus> {
        self.tensors
            .lock()
            .expect("tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn insert_device(&self, device: P::Device) -> DeviceHandle {
        let id = self.next_device_id.fetch_add(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").insert(id, device);
        DeviceHandle(id)
    }

    fn insert_tensor(&self, device_handle: DeviceHandle, tensor: P::FloatTensor) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.tensors.lock().expect("tensor lock").insert(
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
        self.tensors
            .lock()
            .expect("tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
    }

    fn release_tensor(&self, tensor: TensorHandle) {
        self.tensors.lock().expect("tensor lock").remove(&tensor.0);
    }
}

static ADAPTER_STATES: LazyLock<Mutex<HashMap<TypeId, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn state<P: FloatTensorPlugin>() -> &'static AdapterState<P> {
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

fn with_boundary(f: impl FnOnce() -> PluginStatus) -> PluginStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => PluginError::failed(ERR_PANIC).into_status(),
    }
}

fn status_from_result(result: PluginResult<()>) -> PluginStatus {
    match result {
        Ok(()) => ok(),
        Err(error) => error.into_status(),
    }
}

fn try_shape(shape: TensorShapeRef) -> Result<Vec<usize>, PluginStatus> {
    if shape.rank == 0 {
        return Ok(Vec::new());
    }
    if shape.dims.is_null() {
        return Err(invalid_argument());
    }

    let dims = unsafe { slice::from_raw_parts(shape.dims, shape.rank) };
    Ok(dims.to_vec())
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

unsafe extern "C" fn backend_name<P: FloatTensorPlugin>() -> *const c_char {
    P::backend_name().as_ptr().cast()
}

unsafe extern "C" fn seed<P: FloatTensorPlugin>(seed: u64) -> PluginStatus {
    with_boundary(|| {
        let devices = state::<P>().devices_snapshot();
        status_from_result(P::seed(seed, &devices))
    })
}

unsafe extern "C" fn sync<P: FloatTensorPlugin>() -> PluginStatus {
    with_boundary(|| {
        let devices = state::<P>().devices_snapshot();
        status_from_result(P::sync(&devices))
    })
}

unsafe extern "C" fn device_count<P: FloatTensorPlugin>(type_id: u16) -> usize {
    catch_unwind(AssertUnwindSafe(|| P::device_count(type_id))).unwrap_or(0)
}

unsafe extern "C" fn create_device<P: FloatTensorPlugin>(
    type_id: u16,
    ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_device.is_null() {
            return invalid_argument();
        }

        let available = P::device_count(type_id);
        if available != 0 && ordinal >= available {
            return invalid_argument();
        }

        let device = match P::create_device(type_id, ordinal) {
            Ok(device) => device,
            Err(error) => return error.into_status(),
        };
        let handle = state::<P>().insert_device(device);

        unsafe {
            *out_device = handle;
        }
        ok()
    })
}

unsafe extern "C" fn release_device<P: FloatTensorPlugin>(device: DeviceHandle) -> PluginStatus {
    with_boundary(|| {
        state::<P>().release_device(device);
        ok()
    })
}

unsafe extern "C" fn tensor_from_f32_data<P: FloatTensorPlugin>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match state::<P>().lookup_device(device) {
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

        let tensor = match P::tensor_from_f32_data(&device_state, &shape, &values) {
            Ok(tensor) => tensor,
            Err(error) => return error.into_status(),
        };
        let handle = state::<P>().insert_tensor(device, tensor);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn tensor_into_f32_data<P: FloatTensorPlugin>(
    tensor: TensorHandle,
    out_data: *mut OwnedF32Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match state::<P>().lookup_tensor(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mut values = match P::tensor_into_f32_data(&tensor_state.tensor) {
            Ok(values) => values,
            Err(error) => return error.into_status(),
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

unsafe extern "C" fn tensor_shape<P: FloatTensorPlugin>(
    tensor: TensorHandle,
    out_shape: *mut OwnedUsizeBuffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_shape.is_null() {
            return invalid_argument();
        }

        let tensor_state = match state::<P>().lookup_tensor(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mut dims = match P::tensor_shape(&tensor_state.tensor) {
            Ok(dims) => dims,
            Err(error) => return error.into_status(),
        };
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

unsafe extern "C" fn tensor_binary<P: FloatTensorPlugin>(
    op: TensorBinaryOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let lhs_state = match state::<P>().lookup_tensor(lhs) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let rhs_state = match state::<P>().lookup_tensor(rhs) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if lhs_state.device_handle != rhs_state.device_handle {
            return invalid_argument();
        }

        let device = match state::<P>().lookup_device(DeviceHandle(lhs_state.device_handle)) {
            Ok(device) => device,
            Err(status) => return status,
        };

        let out = match P::tensor_binary(op, &device, &lhs_state.tensor, &rhs_state.tensor) {
            Ok(out) => out,
            Err(error) => return error.into_status(),
        };
        let handle = state::<P>().insert_tensor(DeviceHandle(lhs_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn release_tensor<P: FloatTensorPlugin>(tensor: TensorHandle) -> PluginStatus {
    with_boundary(|| {
        state::<P>().release_tensor(tensor);
        ok()
    })
}

unsafe extern "C" fn release_f32_buffer(buffer: OwnedF32Buffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn release_usize_buffer(buffer: OwnedUsizeBuffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

/// Builds the metadata table for a trait-backed plugin implementation.
pub const fn backend_plugin_v1<P: FloatTensorPlugin>() -> BackendPluginV1 {
    BackendPluginV1 {
        abi_version: BACKEND_PLUGIN_ABI_VERSION,
        backend_name: backend_name::<P>,
        seed: seed::<P>,
        sync: sync::<P>,
        device_count: device_count::<P>,
    }
}

/// Builds the tensor operation table for a trait-backed plugin implementation.
pub const fn backend_tensor_ops_v1<P: FloatTensorPlugin>() -> BackendTensorOpsV1 {
    BackendTensorOpsV1 {
        abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
        create_device: create_device::<P>,
        release_device: release_device::<P>,
        tensor_from_f32_data: tensor_from_f32_data::<P>,
        tensor_into_f32_data: tensor_into_f32_data::<P>,
        tensor_shape: tensor_shape::<P>,
        tensor_binary: tensor_binary::<P>,
        release_tensor: release_tensor::<P>,
        release_f32_buffer,
        release_usize_buffer,
    }
}

/// Clears the adapter state for a plugin implementation.
///
/// This is primarily intended for tests.
#[doc(hidden)]
pub fn reset_state<P: FloatTensorPlugin>() {
    state::<P>().clear();
}
