use burn_dylib::{
    BACKEND_PLUGIN_ABI_VERSION, BACKEND_TENSOR_OPS_ABI_VERSION, BackendPluginV1,
    BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer, OwnedUsizeBuffer,
    PluginStatus, PluginStatusCode, TensorHandle, TensorShapeRef, export_backend_plugin_v1,
    export_backend_tensor_ops_v1,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

#[cfg(not(feature = "variant-b"))]
const BACKEND_NAME: &[u8] = b"mock-plugin-a\0";
#[cfg(feature = "variant-b")]
const BACKEND_NAME: &[u8] = b"mock-plugin-b\0";

const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";

#[derive(Clone)]
struct MockTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone)]
struct TensorState {
    device_handle: u64,
    tensor: MockTensor,
}

struct State {
    next_device_id: AtomicU64,
    next_tensor_id: AtomicU64,
    devices: Mutex<HashMap<u64, ()>>,
    tensors: Mutex<HashMap<u64, TensorState>>,
}

impl State {
    fn new() -> Self {
        Self {
            next_device_id: AtomicU64::new(1),
            next_tensor_id: AtomicU64::new(1),
            devices: Mutex::new(HashMap::new()),
            tensors: Mutex::new(HashMap::new()),
        }
    }

    fn insert_device(&self) -> DeviceHandle {
        let id = self.next_device_id.fetch_add(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").insert(id, ());
        DeviceHandle(id)
    }

    fn contains_device(&self, handle: DeviceHandle) -> bool {
        self.devices
            .lock()
            .expect("device lock")
            .contains_key(&handle.0)
    }

    fn release_device(&self, handle: DeviceHandle) {
        self.devices.lock().expect("device lock").remove(&handle.0);
        self.tensors
            .lock()
            .expect("tensor lock")
            .retain(|_, tensor| tensor.device_handle != handle.0);
    }

    fn insert_tensor(&self, device: DeviceHandle, tensor: MockTensor) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.tensors.lock().expect("tensor lock").insert(
            id,
            TensorState {
                device_handle: device.0,
                tensor,
            },
        );
        TensorHandle(id)
    }

    fn lookup_tensor(&self, handle: TensorHandle) -> Option<TensorState> {
        self.tensors
            .lock()
            .expect("tensor lock")
            .get(&handle.0)
            .cloned()
    }

    fn release_tensor(&self, handle: TensorHandle) {
        self.tensors.lock().expect("tensor lock").remove(&handle.0);
    }
}

static STATE: LazyLock<State> = LazyLock::new(State::new);

fn invalid_argument() -> PluginStatus {
    PluginStatus::error(
        PluginStatusCode::InvalidArgument,
        ERR_INVALID_ARGUMENT.as_ptr().cast(),
    )
}

fn ok() -> PluginStatus {
    PluginStatus::ok()
}

fn try_shape(shape: TensorShapeRef) -> Result<Vec<usize>, PluginStatus> {
    if shape.rank == 0 {
        return Ok(Vec::new());
    }
    if shape.dims.is_null() {
        return Err(invalid_argument());
    }

    let dims = unsafe { std::slice::from_raw_parts(shape.dims, shape.rank) };
    Ok(dims.to_vec())
}

fn try_f32_data(data: F32SliceRef) -> Result<Vec<f32>, PluginStatus> {
    if data.len == 0 {
        return Ok(Vec::new());
    }
    if data.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { std::slice::from_raw_parts(data.ptr, data.len) };
    Ok(values.to_vec())
}

fn checked_numel(shape: &[usize]) -> Result<usize, PluginStatus> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(invalid_argument)
}

fn create_tensor(shape: &[usize], data: &[f32]) -> Result<MockTensor, PluginStatus> {
    if checked_numel(shape)? != data.len() {
        return Err(invalid_argument());
    }

    Ok(MockTensor {
        shape: shape.to_vec(),
        data: data.to_vec(),
    })
}

fn tensor_add_impl(lhs: &MockTensor, rhs: &MockTensor) -> Result<MockTensor, PluginStatus> {
    if lhs.shape != rhs.shape {
        return Err(invalid_argument());
    }

    let bias = if cfg!(feature = "variant-b") { 1.0 } else { 0.0 };
    let out_data = lhs
        .data
        .iter()
        .zip(rhs.data.iter())
        .map(|(l, r)| l + r + bias)
        .collect::<Vec<_>>();

    Ok(MockTensor {
        shape: lhs.shape.clone(),
        data: out_data,
    })
}

unsafe extern "C" fn backend_name() -> *const core::ffi::c_char {
    BACKEND_NAME.as_ptr().cast()
}

unsafe extern "C" fn seed(_seed: u64) -> PluginStatus {
    ok()
}

unsafe extern "C" fn sync() -> PluginStatus {
    ok()
}

unsafe extern "C" fn device_count(_type_id: u16) -> usize {
    1
}

unsafe extern "C" fn create_device(
    _type_id: u16,
    ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    if out_device.is_null() || ordinal != 0 {
        return invalid_argument();
    }

    unsafe {
        *out_device = STATE.insert_device();
    }
    ok()
}

unsafe extern "C" fn release_device(device: DeviceHandle) -> PluginStatus {
    STATE.release_device(device);
    ok()
}

unsafe extern "C" fn tensor_from_f32_data(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    if out_tensor.is_null() || !STATE.contains_device(device) {
        return invalid_argument();
    }

    let shape = match try_shape(shape) {
        Ok(shape) => shape,
        Err(status) => return status,
    };
    let data = match try_f32_data(data) {
        Ok(data) => data,
        Err(status) => return status,
    };
    let tensor = match create_tensor(&shape, &data) {
        Ok(tensor) => tensor,
        Err(status) => return status,
    };

    unsafe {
        *out_tensor = STATE.insert_tensor(device, tensor);
    }
    ok()
}

unsafe extern "C" fn tensor_into_f32_data(
    tensor: TensorHandle,
    out_data: *mut OwnedF32Buffer,
) -> PluginStatus {
    if out_data.is_null() {
        return invalid_argument();
    }

    let tensor = match STATE.lookup_tensor(tensor) {
        Some(tensor) => tensor,
        None => return invalid_argument(),
    };
    let mut values = tensor.tensor.data;
    let buffer = OwnedF32Buffer {
        ptr: values.as_mut_ptr(),
        len: values.len(),
    };
    std::mem::forget(values);

    unsafe {
        *out_data = buffer;
    }
    ok()
}

unsafe extern "C" fn tensor_shape(
    tensor: TensorHandle,
    out_shape: *mut OwnedUsizeBuffer,
) -> PluginStatus {
    if out_shape.is_null() {
        return invalid_argument();
    }

    let tensor = match STATE.lookup_tensor(tensor) {
        Some(tensor) => tensor,
        None => return invalid_argument(),
    };
    let mut dims = tensor.tensor.shape;
    let buffer = OwnedUsizeBuffer {
        ptr: dims.as_mut_ptr(),
        len: dims.len(),
    };
    std::mem::forget(dims);

    unsafe {
        *out_shape = buffer;
    }
    ok()
}

unsafe extern "C" fn tensor_add(
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    if out_tensor.is_null() {
        return invalid_argument();
    }

    let lhs = match STATE.lookup_tensor(lhs) {
        Some(tensor) => tensor,
        None => return invalid_argument(),
    };
    let rhs = match STATE.lookup_tensor(rhs) {
        Some(tensor) => tensor,
        None => return invalid_argument(),
    };

    if lhs.device_handle != rhs.device_handle {
        return invalid_argument();
    }

    let tensor = match tensor_add_impl(&lhs.tensor, &rhs.tensor) {
        Ok(tensor) => tensor,
        Err(status) => return status,
    };

    unsafe {
        *out_tensor = STATE.insert_tensor(DeviceHandle(lhs.device_handle), tensor);
    }
    ok()
}

unsafe extern "C" fn release_tensor(tensor: TensorHandle) -> PluginStatus {
    STATE.release_tensor(tensor);
    ok()
}

unsafe extern "C" fn release_f32_buffer(buffer: OwnedF32Buffer) -> PluginStatus {
    if !buffer.ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
        }
    }
    ok()
}

unsafe extern "C" fn release_usize_buffer(buffer: OwnedUsizeBuffer) -> PluginStatus {
    if !buffer.ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
        }
    }
    ok()
}

static PLUGIN: BackendPluginV1 = BackendPluginV1 {
    abi_version: BACKEND_PLUGIN_ABI_VERSION,
    backend_name,
    seed,
    sync,
    device_count,
};

static TENSOR_OPS: BackendTensorOpsV1 = BackendTensorOpsV1 {
    abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
    create_device,
    release_device,
    tensor_from_f32_data,
    tensor_into_f32_data,
    tensor_shape,
    tensor_add,
    release_tensor,
    release_f32_buffer,
    release_usize_buffer,
};

export_backend_plugin_v1!(PLUGIN);
export_backend_tensor_ops_v1!(TENSOR_OPS);
