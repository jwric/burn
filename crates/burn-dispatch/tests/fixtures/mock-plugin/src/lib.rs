use burn_dylib::{
    BACKEND_PLUGIN_ABI_VERSION, BACKEND_TENSOR_OPS_ABI_VERSION, BackendPluginV1,
    BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer, OwnedUsizeBuffer, PluginStatus,
    PluginStatusCode, TensorBinaryOp, TensorHandle, TensorShapeRef, export_backend_plugin_v1,
    export_backend_tensor_ops_v1,
};
use std::collections::HashMap;
use std::ffi::c_char;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

#[cfg(not(feature = "variant-b"))]
const BACKEND_NAME_A: &[u8] = b"mock-plugin-a\0";
#[cfg(feature = "variant-b")]
const BACKEND_NAME_B: &[u8] = b"mock-plugin-b\0";
const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const ERR_FAILED: &[u8] = b"operation failed\0";

#[derive(Clone)]
struct TensorState {
    device_id: u64,
    shape: Vec<usize>,
    data: Vec<f32>,
}

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

static DEVICES: LazyLock<Mutex<HashMap<u64, ()>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static TENSORS: LazyLock<Mutex<HashMap<u64, TensorState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn ok() -> PluginStatus {
    PluginStatus::ok()
}

fn invalid_argument() -> PluginStatus {
    PluginStatus::error(
        PluginStatusCode::InvalidArgument,
        ERR_INVALID_ARGUMENT.as_ptr().cast(),
    )
}

fn failed() -> PluginStatus {
    PluginStatus::error(PluginStatusCode::Failed, ERR_FAILED.as_ptr().cast())
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

fn checked_numel(shape: &[usize]) -> Result<usize, PluginStatus> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(failed)
}

fn create_tensor(
    device_id: u64,
    shape: Vec<usize>,
    data: Vec<f32>,
) -> Result<TensorHandle, PluginStatus> {
    let numel = checked_numel(&shape)?;
    if numel != data.len() {
        return Err(invalid_argument());
    }

    let tensor_id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
    let state = TensorState {
        device_id,
        shape,
        data,
    };

    TENSORS
        .lock()
        .expect("tensor lock")
        .insert(tensor_id, state);

    Ok(TensorHandle(tensor_id))
}

fn lookup_tensor(tensor: TensorHandle) -> Result<TensorState, PluginStatus> {
    TENSORS
        .lock()
        .expect("tensor lock")
        .get(&tensor.0)
        .cloned()
        .ok_or_else(invalid_argument)
}

fn lookup_binary_tensors(
    lhs: TensorHandle,
    rhs: TensorHandle,
) -> Result<(TensorState, TensorState), PluginStatus> {
    let guard = TENSORS.lock().expect("tensor lock");
    let lhs_state = guard.get(&lhs.0).cloned().ok_or_else(invalid_argument)?;
    let rhs_state = guard.get(&rhs.0).cloned().ok_or_else(invalid_argument)?;
    Ok((lhs_state, rhs_state))
}

fn tensor_add_impl(
    lhs_state: &TensorState,
    rhs_state: &TensorState,
) -> Result<(Vec<usize>, Vec<f32>), PluginStatus> {
    if lhs_state.device_id != rhs_state.device_id || lhs_state.shape != rhs_state.shape {
        return Err(invalid_argument());
    }

    let bias = if cfg!(feature = "variant-b") { 1.0 } else { 0.0 };
    let out_data = lhs_state
        .data
        .iter()
        .zip(rhs_state.data.iter())
        .map(|(l, r)| l + r + bias)
        .collect::<Vec<_>>();

    Ok((lhs_state.shape.clone(), out_data))
}

fn tensor_matmul_impl(
    lhs_state: &TensorState,
    rhs_state: &TensorState,
) -> Result<(Vec<usize>, Vec<f32>), PluginStatus> {
    if lhs_state.device_id != rhs_state.device_id {
        return Err(invalid_argument());
    }

    if lhs_state.shape.len() != 2 || rhs_state.shape.len() != 2 {
        return Err(invalid_argument());
    }

    let m = lhs_state.shape[0];
    let k = lhs_state.shape[1];
    let rhs_k = rhs_state.shape[0];
    let n = rhs_state.shape[1];

    if k != rhs_k {
        return Err(invalid_argument());
    }

    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                let lhs_idx = row * k + inner;
                let rhs_idx = inner * n + col;
                acc += lhs_state.data[lhs_idx] * rhs_state.data[rhs_idx];
            }
            out[row * n + col] = acc;
        }
    }

    Ok((vec![m, n], out))
}

unsafe extern "C" fn backend_name() -> *const c_char {
    #[cfg(feature = "variant-b")]
    {
        return BACKEND_NAME_B.as_ptr().cast();
    }

    #[cfg(not(feature = "variant-b"))]
    {
        BACKEND_NAME_A.as_ptr().cast()
    }
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
    _ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    if out_device.is_null() {
        return invalid_argument();
    }

    let device_id = NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed);
    DEVICES.lock().expect("device lock").insert(device_id, ());

    unsafe {
        *out_device = DeviceHandle(device_id);
    }

    ok()
}

unsafe extern "C" fn release_device(device: DeviceHandle) -> PluginStatus {
    DEVICES.lock().expect("device lock").remove(&device.0);
    TENSORS
        .lock()
        .expect("tensor lock")
        .retain(|_, tensor| tensor.device_id != device.0);
    ok()
}

unsafe extern "C" fn tensor_from_f32_data(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    if out_tensor.is_null() {
        return invalid_argument();
    }

    if !DEVICES.lock().expect("device lock").contains_key(&device.0) {
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

    let handle = match create_tensor(device.0, shape, data) {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    unsafe {
        *out_tensor = handle;
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

    let state = match lookup_tensor(tensor) {
        Ok(state) => state,
        Err(status) => return status,
    };

    let mut values = state.data;
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

    let state = match lookup_tensor(tensor) {
        Ok(state) => state,
        Err(status) => return status,
    };

    let mut dims = state.shape;
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

unsafe extern "C" fn tensor_binary(
    op: TensorBinaryOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    if out_tensor.is_null() {
        return invalid_argument();
    }

    let (lhs_state, rhs_state) = match lookup_binary_tensors(lhs, rhs) {
        Ok(pair) => pair,
        Err(status) => return status,
    };

    let (shape, data) = match op {
        TensorBinaryOp::Add => match tensor_add_impl(&lhs_state, &rhs_state) {
            Ok(output) => output,
            Err(status) => return status,
        },
        TensorBinaryOp::Matmul => match tensor_matmul_impl(&lhs_state, &rhs_state) {
            Ok(output) => output,
            Err(status) => return status,
        },
    };

    let out = match create_tensor(lhs_state.device_id, shape, data) {
        Ok(handle) => handle,
        Err(status) => return status,
    };

    unsafe {
        *out_tensor = out;
    }
    ok()
}

unsafe extern "C" fn release_tensor(tensor: TensorHandle) -> PluginStatus {
    TENSORS.lock().expect("tensor lock").remove(&tensor.0);
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
    tensor_binary,
    release_tensor,
    release_f32_buffer,
    release_usize_buffer,
};

export_backend_plugin_v1!(PLUGIN);
export_backend_tensor_ops_v1!(TENSOR_OPS);
