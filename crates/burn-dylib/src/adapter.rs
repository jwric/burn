use burn_backend::{
    Backend, BoolDType, DType, Device as BurnDevice, DeviceId, Distribution, FloatDType, IntDType,
    Scalar, Shape, Slice, TensorData, TensorMetadata,
    ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConvOptions,
        InterpolateMode, InterpolateOptions, TransactionPrimitive, UnfoldOptions,
    },
    quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
        QuantizationParametersPrimitive,
    },
};

use crate::{
    ABI_QUANT_BLOCK_MAX_DIMS, AbiAttentionModuleOptions, AbiBoolDType, AbiConvOptions1,
    AbiConvOptions2, AbiConvOptions3, AbiConvTransposeOptions1, AbiConvTransposeOptions2,
    AbiConvTransposeOptions3, AbiDeformConv2dBackward, AbiDeformConvOptions2, AbiDistribution,
    AbiDistributionKind, AbiFloatDType, AbiIntDType, AbiInterpolateMode, AbiInterpolateOptions,
    AbiMaxPool1dWithIndices, AbiMaxPool2dWithIndices, AbiQuantLevel, AbiQuantMode,
    AbiQuantParam, AbiQuantScheme, AbiQuantStore, AbiQuantValue, AbiRfftOutput, AbiScalar,
    AbiScalarKind, AbiSliceRef, AbiTensorWithIndices, AbiUnfoldOptions,
    BACKEND_PLUGIN_ABI_VERSION,
    BACKEND_TENSOR_OPS_ABI_VERSION, BackendNameFn, BackendPluginV1, BackendTensorOpsV1,
    DeviceHandle, F32SliceRef, OwnedF32Buffer, OwnedQTransactionItem, OwnedU8Buffer,
    OwnedU64Buffer, OwnedUsizeBuffer, PluginStatus, PluginStatusCode, TensorHandle,
    TensorHandleRef, TensorShapeRef, U8SliceRef, U64SliceRef,
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
    quantized_tensors: Mutex<HashMap<u64, TensorState<P::QuantizedTensorPrimitive>>>,
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
            quantized_tensors: Mutex::new(HashMap::new()),
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
        self.quantized_tensors
            .lock()
            .expect("quantized tensor lock")
            .clear();
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

    fn lookup_quantized(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::QuantizedTensorPrimitive>, PluginStatus> {
        self.quantized_tensors
            .lock()
            .expect("quantized tensor lock")
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

        if let Some(state) = self
            .quantized_tensors
            .lock()
            .expect("quantized tensor lock")
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

    fn insert_quantized(
        &self,
        device_handle: DeviceHandle,
        tensor: P::QuantizedTensorPrimitive,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.quantized_tensors
            .lock()
            .expect("quantized tensor lock")
            .insert(
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
        self.quantized_tensors
            .lock()
            .expect("quantized tensor lock")
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
        self.quantized_tensors
            .lock()
            .expect("quantized tensor lock")
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

fn optional_float_state<P: Backend>(
    handle: TensorHandle,
) -> Result<Option<TensorState<P::FloatTensorPrimitive>>, PluginStatus> {
    if !handle.is_valid() {
        return Ok(None);
    }
    adapter_state::<P>().lookup_float(handle).map(Some)
}

fn optional_bool_state<P: Backend>(
    handle: TensorHandle,
) -> Result<Option<TensorState<P::BoolTensorPrimitive>>, PluginStatus> {
    if !handle.is_valid() {
        return Ok(None);
    }
    adapter_state::<P>().lookup_bool(handle).map(Some)
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
    // catch and rethrow panics to prevent unwinding across the FFI boundary, which is undefined behavior.
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

fn try_u64_data(data: U64SliceRef) -> Result<Vec<u64>, PluginStatus> {
    if data.len == 0 {
        return Ok(Vec::new());
    }
    if data.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(data.ptr, data.len) };
    Ok(values.to_vec())
}

fn try_u8_data(data: U8SliceRef) -> Result<Vec<u8>, PluginStatus> {
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

fn try_tensor_handles(handles: TensorHandleRef) -> Result<Vec<TensorHandle>, PluginStatus> {
    if handles.len == 0 {
        return Ok(Vec::new());
    }
    if handles.ptr.is_null() {
        return Err(invalid_argument());
    }

    Ok(unsafe { slice::from_raw_parts(handles.ptr, handles.len) }.to_vec())
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

fn quant_value_from_abi(value: AbiQuantValue) -> QuantValue {
    match value {
        AbiQuantValue::Q8F => QuantValue::Q8F,
        AbiQuantValue::E5M2 => QuantValue::E5M2,
        AbiQuantValue::E4M3 => QuantValue::E4M3,
        AbiQuantValue::Q4F => QuantValue::Q4F,
        AbiQuantValue::E2M1 => QuantValue::E2M1,
        AbiQuantValue::Q2F => QuantValue::Q2F,
        AbiQuantValue::Q8S => QuantValue::Q8S,
        AbiQuantValue::Q4S => QuantValue::Q4S,
        AbiQuantValue::Q2S => QuantValue::Q2S,
    }
}

fn quant_param_from_abi(value: AbiQuantParam) -> QuantParam {
    match value {
        AbiQuantParam::F32 => QuantParam::F32,
        AbiQuantParam::F16 => QuantParam::F16,
        AbiQuantParam::BF16 => QuantParam::BF16,
        AbiQuantParam::UE8M0 => QuantParam::UE8M0,
        AbiQuantParam::UE4M3 => QuantParam::UE4M3,
    }
}

fn quant_store_from_abi(store: AbiQuantStore, packed_dim: usize) -> QuantStore {
    match store {
        AbiQuantStore::Native => QuantStore::Native,
        AbiQuantStore::PackedNative => QuantStore::PackedNative(packed_dim),
        AbiQuantStore::PackedU32 => QuantStore::PackedU32(packed_dim),
    }
}

fn quant_mode_from_abi(mode: AbiQuantMode) -> QuantMode {
    match mode {
        AbiQuantMode::Symmetric => QuantMode::Symmetric,
    }
}

fn quant_scheme_from_abi(value: AbiQuantScheme) -> Result<QuantScheme, PluginStatus> {
    let level = match value.level {
        AbiQuantLevel::Tensor => QuantLevel::Tensor,
        AbiQuantLevel::Block => {
            if value.block_rank == 0 || value.block_rank > ABI_QUANT_BLOCK_MAX_DIMS {
                return Err(invalid_argument());
            }
            QuantLevel::Block(BlockSize::new(&value.block_dims[..value.block_rank]))
        }
    };

    Ok(QuantScheme {
        value: quant_value_from_abi(value.value),
        param: quant_param_from_abi(value.param),
        store: quant_store_from_abi(value.store, value.store_packed_dim),
        level,
        mode: quant_mode_from_abi(value.mode),
    })
}

fn quant_value_to_abi(value: QuantValue) -> AbiQuantValue {
    match value {
        QuantValue::Q8F => AbiQuantValue::Q8F,
        QuantValue::E5M2 => AbiQuantValue::E5M2,
        QuantValue::E4M3 => AbiQuantValue::E4M3,
        QuantValue::Q4F => AbiQuantValue::Q4F,
        QuantValue::E2M1 => AbiQuantValue::E2M1,
        QuantValue::Q2F => AbiQuantValue::Q2F,
        QuantValue::Q8S => AbiQuantValue::Q8S,
        QuantValue::Q4S => AbiQuantValue::Q4S,
        QuantValue::Q2S => AbiQuantValue::Q2S,
    }
}

fn quant_param_to_abi(param: QuantParam) -> AbiQuantParam {
    match param {
        QuantParam::F32 => AbiQuantParam::F32,
        QuantParam::F16 => AbiQuantParam::F16,
        QuantParam::BF16 => AbiQuantParam::BF16,
        QuantParam::UE8M0 => AbiQuantParam::UE8M0,
        QuantParam::UE4M3 => AbiQuantParam::UE4M3,
    }
}

fn quant_store_to_abi(store: QuantStore) -> (AbiQuantStore, usize) {
    match store {
        QuantStore::Native => (AbiQuantStore::Native, 0),
        QuantStore::PackedNative(dim) => (AbiQuantStore::PackedNative, dim),
        QuantStore::PackedU32(dim) => (AbiQuantStore::PackedU32, dim),
    }
}

fn quant_mode_to_abi(mode: QuantMode) -> AbiQuantMode {
    match mode {
        QuantMode::Symmetric => AbiQuantMode::Symmetric,
    }
}

fn quant_scheme_to_abi(scheme: QuantScheme) -> AbiQuantScheme {
    let (store, store_packed_dim) = quant_store_to_abi(scheme.store);
    let (level, block_dims, block_rank) = match scheme.level {
        QuantLevel::Tensor => (AbiQuantLevel::Tensor, [1; ABI_QUANT_BLOCK_MAX_DIMS], 0),
        QuantLevel::Block(block_size) => {
            let mut block_dims = [1; ABI_QUANT_BLOCK_MAX_DIMS];
            let block_slice = block_size.as_slice();
            let block_rank = block_slice.len().min(ABI_QUANT_BLOCK_MAX_DIMS);
            block_dims[..block_rank].copy_from_slice(&block_slice[..block_rank]);
            (AbiQuantLevel::Block, block_dims, block_rank)
        }
    };

    AbiQuantScheme {
        value: quant_value_to_abi(scheme.value),
        param: quant_param_to_abi(scheme.param),
        store,
        store_packed_dim,
        level,
        block_dims,
        block_rank,
        mode: quant_mode_to_abi(scheme.mode),
    }
}

fn conv_options_1_from_abi(options: AbiConvOptions1) -> ConvOptions<1> {
    ConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.groups,
    )
}

fn conv_options_2_from_abi(options: AbiConvOptions2) -> ConvOptions<2> {
    ConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.groups,
    )
}

fn conv_options_3_from_abi(options: AbiConvOptions3) -> ConvOptions<3> {
    ConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.groups,
    )
}

fn deform_conv_options_2_from_abi(options: AbiDeformConvOptions2) -> DeformConvOptions<2> {
    DeformConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.weight_groups,
        options.offset_groups,
    )
}

fn conv_transpose_options_2_from_abi(options: AbiConvTransposeOptions2) -> ConvTransposeOptions<2> {
    ConvTransposeOptions::new(
        options.stride,
        options.padding,
        options.padding_out,
        options.dilation,
        options.groups,
    )
}

fn conv_transpose_options_1_from_abi(
    options: AbiConvTransposeOptions1,
) -> ConvTransposeOptions<1> {
    ConvTransposeOptions::new(
        options.stride,
        options.padding,
        options.padding_out,
        options.dilation,
        options.groups,
    )
}

fn conv_transpose_options_3_from_abi(options: AbiConvTransposeOptions3) -> ConvTransposeOptions<3> {
    ConvTransposeOptions::new(
        options.stride,
        options.padding,
        options.padding_out,
        options.dilation,
        options.groups,
    )
}

fn unfold_options_from_abi(options: AbiUnfoldOptions) -> UnfoldOptions {
    UnfoldOptions::new(options.stride, options.padding, options.dilation)
}

fn interpolate_mode_from_abi(mode: AbiInterpolateMode) -> InterpolateMode {
    match mode {
        AbiInterpolateMode::Nearest => InterpolateMode::Nearest,
        AbiInterpolateMode::Bilinear => InterpolateMode::Bilinear,
        AbiInterpolateMode::Bicubic => InterpolateMode::Bicubic,
        AbiInterpolateMode::Lanczos3 => InterpolateMode::Lanczos3,
    }
}

fn interpolate_options_from_abi(options: AbiInterpolateOptions) -> InterpolateOptions {
    InterpolateOptions {
        mode: interpolate_mode_from_abi(options.mode),
        align_corners: options.align_corners != 0,
    }
}

fn attention_options_from_abi(options: AbiAttentionModuleOptions) -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: if options.has_scale == 0 {
            None
        } else {
            Some(options.scale)
        },
        softcap: if options.has_softcap == 0 {
            None
        } else {
            Some(options.softcap)
        },
        is_causal: options.is_causal != 0,
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

unsafe extern "C" fn abi_create_default_device<P: Backend>(
    out_type_id: *mut u16,
    out_ordinal: *mut usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_type_id.is_null() || out_ordinal.is_null() || out_device.is_null() {
            return invalid_argument();
        }

        let device = P::Device::default();
        let device_id = device.to_id();
        let handle = adapter_state::<P>().insert_device(device);

        unsafe {
            *out_type_id = device_id.type_id;
            *out_ordinal = device_id.index_id as _;
            *out_device = handle;
        }
        ok()
    })
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

unsafe extern "C" fn abi_float_tensor_zeros<P: Backend>(
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

        let out = P::float_zeros(shape, &device_state, float_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_float(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_ones<P: Backend>(
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

        let out = P::float_ones(shape, &device_state, float_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_float(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_full<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    value: AbiScalar,
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

        let out = P::float_full(
            shape,
            scalar_from_abi(value),
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

macro_rules! abi_float_repeat_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            times: usize,
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

                let out = P::$backend_fn(tensor_state.tensor, dim, times);
                let handle =
                    adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_clamp_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            min: AbiScalar,
            max: AbiScalar,
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

                let out = P::$backend_fn(tensor_state.tensor, scalar_from_abi(min), scalar_from_abi(max));
                let handle =
                    adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_bool_reduce_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
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

                let out = P::$backend_fn(tensor_state.tensor, bool_dtype_from_abi(out_dtype));
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

macro_rules! abi_float_bool_reduce_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
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
                    dim,
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

macro_rules! abi_float_with_indices_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            out_dtype: AbiIntDType,
            out_tensors: *mut AbiTensorWithIndices,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensors.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let (values, indices) =
                    P::$backend_fn(tensor_state.tensor, dim, int_dtype_from_abi(out_dtype));
                let values =
                    adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), values);
                let indices =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), indices);

                unsafe {
                    *out_tensors = AbiTensorWithIndices { values, indices };
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_sort_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
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

                let out = P::$backend_fn(tensor_state.tensor, dim, descending != 0);
                let handle =
                    adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_sort_with_indices_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
            out_dtype: AbiIntDType,
            out_tensors: *mut AbiTensorWithIndices,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensors.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_float(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let (values, indices) = P::$backend_fn(
                    tensor_state.tensor,
                    dim,
                    descending != 0,
                    int_dtype_from_abi(out_dtype),
                );
                let values =
                    adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), values);
                let indices =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), indices);

                unsafe {
                    *out_tensors = AbiTensorWithIndices { values, indices };
                }
                ok()
            })
        }
    };
}

macro_rules! abi_float_argsort_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
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

                let out = P::$backend_fn(
                    tensor_state.tensor,
                    dim,
                    descending != 0,
                    int_dtype_from_abi(out_dtype),
                );
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
abi_float_unary_fn!(abi_float_tensor_prod, float_prod);
abi_float_dim_fn!(abi_float_tensor_prod_dim, float_prod_dim);
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
abi_float_repeat_dim_fn!(abi_float_tensor_repeat_dim, float_repeat_dim);
abi_float_scalar_fn!(abi_float_tensor_clamp_min, float_clamp_min);
abi_float_scalar_fn!(abi_float_tensor_clamp_max, float_clamp_max);
abi_float_clamp_fn!(abi_float_tensor_clamp, float_clamp);
abi_float_unary_fn!(abi_float_tensor_neg, float_neg);
abi_float_unary_fn!(abi_float_tensor_transpose, float_transpose);
abi_float_compare_binary_fn!(abi_float_tensor_not_equal, float_not_equal);
abi_float_compare_scalar_fn!(abi_float_tensor_not_equal_elem, float_not_equal_elem);
abi_float_unary_fn!(abi_float_tensor_mean, float_mean);
abi_float_scalar_fn!(abi_float_tensor_powi_scalar, float_powi_scalar_impl);
abi_float_unary_fn!(abi_float_tensor_max, float_max);
abi_float_dim_fn!(abi_float_tensor_max_dim, float_max_dim);
abi_float_with_indices_fn!(abi_float_tensor_max_dim_with_indices, float_max_dim_with_indices);
abi_float_unary_fn!(abi_float_tensor_min, float_min);
abi_float_dim_fn!(abi_float_tensor_min_dim, float_min_dim);
abi_float_with_indices_fn!(abi_float_tensor_min_dim_with_indices, float_min_dim_with_indices);
abi_float_unary_fn!(abi_float_tensor_max_abs, float_max_abs);
abi_float_dim_fn!(abi_float_tensor_max_abs_dim, float_max_abs_dim);
abi_float_bool_reduce_fn!(abi_float_tensor_any, float_any);
abi_float_bool_reduce_dim_fn!(abi_float_tensor_any_dim, float_any_dim);
abi_float_bool_reduce_fn!(abi_float_tensor_all, float_all);
abi_float_bool_reduce_dim_fn!(abi_float_tensor_all_dim, float_all_dim);
abi_float_unary_fn!(abi_float_tensor_sign, float_sign);
abi_float_sort_fn!(abi_float_tensor_sort, float_sort);
abi_float_sort_with_indices_fn!(abi_float_tensor_sort_with_indices, float_sort_with_indices);
abi_float_argsort_fn!(abi_float_tensor_argsort, float_argsort);
abi_float_bool_reduce_fn!(abi_float_tensor_is_nan, float_is_nan);
abi_float_bool_reduce_fn!(abi_float_tensor_is_inf, float_is_inf);
abi_float_scalar_fn!(abi_activation_leaky_relu, leaky_relu);
abi_float_unary_fn!(abi_activation_relu, relu);
abi_float_binary_fn!(abi_activation_relu_backward, relu_backward);
abi_float_unary_fn!(abi_activation_gelu, gelu);
abi_float_binary_fn!(abi_activation_prelu, prelu);
abi_float_binary_fn!(abi_activation_gelu_backward, gelu_backward);
abi_float_unary_fn!(abi_activation_sigmoid, sigmoid);
abi_float_binary_fn!(abi_activation_sigmoid_backward, sigmoid_backward);
abi_float_unary_fn!(abi_activation_log_sigmoid, log_sigmoid);
abi_float_binary_fn!(abi_activation_log_sigmoid_backward, log_sigmoid_backward);

unsafe extern "C" fn abi_activation_hard_sigmoid<P: Backend>(
    tensor: TensorHandle,
    alpha: AbiScalar,
    beta: AbiScalar,
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

        let out = P::hard_sigmoid(
            tensor_state.tensor,
            scalar_from_abi(alpha),
            scalar_from_abi(beta),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_powi<P: Backend>(
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
        let rhs_state = match adapter_state::<P>().lookup_int(rhs) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if lhs_state.device_handle != rhs_state.device_handle {
            return invalid_argument();
        }

        let out = P::float_powi(lhs_state.tensor, rhs_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(lhs_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_float_tensor_cat<P: Backend>(
    tensors: TensorHandleRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handles = match try_tensor_handles(tensors) {
            Ok(handles) => handles,
            Err(status) => return status,
        };
        if handles.is_empty() {
            return invalid_argument();
        }

        let first_state = match adapter_state::<P>().lookup_float(handles[0]) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_handle = first_state.device_handle;

        let mut tensors = Vec::with_capacity(handles.len());
        tensors.push(first_state.tensor);

        for &handle in handles.iter().skip(1) {
            let state = match adapter_state::<P>().lookup_float(handle) {
                Ok(state) => state,
                Err(status) => return status,
            };

            if state.device_handle != device_handle {
                return invalid_argument();
            }

            tensors.push(state.tensor);
        }

        let out = P::float_cat(tensors, dim);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

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

unsafe extern "C" fn abi_int_tensor_from_u64_data<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U64SliceRef,
    dtype: AbiIntDType,
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
        let values = match try_u64_data(data) {
            Ok(values) => values,
            Err(status) => return status,
        };

        let dtype = int_dtype_from_abi(dtype);
        let data = match dtype {
            IntDType::I64 => {
                let values = values.into_iter().map(|v| v as i64).collect::<Vec<i64>>();
                TensorData::new(values, shape)
            }
            IntDType::I32 => {
                let values = values.into_iter().map(|v| v as i32).collect::<Vec<i32>>();
                TensorData::new(values, shape)
            }
            IntDType::I16 => {
                let values = values.into_iter().map(|v| v as i16).collect::<Vec<i16>>();
                TensorData::new(values, shape)
            }
            IntDType::I8 => {
                let values = values.into_iter().map(|v| v as i8).collect::<Vec<i8>>();
                TensorData::new(values, shape)
            }
            IntDType::U64 => TensorData::new(values, shape),
            IntDType::U32 => {
                let values = values.into_iter().map(|v| v as u32).collect::<Vec<u32>>();
                TensorData::new(values, shape)
            }
            IntDType::U16 => {
                let values = values.into_iter().map(|v| v as u16).collect::<Vec<u16>>();
                TensorData::new(values, shape)
            }
            IntDType::U8 => {
                let values = values.into_iter().map(|v| v as u8).collect::<Vec<u8>>();
                TensorData::new(values, shape)
            }
        };

        let tensor = P::int_from_data(data, &device_state);
        let handle = adapter_state::<P>().insert_int(device, tensor);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_into_u64_data<P: Backend>(
    tensor: TensorHandle,
    out_data: *mut OwnedU64Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let data = match burn_backend::read_sync(P::int_into_data(tensor_state.tensor)) {
            Ok(data) => data,
            Err(_) => return execution_error(),
        };

        let mut values = match data.dtype {
            burn_backend::DType::I64 => match data.into_vec::<i64>() {
                Ok(values) => values.into_iter().map(|v| v as u64).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::I32 => match data.into_vec::<i32>() {
                Ok(values) => values.into_iter().map(|v| v as u64).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::I16 => match data.into_vec::<i16>() {
                Ok(values) => values.into_iter().map(|v| v as u64).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::I8 => match data.into_vec::<i8>() {
                Ok(values) => values.into_iter().map(|v| v as u64).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::U64 => match data.into_vec::<u64>() {
                Ok(values) => values,
                Err(_) => return execution_error(),
            },
            burn_backend::DType::U32 => match data.into_vec::<u32>() {
                Ok(values) => values.into_iter().map(u64::from).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::U16 => match data.into_vec::<u16>() {
                Ok(values) => values.into_iter().map(u64::from).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            burn_backend::DType::U8 => match data.into_vec::<u8>() {
                Ok(values) => values.into_iter().map(u64::from).collect::<Vec<u64>>(),
                Err(_) => return execution_error(),
            },
            _ => return execution_error(),
        };

        let buffer = OwnedU64Buffer {
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

unsafe extern "C" fn abi_bool_tensor_from_u8_data<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U8SliceRef,
    dtype: AbiBoolDType,
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
        let values = match try_u8_data(data) {
            Ok(values) => values,
            Err(status) => return status,
        };

        let dtype = bool_dtype_from_abi(dtype);
        let bool_values = values.into_iter().map(|v| v != 0).collect::<Vec<bool>>();
        let data = TensorData::new(bool_values, shape).convert_dtype(dtype.into());
        let tensor = P::bool_from_data(data, &device_state);
        let handle = adapter_state::<P>().insert_bool(device, tensor);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_into_u8_data<P: Backend>(
    tensor: TensorHandle,
    out_data: *mut OwnedU8Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let data = match burn_backend::read_sync(P::bool_into_data(tensor_state.tensor)) {
            Ok(data) => data,
            Err(_) => return execution_error(),
        };
        let mut values = match data.convert::<u8>().into_vec::<u8>() {
            Ok(values) => values,
            Err(_) => return execution_error(),
        };

        let buffer = OwnedU8Buffer {
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

unsafe extern "C" fn abi_int_tensor_to_device<P: Backend>(
    tensor: TensorHandle,
    device: DeviceHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };

        let out = P::int_to_device(tensor_state.tensor, &device_state);
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_to_device<P: Backend>(
    tensor: TensorHandle,
    device: DeviceHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };

        let out = P::bool_to_device(tensor_state.tensor, &device_state);
        let handle = adapter_state::<P>().insert_bool(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_empty<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiIntDType,
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

        let out = P::int_empty(shape, &device_state, int_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_random<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    distribution: AbiDistribution,
    dtype: AbiIntDType,
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

        let out = P::int_random(
            shape,
            distribution_from_abi(distribution),
            &device_state,
            int_dtype_from_abi(dtype),
        );
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_empty<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiBoolDType,
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

        let out = P::bool_empty(shape, &device_state, bool_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_bool(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_zeros<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiBoolDType,
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

        let out = P::bool_zeros(shape, &device_state, bool_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_bool(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_ones<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiBoolDType,
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

        let out = P::bool_ones(shape, &device_state, bool_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_bool(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_into_float<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::int_into_float(tensor_state.tensor, float_dtype_from_abi(out_dtype));
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_cast<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::int_cast(tensor_state.tensor, int_dtype_from_abi(out_dtype));
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_into_int<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::bool_into_int(tensor_state.tensor, int_dtype_from_abi(out_dtype));
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_into_float<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::bool_into_float(tensor_state.tensor, float_dtype_from_abi(out_dtype));
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

macro_rules! abi_int_unary_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor);
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

macro_rules! abi_int_binary_fn {
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

                let lhs_state = match adapter_state::<P>().lookup_int(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs_state = match adapter_state::<P>().lookup_int(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                if lhs_state.device_handle != rhs_state.device_handle {
                    return invalid_argument();
                }

                let out = P::$backend_fn(lhs_state.tensor, rhs_state.tensor);
                let handle =
                    adapter_state::<P>().insert_int(DeviceHandle(lhs_state.device_handle), out);

                unsafe {
                    *out_tensor = handle;
                }
                ok()
            })
        }
    };
}

macro_rules! abi_int_scalar_fn {
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

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, scalar_from_abi(scalar));
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

macro_rules! abi_int_dim_fn {
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

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim);
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

macro_rules! abi_int_compare_binary_fn {
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

                let lhs_state = match adapter_state::<P>().lookup_int(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs_state = match adapter_state::<P>().lookup_int(rhs) {
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

macro_rules! abi_int_compare_scalar_fn {
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

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
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

abi_int_binary_fn!(abi_int_tensor_add, int_add);
abi_int_scalar_fn!(abi_int_tensor_add_scalar, int_add_scalar);
abi_int_binary_fn!(abi_int_tensor_sub, int_sub);
abi_int_scalar_fn!(abi_int_tensor_sub_scalar, int_sub_scalar);
abi_int_binary_fn!(abi_int_tensor_mul, int_mul);
abi_int_scalar_fn!(abi_int_tensor_mul_scalar, int_mul_scalar);
abi_int_binary_fn!(abi_int_tensor_div, int_div);
abi_int_scalar_fn!(abi_int_tensor_div_scalar, int_div_scalar);
abi_int_binary_fn!(abi_int_tensor_remainder, int_remainder);
abi_int_scalar_fn!(abi_int_tensor_remainder_scalar, int_remainder_scalar);
abi_int_binary_fn!(abi_int_tensor_matmul, int_matmul);
abi_int_unary_fn!(abi_int_tensor_abs, int_abs);
abi_int_unary_fn!(abi_int_tensor_sum, int_sum);
abi_int_dim_fn!(abi_int_tensor_sum_dim, int_sum_dim);
abi_int_unary_fn!(abi_int_tensor_prod, int_prod);
abi_int_dim_fn!(abi_int_tensor_prod_dim, int_prod_dim);
abi_int_dim_fn!(abi_int_tensor_mean_dim, int_mean_dim);
abi_int_dim_fn!(abi_int_tensor_cumsum, int_cumsum);
abi_int_dim_fn!(abi_int_tensor_cumprod, int_cumprod);
abi_int_dim_fn!(abi_int_tensor_cummin, int_cummin);
abi_int_dim_fn!(abi_int_tensor_cummax, int_cummax);
abi_int_dim_fn!(abi_int_tensor_argmax, int_argmax);
abi_int_dim_fn!(abi_int_tensor_argmin, int_argmin);
abi_int_compare_binary_fn!(abi_int_tensor_equal, int_equal);
abi_int_compare_scalar_fn!(abi_int_tensor_equal_elem, int_equal_elem);
abi_int_compare_binary_fn!(abi_int_tensor_greater, int_greater);
abi_int_compare_scalar_fn!(abi_int_tensor_greater_elem, int_greater_elem);
abi_int_compare_binary_fn!(abi_int_tensor_greater_equal, int_greater_equal);
abi_int_compare_scalar_fn!(abi_int_tensor_greater_equal_elem, int_greater_equal_elem);
abi_int_compare_binary_fn!(abi_int_tensor_lower, int_lower);
abi_int_compare_scalar_fn!(abi_int_tensor_lower_elem, int_lower_elem);
abi_int_compare_binary_fn!(abi_int_tensor_lower_equal, int_lower_equal);
abi_int_compare_scalar_fn!(abi_int_tensor_lower_equal_elem, int_lower_equal_elem);
abi_int_binary_fn!(abi_int_tensor_bitwise_and, bitwise_and);
abi_int_scalar_fn!(abi_int_tensor_bitwise_and_scalar, bitwise_and_scalar);
abi_int_binary_fn!(abi_int_tensor_bitwise_or, bitwise_or);
abi_int_scalar_fn!(abi_int_tensor_bitwise_or_scalar, bitwise_or_scalar);
abi_int_binary_fn!(abi_int_tensor_bitwise_xor, bitwise_xor);
abi_int_scalar_fn!(abi_int_tensor_bitwise_xor_scalar, bitwise_xor_scalar);
abi_int_unary_fn!(abi_int_tensor_bitwise_not, bitwise_not);
abi_int_binary_fn!(abi_int_tensor_bitwise_left_shift, bitwise_left_shift);
abi_int_scalar_fn!(
    abi_int_tensor_bitwise_left_shift_scalar,
    bitwise_left_shift_scalar
);
abi_int_binary_fn!(abi_int_tensor_bitwise_right_shift, bitwise_right_shift);
abi_int_scalar_fn!(
    abi_int_tensor_bitwise_right_shift_scalar,
    bitwise_right_shift_scalar
);

macro_rules! abi_int_repeat_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            times: usize,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim, times);
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

macro_rules! abi_int_clamp_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            min: AbiScalar,
            max: AbiScalar,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, scalar_from_abi(min), scalar_from_abi(max));
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

macro_rules! abi_int_bool_reduce_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            out_dtype: AbiBoolDType,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, bool_dtype_from_abi(out_dtype));
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

macro_rules! abi_int_bool_reduce_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            out_dtype: AbiBoolDType,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(
                    tensor_state.tensor,
                    dim,
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

macro_rules! abi_int_with_indices_no_dtype_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            out_tensors: *mut AbiTensorWithIndices,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensors.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let (values, indices) = P::$backend_fn(tensor_state.tensor, dim);
                let values =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), values);
                let indices =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), indices);

                unsafe {
                    *out_tensors = AbiTensorWithIndices { values, indices };
                }
                ok()
            })
        }
    };
}

macro_rules! abi_int_sort_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim, descending != 0);
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

macro_rules! abi_int_sort_with_indices_no_dtype_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
            out_tensors: *mut AbiTensorWithIndices,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensors.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let (values, indices) = P::$backend_fn(tensor_state.tensor, dim, descending != 0);
                let values =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), values);
                let indices =
                    adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), indices);

                unsafe {
                    *out_tensors = AbiTensorWithIndices { values, indices };
                }
                ok()
            })
        }
    };
}

macro_rules! abi_int_argsort_no_dtype_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            descending: u8,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim, descending != 0);
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

unsafe extern "C" fn abi_int_tensor_zeros<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiIntDType,
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

        let out = P::int_zeros(shape, &device_state, int_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_ones<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiIntDType,
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

        let out = P::int_ones(shape, &device_state, int_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_full<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    value: AbiScalar,
    dtype: AbiIntDType,
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

        let out = P::int_full(
            shape,
            scalar_from_abi(value),
            &device_state,
            int_dtype_from_abi(dtype),
        );
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_cat<P: Backend>(
    tensors: TensorHandleRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handles = match try_tensor_handles(tensors) {
            Ok(handles) => handles,
            Err(status) => return status,
        };
        if handles.is_empty() {
            return invalid_argument();
        }

        let first_state = match adapter_state::<P>().lookup_int(handles[0]) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_handle = first_state.device_handle;

        let mut tensors = Vec::with_capacity(handles.len());
        tensors.push(first_state.tensor);

        for &handle in handles.iter().skip(1) {
            let state = match adapter_state::<P>().lookup_int(handle) {
                Ok(state) => state,
                Err(status) => return status,
            };

            if state.device_handle != device_handle {
                return invalid_argument();
            }

            tensors.push(state.tensor);
        }

        let out = P::int_cat(tensors, dim);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_arange_step<P: Backend>(
    start: i64,
    end: i64,
    step: usize,
    device: DeviceHandle,
    dtype: AbiIntDType,
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
        let out = P::int_arange_step(start..end, step, &device_state, int_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_arange<P: Backend>(
    start: i64,
    end: i64,
    device: DeviceHandle,
    dtype: AbiIntDType,
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
        let out = P::int_arange(start..end, &device_state, int_dtype_from_abi(dtype));
        let handle = adapter_state::<P>().insert_int(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

abi_int_repeat_dim_fn!(abi_int_tensor_repeat_dim, int_repeat_dim);
abi_int_compare_binary_fn!(abi_int_tensor_not_equal, int_not_equal);
abi_int_compare_scalar_fn!(abi_int_tensor_not_equal_elem, int_not_equal_elem);
abi_int_binary_fn!(abi_int_tensor_powi, int_powi);
abi_int_scalar_fn!(abi_int_tensor_powi_scalar, int_powi_scalar_impl);
abi_int_scalar_fn!(abi_int_tensor_clamp_min, int_clamp_min);
abi_int_scalar_fn!(abi_int_tensor_clamp_max, int_clamp_max);
abi_int_clamp_fn!(abi_int_tensor_clamp, int_clamp);
abi_int_unary_fn!(abi_int_tensor_neg, int_neg);
abi_int_unary_fn!(abi_int_tensor_mean, int_mean);
abi_int_unary_fn!(abi_int_tensor_max, int_max);
abi_int_dim_fn!(abi_int_tensor_max_dim, int_max_dim);
abi_int_with_indices_no_dtype_fn!(abi_int_tensor_max_dim_with_indices, int_max_dim_with_indices);
abi_int_unary_fn!(abi_int_tensor_max_abs, int_max_abs);
abi_int_dim_fn!(abi_int_tensor_max_abs_dim, int_max_abs_dim);
abi_int_unary_fn!(abi_int_tensor_min, int_min);
abi_int_dim_fn!(abi_int_tensor_min_dim, int_min_dim);
abi_int_with_indices_no_dtype_fn!(abi_int_tensor_min_dim_with_indices, int_min_dim_with_indices);
abi_int_unary_fn!(abi_int_tensor_transpose, int_transpose);
abi_int_bool_reduce_fn!(abi_int_tensor_any, int_any);
abi_int_bool_reduce_dim_fn!(abi_int_tensor_any_dim, int_any_dim);
abi_int_bool_reduce_fn!(abi_int_tensor_all, int_all);
abi_int_bool_reduce_dim_fn!(abi_int_tensor_all_dim, int_all_dim);
abi_int_unary_fn!(abi_int_tensor_sign, int_sign);
abi_int_sort_fn!(abi_int_tensor_sort, int_sort);
abi_int_sort_with_indices_no_dtype_fn!(abi_int_tensor_sort_with_indices, int_sort_with_indices);
abi_int_argsort_no_dtype_fn!(abi_int_tensor_argsort, int_argsort);

macro_rules! abi_bool_unary_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor);
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

macro_rules! abi_bool_binary_fn {
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

                let lhs_state = match adapter_state::<P>().lookup_bool(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs_state = match adapter_state::<P>().lookup_bool(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                if lhs_state.device_handle != rhs_state.device_handle {
                    return invalid_argument();
                }

                let out = P::$backend_fn(lhs_state.tensor, rhs_state.tensor);
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

macro_rules! abi_bool_scalar_fn {
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

                let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, scalar_from_abi(scalar));
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

macro_rules! abi_bool_dim_fn {
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

                let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim);
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

macro_rules! abi_bool_repeat_dim_fn {
    ($fn_name:ident, $backend_fn:ident) => {
        unsafe extern "C" fn $fn_name<P: Backend>(
            tensor: TensorHandle,
            dim: usize,
            times: usize,
            out_tensor: *mut TensorHandle,
        ) -> PluginStatus {
            with_boundary(|| {
                if out_tensor.is_null() {
                    return invalid_argument();
                }

                let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };

                let out = P::$backend_fn(tensor_state.tensor, dim, times);
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

unsafe extern "C" fn abi_bool_tensor_cat<P: Backend>(
    tensors: TensorHandleRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handles = match try_tensor_handles(tensors) {
            Ok(handles) => handles,
            Err(status) => return status,
        };
        if handles.is_empty() {
            return invalid_argument();
        }

        let first_state = match adapter_state::<P>().lookup_bool(handles[0]) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_handle = first_state.device_handle;

        let mut tensors = Vec::with_capacity(handles.len());
        tensors.push(first_state.tensor);

        for &handle in handles.iter().skip(1) {
            let state = match adapter_state::<P>().lookup_bool(handle) {
                Ok(state) => state,
                Err(status) => return status,
            };

            if state.device_handle != device_handle {
                return invalid_argument();
            }

            tensors.push(state.tensor);
        }

        let out = P::bool_cat(tensors, dim);
        let handle = adapter_state::<P>().insert_bool(DeviceHandle(device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

abi_bool_binary_fn!(abi_bool_tensor_equal, bool_equal);
abi_bool_scalar_fn!(abi_bool_tensor_equal_elem, bool_equal_elem);
abi_bool_unary_fn!(abi_bool_tensor_not, bool_not);
abi_bool_binary_fn!(abi_bool_tensor_and, bool_and);
abi_bool_binary_fn!(abi_bool_tensor_or, bool_or);
abi_bool_repeat_dim_fn!(abi_bool_tensor_repeat_dim, bool_repeat_dim);
abi_bool_binary_fn!(abi_bool_tensor_not_equal, bool_not_equal);
abi_bool_scalar_fn!(abi_bool_tensor_not_equal_elem, bool_not_equal_elem);
abi_bool_binary_fn!(abi_bool_tensor_xor, bool_xor);
abi_bool_unary_fn!(abi_bool_tensor_transpose, bool_transpose);
abi_bool_unary_fn!(abi_bool_tensor_any, bool_any);
abi_bool_dim_fn!(abi_bool_tensor_any_dim, bool_any_dim);
abi_bool_unary_fn!(abi_bool_tensor_all, bool_all);
abi_bool_dim_fn!(abi_bool_tensor_all_dim, bool_all_dim);

unsafe extern "C" fn abi_int_tensor_swap_dims<P: Backend>(
    tensor: TensorHandle,
    dim1: usize,
    dim2: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::int_swap_dims(tensor_state.tensor, dim1, dim2);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_permute<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::int_permute(tensor_state.tensor, axes.as_slice());
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_flip<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::int_flip(tensor_state.tensor, axes.as_slice());
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_reshape<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::int_reshape(tensor_state.tensor, shape);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_gather<P: Backend>(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
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

        let out = P::int_gather(dim, tensor_state.tensor, indices_state.tensor);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_scatter_add<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_int(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::int_scatter_add(
            dim,
            tensor_state.tensor,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_select<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
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

        let out = P::int_select(tensor_state.tensor, dim, indices_state.tensor);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_select_add<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_int(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::int_select_add(
            tensor_state.tensor,
            dim,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_slice<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };

        let out = P::int_slice(tensor_state.tensor, &slices);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_slice_assign<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_int(value) {
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

        let out = P::int_slice_assign(tensor_state.tensor, &slices, value_state.tensor);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_mask_where<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match adapter_state::<P>().lookup_bool(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_int(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != mask_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::int_mask_where(tensor_state.tensor, mask_state.tensor, value_state.tensor);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_mask_fill<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
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

        let out = P::int_mask_fill(
            tensor_state.tensor,
            mask_state.tensor,
            scalar_from_abi(value),
        );
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_expand<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::int_expand(tensor_state.tensor, shape);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_int_tensor_unfold<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_int(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::int_unfold(tensor_state.tensor, dim, size, step);
        let handle = adapter_state::<P>().insert_int(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_reshape<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::bool_reshape(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_gather<P: Backend>(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
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

        let out = P::bool_gather(dim, tensor_state.tensor, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_scatter_or<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_bool(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::bool_scatter_or(
            dim,
            tensor_state.tensor,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_select<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
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

        let out = P::bool_select(tensor_state.tensor, dim, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_select_or<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_bool(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != indices_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::bool_select_or(
            tensor_state.tensor,
            dim,
            indices_state.tensor,
            value_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_slice<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };

        let out = P::bool_slice(tensor_state.tensor, &slices);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_slice_assign<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_bool(value) {
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

        let out = P::bool_slice_assign(tensor_state.tensor, &slices, value_state.tensor);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_mask_where<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match adapter_state::<P>().lookup_bool(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_bool(value) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if tensor_state.device_handle != mask_state.device_handle
            || tensor_state.device_handle != value_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::bool_mask_where(tensor_state.tensor, mask_state.tensor, value_state.tensor);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_mask_fill<P: Backend>(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
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

        let out = P::bool_mask_fill(
            tensor_state.tensor,
            mask_state.tensor,
            scalar_from_abi(value),
        );
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_swap_dims<P: Backend>(
    tensor: TensorHandle,
    dim1: usize,
    dim2: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::bool_swap_dims(tensor_state.tensor, dim1, dim2);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_permute<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::bool_permute(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_flip<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::bool_flip(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_expand<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::bool_expand(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_bool_tensor_unfold<P: Backend>(
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

        let tensor_state = match adapter_state::<P>().lookup_bool(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::bool_unfold(tensor_state.tensor, dim, size, step);
        let handle =
            adapter_state::<P>().insert_bool(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_from_u8_data<P: Backend>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U8SliceRef,
    scheme: AbiQuantScheme,
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
        let values = match try_u8_data(data) {
            Ok(values) => values,
            Err(status) => return status,
        };
        let scheme = match quant_scheme_from_abi(scheme) {
            Ok(scheme) => scheme,
            Err(status) => return status,
        };

        let data = TensorData::from_bytes_vec(values, shape, DType::QFloat(scheme));
        let tensor = P::q_from_data(data, &device_state);
        let handle = adapter_state::<P>().insert_quantized(device, tensor);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_into_u8_data<P: Backend>(
    tensor: TensorHandle,
    out_scheme: *mut AbiQuantScheme,
    out_data: *mut OwnedU8Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_scheme.is_null() || out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let data = match burn_backend::read_sync(P::q_into_data(tensor_state.tensor)) {
            Ok(data) => data,
            Err(_) => return execution_error(),
        };
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            _ => return execution_error(),
        };

        let mut values = data.into_bytes().to_vec();
        let buffer = OwnedU8Buffer {
            ptr: values.as_mut_ptr(),
            len: values.len(),
        };
        std::mem::forget(values);

        unsafe {
            *out_scheme = quant_scheme_to_abi(scheme);
            *out_data = buffer;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_quantize<P: Backend>(
    tensor: TensorHandle,
    scheme: AbiQuantScheme,
    scales: TensorHandle,
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
        let scales_state = match adapter_state::<P>().lookup_float(scales) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if tensor_state.device_handle != scales_state.device_handle {
            return invalid_argument();
        }

        let scheme = match quant_scheme_from_abi(scheme) {
            Ok(scheme) => scheme,
            Err(status) => return status,
        };
        let qparams = QuantizationParametersPrimitive {
            scales: scales_state.tensor,
        };
        let out = P::quantize(tensor_state.tensor, &scheme, qparams);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_dequantize<P: Backend>(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::dequantize(tensor_state.tensor, float_dtype_from_abi(out_dtype));
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_to_device<P: Backend>(
    tensor: TensorHandle,
    device: DeviceHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };

        let out = P::q_to_device(tensor_state.tensor, &device_state);
        let handle = adapter_state::<P>().insert_quantized(device, out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_reshape<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::q_reshape(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_expand<P: Backend>(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::q_expand(tensor_state.tensor, shape);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_swap_dims<P: Backend>(
    tensor: TensorHandle,
    dim1: usize,
    dim2: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::q_swap_dims(tensor_state.tensor, dim1, dim2);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_permute<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::q_permute(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_flip<P: Backend>(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let axes = match try_shape(axes) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let out = P::q_flip(tensor_state.tensor, axes.as_slice());
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_select<P: Backend>(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
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

        let out = P::q_select(tensor_state.tensor, dim, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_q_tensor_slice<P: Backend>(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_quantized(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };

        let out = P::q_slice(tensor_state.tensor, &slices);
        let handle =
            adapter_state::<P>().insert_quantized(DeviceHandle(tensor_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_embedding<P: Backend>(
    weights: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let weights_state = match adapter_state::<P>().lookup_float(weights) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if weights_state.device_handle != indices_state.device_handle {
            return invalid_argument();
        }

        let out = P::embedding(weights_state.tensor, indices_state.tensor);
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(weights_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_embedding_backward<P: Backend>(
    weights: TensorHandle,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let weights_state = match adapter_state::<P>().lookup_float(weights) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if weights_state.device_handle != output_grad_state.device_handle
            || weights_state.device_handle != indices_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::embedding_backward(
            weights_state.tensor,
            output_grad_state.tensor,
            indices_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(weights_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv1d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv1d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_options_1_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv1d_x_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv1d_x_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_1_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv1d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv1d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_1_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv1d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv1d_bias_backward(x_state.tensor, bias_state.tensor, output_grad_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv2d_x_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv2d_x_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv2d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv2d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv2d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv2d_bias_backward(x_state.tensor, bias_state.tensor, output_grad_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv3d_x_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv3d_x_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_3_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv3d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv3d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_options_3_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv3d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv3d_bias_backward(x_state.tensor, bias_state.tensor, output_grad_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose1d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv_transpose1d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_transpose_options_1_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose1d_x_backward<P: Backend>(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if weight_state.device_handle != output_grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::conv_transpose1d_x_backward(
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_1_from_abi(options),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(weight_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose1d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose1d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_1_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose1d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose1d_bias_backward(
            x_state.tensor,
            bias_state.tensor,
            output_grad_state.tensor,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose2d_x_backward<P: Backend>(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if weight_state.device_handle != output_grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::conv_transpose2d_x_backward(
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_2_from_abi(options),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(weight_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose2d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose2d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose2d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose2d_bias_backward(
            x_state.tensor,
            bias_state.tensor,
            output_grad_state.tensor,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose3d_x_backward<P: Backend>(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if weight_state.device_handle != output_grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::conv_transpose3d_x_backward(
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_3_from_abi(options),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(weight_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose3d_weight_backward<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose3d_weight_backward(
            x_state.tensor,
            weight_state.tensor,
            output_grad_state.tensor,
            conv_transpose_options_3_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose3d_bias_backward<P: Backend>(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match adapter_state::<P>().lookup_float(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != bias_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::conv_transpose3d_bias_backward(
            x_state.tensor,
            bias_state.tensor,
            output_grad_state.tensor,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_unfold4d<P: Backend>(
    x: TensorHandle,
    kernel_size: [usize; 2],
    options: AbiUnfoldOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::unfold4d(x_state.tensor, kernel_size, unfold_options_from_abi(options));
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_avg_pool1d<P: Backend>(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::avg_pool1d(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            count_include_pad != 0,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_avg_pool1d_backward<P: Backend>(
    x: TensorHandle,
    grad: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let grad_state = match adapter_state::<P>().lookup_float(grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::avg_pool1d_backward(
            x_state.tensor,
            grad_state.tensor,
            kernel_size,
            stride,
            padding,
            count_include_pad != 0,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_adaptive_avg_pool1d<P: Backend>(
    x: TensorHandle,
    output_size: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::adaptive_avg_pool1d(x_state.tensor, output_size);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_adaptive_avg_pool1d_backward<P: Backend>(
    x: TensorHandle,
    grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let grad_state = match adapter_state::<P>().lookup_float(grad) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::adaptive_avg_pool1d_backward(x_state.tensor, grad_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_max_pool1d<P: Backend>(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::max_pool1d(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_max_pool1d_with_indices<P: Backend>(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    out_tensors: *mut AbiMaxPool1dWithIndices,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensors.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::max_pool1d_with_indices(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
        );
        let output =
            adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out.output);
        let indices =
            adapter_state::<P>().insert_int(DeviceHandle(x_state.device_handle), out.indices);

        unsafe {
            *out_tensors = AbiMaxPool1dWithIndices { output, indices };
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_max_pool1d_with_indices_backward<P: Backend>(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != output_grad_state.device_handle
            || x_state.device_handle != indices_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::max_pool1d_with_indices_backward(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
            output_grad_state.tensor,
            indices_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out.x_grad);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv2d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv2d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_deform_conv2d<P: Backend>(
    x: TensorHandle,
    offset: TensorHandle,
    weight: TensorHandle,
    mask: TensorHandle,
    bias: TensorHandle,
    options: AbiDeformConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let offset_state = match adapter_state::<P>().lookup_float(offset) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match optional_float_state::<P>(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != offset_state.device_handle
            || x_state.device_handle != weight_state.device_handle
            || mask_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::deform_conv2d(
            x_state.tensor,
            offset_state.tensor,
            weight_state.tensor,
            mask_state.map(|state| state.tensor),
            bias_state.map(|state| state.tensor),
            deform_conv_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_deform_conv2d_backward<P: Backend>(
    x: TensorHandle,
    offset: TensorHandle,
    weight: TensorHandle,
    mask: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    options: AbiDeformConvOptions2,
    out_tensors: *mut AbiDeformConv2dBackward,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensors.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let offset_state = match adapter_state::<P>().lookup_float(offset) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match optional_float_state::<P>(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != offset_state.device_handle
            || x_state.device_handle != weight_state.device_handle
            || x_state.device_handle != output_grad_state.device_handle
            || mask_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::deform_conv2d_backward(
            x_state.tensor,
            offset_state.tensor,
            weight_state.tensor,
            mask_state.map(|state| state.tensor),
            bias_state.map(|state| state.tensor),
            output_grad_state.tensor,
            deform_conv_options_2_from_abi(options),
        );

        let device = DeviceHandle(x_state.device_handle);
        let x_grad = adapter_state::<P>().insert_float(device, out.x_grad);
        let offset_grad = adapter_state::<P>().insert_float(device, out.offset_grad);
        let weight_grad = adapter_state::<P>().insert_float(device, out.weight_grad);
        let (mask_grad, has_mask_grad) = if let Some(mask_grad) = out.mask_grad {
            (adapter_state::<P>().insert_float(device, mask_grad), 1)
        } else {
            (TensorHandle::INVALID, 0)
        };
        let (bias_grad, has_bias_grad) = if let Some(bias_grad) = out.bias_grad {
            (adapter_state::<P>().insert_float(device, bias_grad), 1)
        } else {
            (TensorHandle::INVALID, 0)
        };

        unsafe {
            *out_tensors = AbiDeformConv2dBackward {
                x_grad,
                offset_grad,
                weight_grad,
                mask_grad,
                bias_grad,
                has_mask_grad,
                has_bias_grad,
            };
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv3d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv3d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_options_3_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose2d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv_transpose2d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_transpose_options_2_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_conv_transpose3d<P: Backend>(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let weight_state = match adapter_state::<P>().lookup_float(weight) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let bias_state = match optional_float_state::<P>(bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if x_state.device_handle != weight_state.device_handle
            || bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != x_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::conv_transpose3d(
            x_state.tensor,
            weight_state.tensor,
            bias_state.map(|state| state.tensor),
            conv_transpose_options_3_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_avg_pool2d<P: Backend>(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::avg_pool2d(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            count_include_pad != 0,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_avg_pool2d_backward<P: Backend>(
    x: TensorHandle,
    grad: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let grad_state = match adapter_state::<P>().lookup_float(grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if x_state.device_handle != grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::avg_pool2d_backward(
            x_state.tensor,
            grad_state.tensor,
            kernel_size,
            stride,
            padding,
            count_include_pad != 0,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_adaptive_avg_pool2d<P: Backend>(
    x: TensorHandle,
    output_size: [usize; 2],
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::adaptive_avg_pool2d(x_state.tensor, output_size);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_adaptive_avg_pool2d_backward<P: Backend>(
    x: TensorHandle,
    grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let grad_state = match adapter_state::<P>().lookup_float(grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if x_state.device_handle != grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::adaptive_avg_pool2d_backward(x_state.tensor, grad_state.tensor);
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_max_pool2d<P: Backend>(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::max_pool2d(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_max_pool2d_with_indices<P: Backend>(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    out_tensors: *mut AbiMaxPool2dWithIndices,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensors.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::max_pool2d_with_indices(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
        );
        let output =
            adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out.output);
        let indices =
            adapter_state::<P>().insert_int(DeviceHandle(x_state.device_handle), out.indices);

        unsafe {
            *out_tensors = AbiMaxPool2dWithIndices { output, indices };
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_max_pool2d_with_indices_backward<P: Backend>(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let output_grad_state = match adapter_state::<P>().lookup_float(output_grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let indices_state = match adapter_state::<P>().lookup_int(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if x_state.device_handle != output_grad_state.device_handle
            || x_state.device_handle != indices_state.device_handle
        {
            return invalid_argument();
        }

        let out = P::max_pool2d_with_indices_backward(
            x_state.tensor,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode != 0,
            output_grad_state.tensor,
            indices_state.tensor,
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out.x_grad);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_interpolate<P: Backend>(
    x: TensorHandle,
    output_size: [usize; 2],
    options: AbiInterpolateOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let out = P::interpolate(
            x_state.tensor,
            output_size,
            interpolate_options_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn abi_module_interpolate_backward<P: Backend>(
    x: TensorHandle,
    grad: TensorHandle,
    output_size: [usize; 2],
    options: AbiInterpolateOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let x_state = match adapter_state::<P>().lookup_float(x) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let grad_state = match adapter_state::<P>().lookup_float(grad) {
            Ok(state) => state,
            Err(status) => return status,
        };
        if x_state.device_handle != grad_state.device_handle {
            return invalid_argument();
        }

        let out = P::interpolate_backward(
            x_state.tensor,
            grad_state.tensor,
            output_size,
            interpolate_options_from_abi(options),
        );
        let handle = adapter_state::<P>().insert_float(DeviceHandle(x_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_attention<P: Backend>(
    query: TensorHandle,
    key: TensorHandle,
    value: TensorHandle,
    mask: TensorHandle,
    attn_bias: TensorHandle,
    options: AbiAttentionModuleOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let query_state = match adapter_state::<P>().lookup_float(query) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let key_state = match adapter_state::<P>().lookup_float(key) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let value_state = match adapter_state::<P>().lookup_float(value) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mask_state = match optional_bool_state::<P>(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let attn_bias_state = match optional_float_state::<P>(attn_bias) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if query_state.device_handle != key_state.device_handle
            || query_state.device_handle != value_state.device_handle
            || mask_state
                .as_ref()
                .is_some_and(|state| state.device_handle != query_state.device_handle)
            || attn_bias_state
                .as_ref()
                .is_some_and(|state| state.device_handle != query_state.device_handle)
        {
            return invalid_argument();
        }

        let out = P::attention(
            query_state.tensor,
            key_state.tensor,
            value_state.tensor,
            mask_state.map(|state| state.tensor),
            attn_bias_state.map(|state| state.tensor),
            attention_options_from_abi(options),
        );
        let handle =
            adapter_state::<P>().insert_float(DeviceHandle(query_state.device_handle), out);

        unsafe {
            *out_tensor = handle;
        }
        ok()
    })
}

unsafe extern "C" fn abi_module_rfft<P: Backend>(
    signal: TensorHandle,
    dim: usize,
    out_tensors: *mut AbiRfftOutput,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensors.is_null() {
            return invalid_argument();
        }
        let signal_state = match adapter_state::<P>().lookup_float(signal) {
            Ok(state) => state,
            Err(status) => return status,
        };

        let (real, imag) = P::rfft(signal_state.tensor, dim);
        let real =
            adapter_state::<P>().insert_float(DeviceHandle(signal_state.device_handle), real);
        let imag =
            adapter_state::<P>().insert_float(DeviceHandle(signal_state.device_handle), imag);

        unsafe {
            *out_tensors = AbiRfftOutput { real, imag };
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

unsafe extern "C" fn abi_release_u64_buffer(buffer: OwnedU64Buffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn abi_release_u8_buffer(buffer: OwnedU8Buffer) -> PluginStatus {
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

unsafe extern "C" fn abi_transaction_execute<P: Backend>(
    floats: TensorHandleRef,
    qfloats: TensorHandleRef,
    ints: TensorHandleRef,
    bools: TensorHandleRef,
    out_floats: *mut OwnedF32Buffer,
    out_qfloats: *mut OwnedQTransactionItem,
    out_ints: *mut OwnedU64Buffer,
    out_bools: *mut OwnedU8Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if floats.len > 0 && out_floats.is_null() {
            return invalid_argument();
        }
        if qfloats.len > 0 && out_qfloats.is_null() {
            return invalid_argument();
        }
        if ints.len > 0 && out_ints.is_null() {
            return invalid_argument();
        }
        if bools.len > 0 && out_bools.is_null() {
            return invalid_argument();
        }

        let float_handles = match try_tensor_handles(floats) {
            Ok(h) => h,
            Err(s) => return s,
        };
        let qfloat_handles = match try_tensor_handles(qfloats) {
            Ok(h) => h,
            Err(s) => return s,
        };
        let int_handles = match try_tensor_handles(ints) {
            Ok(h) => h,
            Err(s) => return s,
        };
        let bool_handles = match try_tensor_handles(bools) {
            Ok(h) => h,
            Err(s) => return s,
        };

        let state = adapter_state::<P>();

        let mut float_primitives = Vec::with_capacity(float_handles.len());
        for &h in &float_handles {
            match state.lookup_float(h) {
                Ok(s) => float_primitives.push(s.tensor),
                Err(st) => return st,
            }
        }

        let mut qfloat_primitives = Vec::with_capacity(qfloat_handles.len());
        for &h in &qfloat_handles {
            match state.lookup_quantized(h) {
                Ok(s) => qfloat_primitives.push(s.tensor),
                Err(st) => return st,
            }
        }

        let mut int_primitives = Vec::with_capacity(int_handles.len());
        for &h in &int_handles {
            match state.lookup_int(h) {
                Ok(s) => int_primitives.push(s.tensor),
                Err(st) => return st,
            }
        }

        let mut bool_primitives = Vec::with_capacity(bool_handles.len());
        for &h in &bool_handles {
            match state.lookup_bool(h) {
                Ok(s) => bool_primitives.push(s.tensor),
                Err(st) => return st,
            }
        }

        let transaction = TransactionPrimitive::new(
            float_primitives,
            qfloat_primitives,
            int_primitives,
            bool_primitives,
        );
        let result = match burn_backend::read_sync(P::tr_execute(transaction)) {
            Ok(data) => data,
            Err(_) => return execution_error(),
        };

        for (i, data) in result.read_floats.into_iter().enumerate() {
            let mut values = match data.into_vec::<f32>() {
                Ok(v) => v,
                Err(_) => return execution_error(),
            };
            let buf = OwnedF32Buffer {
                ptr: values.as_mut_ptr(),
                len: values.len(),
            };
            std::mem::forget(values);
            unsafe { *out_floats.add(i) = buf; }
        }

        for (i, data) in result.read_qfloats.into_iter().enumerate() {
            let scheme = match data.dtype {
                DType::QFloat(s) => s,
                _ => return execution_error(),
            };
            let mut bytes = data.into_bytes().to_vec();
            let buf = OwnedU8Buffer {
                ptr: bytes.as_mut_ptr(),
                len: bytes.len(),
            };
            std::mem::forget(bytes);
            unsafe {
                *out_qfloats.add(i) = OwnedQTransactionItem {
                    scheme: quant_scheme_to_abi(scheme),
                    data: buf,
                };
            }
        }

        for (i, data) in result.read_ints.into_iter().enumerate() {
            let mut values: Vec<u64> = match data.dtype {
                DType::I64 => match data.into_vec::<i64>() {
                    Ok(v) => v.into_iter().map(|x| x as u64).collect(),
                    Err(_) => return execution_error(),
                },
                DType::I32 => match data.into_vec::<i32>() {
                    Ok(v) => v.into_iter().map(|x| x as u64).collect(),
                    Err(_) => return execution_error(),
                },
                DType::I16 => match data.into_vec::<i16>() {
                    Ok(v) => v.into_iter().map(|x| x as u64).collect(),
                    Err(_) => return execution_error(),
                },
                DType::I8 => match data.into_vec::<i8>() {
                    Ok(v) => v.into_iter().map(|x| x as u64).collect(),
                    Err(_) => return execution_error(),
                },
                DType::U64 => match data.into_vec::<u64>() {
                    Ok(v) => v,
                    Err(_) => return execution_error(),
                },
                DType::U32 => match data.into_vec::<u32>() {
                    Ok(v) => v.into_iter().map(u64::from).collect(),
                    Err(_) => return execution_error(),
                },
                DType::U16 => match data.into_vec::<u16>() {
                    Ok(v) => v.into_iter().map(u64::from).collect(),
                    Err(_) => return execution_error(),
                },
                DType::U8 => match data.into_vec::<u8>() {
                    Ok(v) => v.into_iter().map(u64::from).collect(),
                    Err(_) => return execution_error(),
                },
                _ => return execution_error(),
            };
            let buf = OwnedU64Buffer {
                ptr: values.as_mut_ptr(),
                len: values.len(),
            };
            std::mem::forget(values);
            unsafe { *out_ints.add(i) = buf; }
        }

        for (i, data) in result.read_bools.into_iter().enumerate() {
            let mut values = match data.into_vec::<bool>() {
                Ok(v) => v.into_iter().map(u8::from).collect::<Vec<u8>>(),
                Err(_) => return execution_error(),
            };
            let buf = OwnedU8Buffer {
                ptr: values.as_mut_ptr(),
                len: values.len(),
            };
            std::mem::forget(values);
            unsafe { *out_bools.add(i) = buf; }
        }

        ok()
    })
}

/// Builds the tensor operation table for a backend-backed plugin implementation.
pub const fn backend_tensor_ops_v1<P: Backend>() -> BackendTensorOpsV1 {
    BackendTensorOpsV1 {
        abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
        create_default_device: abi_create_default_device::<P>,
        create_device: abi_create_device::<P>,
        release_device: abi_release_device::<P>,
        tensor_from_f32_data: abi_float_tensor_from_f32_data::<P>,
        tensor_into_f32_data: abi_float_tensor_into_f32_data::<P>,
        tensor_shape: abi_float_tensor_shape::<P>,
        tensor_random: abi_float_tensor_random::<P>,
        tensor_to_device: abi_float_tensor_to_device::<P>,
        tensor_empty: abi_float_tensor_empty::<P>,
        tensor_zeros: abi_float_tensor_zeros::<P>,
        tensor_ones: abi_float_tensor_ones::<P>,
        tensor_full: abi_float_tensor_full::<P>,
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
        tensor_prod: abi_float_tensor_prod::<P>,
        tensor_prod_dim: abi_float_tensor_prod_dim::<P>,
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
        tensor_repeat_dim: abi_float_tensor_repeat_dim::<P>,
        tensor_clamp_min: abi_float_tensor_clamp_min::<P>,
        tensor_clamp_max: abi_float_tensor_clamp_max::<P>,
        tensor_clamp: abi_float_tensor_clamp::<P>,
        tensor_neg: abi_float_tensor_neg::<P>,
        tensor_transpose: abi_float_tensor_transpose::<P>,
        tensor_not_equal: abi_float_tensor_not_equal::<P>,
        tensor_not_equal_elem: abi_float_tensor_not_equal_elem::<P>,
        tensor_mean: abi_float_tensor_mean::<P>,
        tensor_powi: abi_float_tensor_powi::<P>,
        tensor_powi_scalar: abi_float_tensor_powi_scalar::<P>,
        tensor_cat: abi_float_tensor_cat::<P>,
        tensor_max: abi_float_tensor_max::<P>,
        tensor_max_dim: abi_float_tensor_max_dim::<P>,
        tensor_max_dim_with_indices: abi_float_tensor_max_dim_with_indices::<P>,
        tensor_min: abi_float_tensor_min::<P>,
        tensor_min_dim: abi_float_tensor_min_dim::<P>,
        tensor_min_dim_with_indices: abi_float_tensor_min_dim_with_indices::<P>,
        tensor_max_abs: abi_float_tensor_max_abs::<P>,
        tensor_max_abs_dim: abi_float_tensor_max_abs_dim::<P>,
        tensor_any: abi_float_tensor_any::<P>,
        tensor_any_dim: abi_float_tensor_any_dim::<P>,
        tensor_all: abi_float_tensor_all::<P>,
        tensor_all_dim: abi_float_tensor_all_dim::<P>,
        tensor_sign: abi_float_tensor_sign::<P>,
        tensor_sort: abi_float_tensor_sort::<P>,
        tensor_sort_with_indices: abi_float_tensor_sort_with_indices::<P>,
        tensor_argsort: abi_float_tensor_argsort::<P>,
        tensor_is_nan: abi_float_tensor_is_nan::<P>,
        tensor_is_inf: abi_float_tensor_is_inf::<P>,
        int_tensor_from_u64_data: abi_int_tensor_from_u64_data::<P>,
        int_tensor_into_u64_data: abi_int_tensor_into_u64_data::<P>,
        int_tensor_to_device: abi_int_tensor_to_device::<P>,
        int_tensor_empty: abi_int_tensor_empty::<P>,
        int_tensor_zeros: abi_int_tensor_zeros::<P>,
        int_tensor_ones: abi_int_tensor_ones::<P>,
        int_tensor_full: abi_int_tensor_full::<P>,
        int_tensor_random: abi_int_tensor_random::<P>,
        int_tensor_into_float: abi_int_tensor_into_float::<P>,
        int_tensor_cast: abi_int_tensor_cast::<P>,
        int_tensor_add: abi_int_tensor_add::<P>,
        int_tensor_add_scalar: abi_int_tensor_add_scalar::<P>,
        int_tensor_sub: abi_int_tensor_sub::<P>,
        int_tensor_sub_scalar: abi_int_tensor_sub_scalar::<P>,
        int_tensor_mul: abi_int_tensor_mul::<P>,
        int_tensor_mul_scalar: abi_int_tensor_mul_scalar::<P>,
        int_tensor_div: abi_int_tensor_div::<P>,
        int_tensor_div_scalar: abi_int_tensor_div_scalar::<P>,
        int_tensor_remainder: abi_int_tensor_remainder::<P>,
        int_tensor_remainder_scalar: abi_int_tensor_remainder_scalar::<P>,
        int_tensor_matmul: abi_int_tensor_matmul::<P>,
        int_tensor_abs: abi_int_tensor_abs::<P>,
        int_tensor_sum: abi_int_tensor_sum::<P>,
        int_tensor_sum_dim: abi_int_tensor_sum_dim::<P>,
        int_tensor_prod: abi_int_tensor_prod::<P>,
        int_tensor_prod_dim: abi_int_tensor_prod_dim::<P>,
        int_tensor_mean_dim: abi_int_tensor_mean_dim::<P>,
        int_tensor_cumsum: abi_int_tensor_cumsum::<P>,
        int_tensor_cumprod: abi_int_tensor_cumprod::<P>,
        int_tensor_cummin: abi_int_tensor_cummin::<P>,
        int_tensor_cummax: abi_int_tensor_cummax::<P>,
        int_tensor_argmax: abi_int_tensor_argmax::<P>,
        int_tensor_argmin: abi_int_tensor_argmin::<P>,
        int_tensor_swap_dims: abi_int_tensor_swap_dims::<P>,
        int_tensor_permute: abi_int_tensor_permute::<P>,
        int_tensor_flip: abi_int_tensor_flip::<P>,
        int_tensor_reshape: abi_int_tensor_reshape::<P>,
        int_tensor_gather: abi_int_tensor_gather::<P>,
        int_tensor_scatter_add: abi_int_tensor_scatter_add::<P>,
        int_tensor_select: abi_int_tensor_select::<P>,
        int_tensor_select_add: abi_int_tensor_select_add::<P>,
        int_tensor_slice: abi_int_tensor_slice::<P>,
        int_tensor_slice_assign: abi_int_tensor_slice_assign::<P>,
        int_tensor_mask_where: abi_int_tensor_mask_where::<P>,
        int_tensor_mask_fill: abi_int_tensor_mask_fill::<P>,
        int_tensor_equal: abi_int_tensor_equal::<P>,
        int_tensor_equal_elem: abi_int_tensor_equal_elem::<P>,
        int_tensor_greater: abi_int_tensor_greater::<P>,
        int_tensor_greater_elem: abi_int_tensor_greater_elem::<P>,
        int_tensor_greater_equal: abi_int_tensor_greater_equal::<P>,
        int_tensor_greater_equal_elem: abi_int_tensor_greater_equal_elem::<P>,
        int_tensor_lower: abi_int_tensor_lower::<P>,
        int_tensor_lower_elem: abi_int_tensor_lower_elem::<P>,
        int_tensor_lower_equal: abi_int_tensor_lower_equal::<P>,
        int_tensor_lower_equal_elem: abi_int_tensor_lower_equal_elem::<P>,
        int_tensor_bitwise_and: abi_int_tensor_bitwise_and::<P>,
        int_tensor_bitwise_and_scalar: abi_int_tensor_bitwise_and_scalar::<P>,
        int_tensor_bitwise_or: abi_int_tensor_bitwise_or::<P>,
        int_tensor_bitwise_or_scalar: abi_int_tensor_bitwise_or_scalar::<P>,
        int_tensor_bitwise_xor: abi_int_tensor_bitwise_xor::<P>,
        int_tensor_bitwise_xor_scalar: abi_int_tensor_bitwise_xor_scalar::<P>,
        int_tensor_bitwise_not: abi_int_tensor_bitwise_not::<P>,
        int_tensor_bitwise_left_shift: abi_int_tensor_bitwise_left_shift::<P>,
        int_tensor_bitwise_left_shift_scalar: abi_int_tensor_bitwise_left_shift_scalar::<P>,
        int_tensor_bitwise_right_shift: abi_int_tensor_bitwise_right_shift::<P>,
        int_tensor_bitwise_right_shift_scalar: abi_int_tensor_bitwise_right_shift_scalar::<P>,
        int_tensor_expand: abi_int_tensor_expand::<P>,
        int_tensor_unfold: abi_int_tensor_unfold::<P>,
        int_tensor_repeat_dim: abi_int_tensor_repeat_dim::<P>,
        int_tensor_cat: abi_int_tensor_cat::<P>,
        int_tensor_not_equal: abi_int_tensor_not_equal::<P>,
        int_tensor_not_equal_elem: abi_int_tensor_not_equal_elem::<P>,
        int_tensor_powi: abi_int_tensor_powi::<P>,
        int_tensor_powi_scalar: abi_int_tensor_powi_scalar::<P>,
        int_tensor_clamp_min: abi_int_tensor_clamp_min::<P>,
        int_tensor_clamp_max: abi_int_tensor_clamp_max::<P>,
        int_tensor_clamp: abi_int_tensor_clamp::<P>,
        int_tensor_neg: abi_int_tensor_neg::<P>,
        int_tensor_mean: abi_int_tensor_mean::<P>,
        int_tensor_max: abi_int_tensor_max::<P>,
        int_tensor_max_dim: abi_int_tensor_max_dim::<P>,
        int_tensor_max_dim_with_indices: abi_int_tensor_max_dim_with_indices::<P>,
        int_tensor_max_abs: abi_int_tensor_max_abs::<P>,
        int_tensor_max_abs_dim: abi_int_tensor_max_abs_dim::<P>,
        int_tensor_min: abi_int_tensor_min::<P>,
        int_tensor_min_dim: abi_int_tensor_min_dim::<P>,
        int_tensor_min_dim_with_indices: abi_int_tensor_min_dim_with_indices::<P>,
        int_tensor_transpose: abi_int_tensor_transpose::<P>,
        int_tensor_arange_step: abi_int_tensor_arange_step::<P>,
        int_tensor_arange: abi_int_tensor_arange::<P>,
        int_tensor_any: abi_int_tensor_any::<P>,
        int_tensor_any_dim: abi_int_tensor_any_dim::<P>,
        int_tensor_all: abi_int_tensor_all::<P>,
        int_tensor_all_dim: abi_int_tensor_all_dim::<P>,
        int_tensor_sign: abi_int_tensor_sign::<P>,
        int_tensor_sort: abi_int_tensor_sort::<P>,
        int_tensor_sort_with_indices: abi_int_tensor_sort_with_indices::<P>,
        int_tensor_argsort: abi_int_tensor_argsort::<P>,
        bool_tensor_from_u8_data: abi_bool_tensor_from_u8_data::<P>,
        bool_tensor_into_u8_data: abi_bool_tensor_into_u8_data::<P>,
        bool_tensor_into_int: abi_bool_tensor_into_int::<P>,
        bool_tensor_into_float: abi_bool_tensor_into_float::<P>,
        bool_tensor_to_device: abi_bool_tensor_to_device::<P>,
        bool_tensor_empty: abi_bool_tensor_empty::<P>,
        bool_tensor_zeros: abi_bool_tensor_zeros::<P>,
        bool_tensor_ones: abi_bool_tensor_ones::<P>,
        bool_tensor_reshape: abi_bool_tensor_reshape::<P>,
        bool_tensor_gather: abi_bool_tensor_gather::<P>,
        bool_tensor_scatter_or: abi_bool_tensor_scatter_or::<P>,
        bool_tensor_select: abi_bool_tensor_select::<P>,
        bool_tensor_select_or: abi_bool_tensor_select_or::<P>,
        bool_tensor_slice: abi_bool_tensor_slice::<P>,
        bool_tensor_slice_assign: abi_bool_tensor_slice_assign::<P>,
        bool_tensor_mask_where: abi_bool_tensor_mask_where::<P>,
        bool_tensor_mask_fill: abi_bool_tensor_mask_fill::<P>,
        bool_tensor_equal: abi_bool_tensor_equal::<P>,
        bool_tensor_equal_elem: abi_bool_tensor_equal_elem::<P>,
        bool_tensor_not: abi_bool_tensor_not::<P>,
        bool_tensor_and: abi_bool_tensor_and::<P>,
        bool_tensor_or: abi_bool_tensor_or::<P>,
        bool_tensor_swap_dims: abi_bool_tensor_swap_dims::<P>,
        bool_tensor_permute: abi_bool_tensor_permute::<P>,
        bool_tensor_flip: abi_bool_tensor_flip::<P>,
        bool_tensor_expand: abi_bool_tensor_expand::<P>,
        bool_tensor_unfold: abi_bool_tensor_unfold::<P>,
        bool_tensor_repeat_dim: abi_bool_tensor_repeat_dim::<P>,
        bool_tensor_cat: abi_bool_tensor_cat::<P>,
        bool_tensor_not_equal: abi_bool_tensor_not_equal::<P>,
        bool_tensor_not_equal_elem: abi_bool_tensor_not_equal_elem::<P>,
        bool_tensor_xor: abi_bool_tensor_xor::<P>,
        bool_tensor_transpose: abi_bool_tensor_transpose::<P>,
        bool_tensor_any: abi_bool_tensor_any::<P>,
        bool_tensor_any_dim: abi_bool_tensor_any_dim::<P>,
        bool_tensor_all: abi_bool_tensor_all::<P>,
        bool_tensor_all_dim: abi_bool_tensor_all_dim::<P>,
        q_tensor_from_u8_data: abi_q_tensor_from_u8_data::<P>,
        q_tensor_into_u8_data: abi_q_tensor_into_u8_data::<P>,
        q_tensor_quantize: abi_q_tensor_quantize::<P>,
        q_tensor_dequantize: abi_q_tensor_dequantize::<P>,
        q_tensor_to_device: abi_q_tensor_to_device::<P>,
        q_tensor_reshape: abi_q_tensor_reshape::<P>,
        q_tensor_expand: abi_q_tensor_expand::<P>,
        q_tensor_swap_dims: abi_q_tensor_swap_dims::<P>,
        q_tensor_permute: abi_q_tensor_permute::<P>,
        q_tensor_flip: abi_q_tensor_flip::<P>,
        q_tensor_select: abi_q_tensor_select::<P>,
        q_tensor_slice: abi_q_tensor_slice::<P>,
        module_embedding: abi_module_embedding::<P>,
        module_embedding_backward: abi_module_embedding_backward::<P>,
        module_conv1d: abi_module_conv1d::<P>,
        module_conv1d_x_backward: abi_module_conv1d_x_backward::<P>,
        module_conv1d_weight_backward: abi_module_conv1d_weight_backward::<P>,
        module_conv1d_bias_backward: abi_module_conv1d_bias_backward::<P>,
        module_conv2d_x_backward: abi_module_conv2d_x_backward::<P>,
        module_conv2d_weight_backward: abi_module_conv2d_weight_backward::<P>,
        module_conv2d_bias_backward: abi_module_conv2d_bias_backward::<P>,
        module_conv3d_x_backward: abi_module_conv3d_x_backward::<P>,
        module_conv3d_weight_backward: abi_module_conv3d_weight_backward::<P>,
        module_conv3d_bias_backward: abi_module_conv3d_bias_backward::<P>,
        module_conv_transpose1d: abi_module_conv_transpose1d::<P>,
        module_conv_transpose1d_x_backward: abi_module_conv_transpose1d_x_backward::<P>,
        module_conv_transpose1d_weight_backward: abi_module_conv_transpose1d_weight_backward::<P>,
        module_conv_transpose1d_bias_backward: abi_module_conv_transpose1d_bias_backward::<P>,
        module_conv_transpose2d_x_backward: abi_module_conv_transpose2d_x_backward::<P>,
        module_conv_transpose2d_weight_backward: abi_module_conv_transpose2d_weight_backward::<P>,
        module_conv_transpose2d_bias_backward: abi_module_conv_transpose2d_bias_backward::<P>,
        module_conv_transpose3d_x_backward: abi_module_conv_transpose3d_x_backward::<P>,
        module_conv_transpose3d_weight_backward: abi_module_conv_transpose3d_weight_backward::<P>,
        module_conv_transpose3d_bias_backward: abi_module_conv_transpose3d_bias_backward::<P>,
        module_unfold4d: abi_module_unfold4d::<P>,
        module_avg_pool1d: abi_module_avg_pool1d::<P>,
        module_avg_pool1d_backward: abi_module_avg_pool1d_backward::<P>,
        module_adaptive_avg_pool1d: abi_module_adaptive_avg_pool1d::<P>,
        module_adaptive_avg_pool1d_backward: abi_module_adaptive_avg_pool1d_backward::<P>,
        module_max_pool1d: abi_module_max_pool1d::<P>,
        module_max_pool1d_with_indices: abi_module_max_pool1d_with_indices::<P>,
        module_max_pool1d_with_indices_backward: abi_module_max_pool1d_with_indices_backward::<P>,
        module_conv2d: abi_module_conv2d::<P>,
        module_deform_conv2d: abi_module_deform_conv2d::<P>,
        module_deform_conv2d_backward: abi_module_deform_conv2d_backward::<P>,
        module_conv3d: abi_module_conv3d::<P>,
        module_conv_transpose2d: abi_module_conv_transpose2d::<P>,
        module_conv_transpose3d: abi_module_conv_transpose3d::<P>,
        module_avg_pool2d: abi_module_avg_pool2d::<P>,
        module_avg_pool2d_backward: abi_module_avg_pool2d_backward::<P>,
        module_adaptive_avg_pool2d: abi_module_adaptive_avg_pool2d::<P>,
        module_adaptive_avg_pool2d_backward: abi_module_adaptive_avg_pool2d_backward::<P>,
        module_max_pool2d: abi_module_max_pool2d::<P>,
        module_max_pool2d_with_indices: abi_module_max_pool2d_with_indices::<P>,
        module_max_pool2d_with_indices_backward: abi_module_max_pool2d_with_indices_backward::<P>,
        module_interpolate: abi_module_interpolate::<P>,
        module_interpolate_backward: abi_module_interpolate_backward::<P>,
        module_attention: abi_module_attention::<P>,
        module_rfft: abi_module_rfft::<P>,
        activation_leaky_relu: abi_activation_leaky_relu::<P>,
        activation_relu: abi_activation_relu::<P>,
        activation_relu_backward: abi_activation_relu_backward::<P>,
        activation_gelu: abi_activation_gelu::<P>,
        activation_prelu: abi_activation_prelu::<P>,
        activation_gelu_backward: abi_activation_gelu_backward::<P>,
        activation_sigmoid: abi_activation_sigmoid::<P>,
        activation_sigmoid_backward: abi_activation_sigmoid_backward::<P>,
        activation_hard_sigmoid: abi_activation_hard_sigmoid::<P>,
        activation_log_sigmoid: abi_activation_log_sigmoid::<P>,
        activation_log_sigmoid_backward: abi_activation_log_sigmoid_backward::<P>,
        transaction_execute: abi_transaction_execute::<P>,
        release_tensor: abi_release_tensor::<P>,
        release_f32_buffer: abi_release_f32_buffer,
        release_u64_buffer: abi_release_u64_buffer,
        release_u8_buffer: abi_release_u8_buffer,
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
