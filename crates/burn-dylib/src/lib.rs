#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn dynamic backend plugin ABI.
//!
//! This crate provides two layers:
//! - A versioned metadata ABI (`BackendPluginV1`) for backend plugins.
//! - A versioned tensor/device operation ABI (`BackendTensorOpsV1`) for device and tensor
//!   operations such as tensor creation, reads, and addition.
//! - A runtime loader (`loader`) to load both tables from a shared library.
//!
//! # Design Goal
//!
//! Compile application code without linking any heavy backend, then load a backend plugin (`.so`,
//! `.dylib`, `.dll`) at runtime.
//!
//! # Naming Conventions
//!
//! The dylib stack follows a strict naming convention so call paths are predictable and easy to
//! maintain:
//! - Loader methods use `backend_*` for backend metadata/control and `float_tensor_*` for current
//!   tensor ops.
//! - Runtime forwarding functions mirror loader names (`backend_*`, `float_tensor_*`).
//! - Adapter FFI shims are prefixed with `abi_*` to clearly separate C ABI glue from backend
//!   trait calls.

use core::ffi::c_char;

/// Backend-backed helpers for implementing backend plugins without hand-writing
/// the whole C ABI shim.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod adapter;

mod backend;
mod device;
mod ops;
mod runtime;
mod tensor;

pub use backend::Dylib;
pub use device::DylibDevice;
pub use runtime::DylibError;

pub use runtime::{
    create_default_device, create_default_device_from_path, create_device_from_path,
    device_from_registry,
};

/// Symbol name that backend plugins must export.
pub const BACKEND_PLUGIN_SYMBOL: &[u8] = b"burn_backend_plugin_v1\0";

/// Symbol name that backend tensor operation table exports must use.
pub const BACKEND_TENSOR_OPS_SYMBOL: &[u8] = b"burn_backend_tensor_ops_v1\0";

/// Current plugin ABI version.
pub const BACKEND_PLUGIN_ABI_VERSION: u32 = 1;

/// Current tensor operations ABI version.
pub const BACKEND_TENSOR_OPS_ABI_VERSION: u32 = 7;

/// Status code returned by plugin callbacks.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginStatusCode {
    /// Operation completed successfully.
    Ok = 0,
    /// Generic failure.
    Failed = 1,
    /// Invalid input argument.
    InvalidArgument = 2,
    /// Operation is not supported by this backend.
    Unsupported = 3,
}

/// Return value for plugin callbacks.
///
/// `message` should either be null or point to a null-terminated static string.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginStatus {
    /// Result code.
    pub code: PluginStatusCode,
    /// Optional message.
    pub message: *const c_char,
}

impl PluginStatus {
    /// Create a successful status.
    pub const fn ok() -> Self {
        Self {
            code: PluginStatusCode::Ok,
            message: core::ptr::null(),
        }
    }

    /// Create a failing status with a custom code and message pointer.
    pub const fn error(code: PluginStatusCode, message: *const c_char) -> Self {
        Self { code, message }
    }

    /// Create a generic failing status.
    pub const fn failed(message: *const c_char) -> Self {
        Self::error(PluginStatusCode::Failed, message)
    }
}

/// Callback type for backend name.
pub type BackendNameFn = unsafe extern "C" fn() -> *const c_char;

/// Callback type for seeding backend state.
pub type BackendSeedFn = unsafe extern "C" fn(seed: u64) -> PluginStatus;

/// Callback type for synchronizing backend execution.
pub type BackendSyncFn = unsafe extern "C" fn() -> PluginStatus;

/// Callback type for reporting available device count.
pub type BackendDeviceCountFn = unsafe extern "C" fn(type_id: u16) -> usize;

/// Opaque device handle managed by a plugin backend.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceHandle(pub u64);

impl DeviceHandle {
    /// Invalid handle value.
    pub const INVALID: Self = Self(0);

    /// Returns true when the handle is valid.
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Opaque tensor handle managed by a plugin backend.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorHandle(pub u64);

impl TensorHandle {
    /// Invalid handle value.
    pub const INVALID: Self = Self(0);

    /// Returns true when the handle is valid.
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

/// Borrowed tensor shape descriptor passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorShapeRef {
    /// Pointer to a contiguous list of dimensions.
    pub dims: *const usize,
    /// Number of dimensions.
    pub rank: usize,
}

/// Borrowed tensor-handle list descriptor passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorHandleRef {
    /// Pointer to a contiguous list of tensor handles.
    pub ptr: *const TensorHandle,
    /// Number of tensor handles.
    pub len: usize,
}

/// Borrowed f32 data slice passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct F32SliceRef {
    /// Pointer to contiguous f32 data.
    pub ptr: *const f32,
    /// Number of f32 elements.
    pub len: usize,
}

/// Borrowed `u64` data slice passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct U64SliceRef {
    /// Pointer to contiguous `u64` data.
    pub ptr: *const u64,
    /// Number of `u64` elements.
    pub len: usize,
}

/// Borrowed `u8` data slice passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct U8SliceRef {
    /// Pointer to contiguous `u8` data.
    pub ptr: *const u8,
    /// Number of `u8` elements.
    pub len: usize,
}

/// Owned f32 buffer returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedF32Buffer {
    /// Pointer to contiguous f32 data allocated by the plugin.
    pub ptr: *mut f32,
    /// Number of f32 elements.
    pub len: usize,
}

impl OwnedF32Buffer {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// Owned `u64` buffer returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedU64Buffer {
    /// Pointer to contiguous `u64` data allocated by the plugin.
    pub ptr: *mut u64,
    /// Number of `u64` elements.
    pub len: usize,
}

impl OwnedU64Buffer {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// Owned `u8` buffer returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedU8Buffer {
    /// Pointer to contiguous `u8` data allocated by the plugin.
    pub ptr: *mut u8,
    /// Number of `u8` elements.
    pub len: usize,
}

impl OwnedU8Buffer {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// Owned shape buffer returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedUsizeBuffer {
    /// Pointer to contiguous dimensions allocated by the plugin.
    pub ptr: *mut usize,
    /// Number of dimensions.
    pub len: usize,
}

impl OwnedUsizeBuffer {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// ABI float dtype representation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiFloatDType {
    /// 64-bit float.
    F64 = 0,
    /// 32-bit float.
    F32 = 1,
    /// Flexible 32-bit float.
    Flex32 = 2,
    /// 16-bit float.
    F16 = 3,
    /// Brain float 16.
    BF16 = 4,
}

/// ABI int dtype representation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiIntDType {
    /// 64-bit signed integer.
    I64 = 0,
    /// 32-bit signed integer.
    I32 = 1,
    /// 16-bit signed integer.
    I16 = 2,
    /// 8-bit signed integer.
    I8 = 3,
    /// 64-bit unsigned integer.
    U64 = 4,
    /// 32-bit unsigned integer.
    U32 = 5,
    /// 16-bit unsigned integer.
    U16 = 6,
    /// 8-bit unsigned integer.
    U8 = 7,
}

/// ABI bool dtype representation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiBoolDType {
    /// Native bool storage.
    Native = 0,
    /// `u8` bool storage.
    U8 = 1,
    /// `u32` bool storage.
    U32 = 2,
}

/// Max number of dimensions encoded for quantized block-size descriptors.
pub const ABI_QUANT_BLOCK_MAX_DIMS: usize = 5;

/// ABI quantized value representation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantValue {
    /// 8-bit quantization with full range.
    Q8F = 0,
    /// 8-bit floating-point format (e5m2).
    E5M2 = 1,
    /// 8-bit floating-point format (e4m3).
    E4M3 = 2,
    /// 4-bit quantization with full range.
    Q4F = 3,
    /// 4-bit floating-point format (e2m1).
    E2M1 = 4,
    /// 2-bit quantization with full range.
    Q2F = 5,
    /// 8-bit quantization with symmetric range.
    Q8S = 6,
    /// 4-bit quantization with symmetric range.
    Q4S = 7,
    /// 2-bit quantization with symmetric range.
    Q2S = 8,
}

/// ABI quantization parameter precision.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantParam {
    /// Full precision.
    F32 = 0,
    /// Half precision.
    F16 = 1,
    /// bfloat16 precision.
    BF16 = 2,
    /// Unsigned floating-point format (e8m0).
    UE8M0 = 3,
    /// Unsigned floating-point format (e4m3).
    UE4M3 = 4,
}

/// ABI quantization storage descriptor.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantStore {
    /// Native quantization storage.
    Native = 0,
    /// Native packed quantization storage.
    PackedNative = 1,
    /// 32-bit packed quantization storage.
    PackedU32 = 2,
}

/// ABI quantization mode.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantMode {
    /// Symmetric quantization mode.
    Symmetric = 0,
}

/// ABI quantization granularity level.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantLevel {
    /// Per-tensor quantization.
    Tensor = 0,
    /// Per-block quantization.
    Block = 1,
}

/// ABI quantization scheme descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiQuantScheme {
    /// Quantized value representation.
    pub value: AbiQuantValue,
    /// Quantization parameter precision.
    pub param: AbiQuantParam,
    /// Storage format for quantized values.
    pub store: AbiQuantStore,
    /// Packing dimension for packed stores.
    pub store_packed_dim: usize,
    /// Quantization granularity level.
    pub level: AbiQuantLevel,
    /// Block dimensions for block-level quantization.
    pub block_dims: [u8; ABI_QUANT_BLOCK_MAX_DIMS],
    /// Number of valid entries in `block_dims`.
    pub block_rank: usize,
    /// Quantization mode.
    pub mode: AbiQuantMode,
}

/// ABI random distribution tag.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiDistributionKind {
    /// Default distribution.
    Default = 0,
    /// Bernoulli distribution.
    Bernoulli = 1,
    /// Uniform distribution.
    Uniform = 2,
    /// Normal distribution.
    Normal = 3,
}

/// ABI random distribution descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiDistribution {
    /// Distribution kind.
    pub kind: AbiDistributionKind,
    /// First distribution parameter.
    pub param0: f64,
    /// Second distribution parameter.
    pub param1: f64,
}

/// ABI scalar tag.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiScalarKind {
    /// `f64` scalar payload.
    Float = 0,
    /// `i64` scalar payload.
    Int = 1,
    /// `u64` scalar payload.
    UInt = 2,
    /// `bool` scalar payload (`0` or `1`).
    Bool = 3,
}

/// ABI scalar payload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiScalar {
    /// Scalar kind.
    pub kind: AbiScalarKind,
    /// Scalar payload bits.
    pub payload: u64,
}

/// ABI slice descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiSlice {
    /// Slice start.
    pub start: isize,
    /// Slice end value when present.
    pub end: isize,
    /// Slice step.
    pub step: isize,
    /// Whether `end` is present.
    pub has_end: u8,
}

/// Borrowed slice list descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiSliceRef {
    /// Pointer to contiguous slice descriptors.
    pub ptr: *const AbiSlice,
    /// Number of slices.
    pub len: usize,
}

/// ABI interpolation mode tag.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiInterpolateMode {
    /// Nearest-neighbor interpolation.
    Nearest = 0,
    /// Bilinear interpolation.
    Bilinear = 1,
    /// Bicubic interpolation.
    Bicubic = 2,
    /// Lanczos3 interpolation.
    Lanczos3 = 3,
}

/// ABI interpolation options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiInterpolateOptions {
    /// Interpolation mode.
    pub mode: AbiInterpolateMode,
    /// Whether corners should be aligned.
    pub align_corners: u8,
}

/// ABI 2D convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvOptions2 {
    /// Stride.
    pub stride: [usize; 2],
    /// Padding.
    pub padding: [usize; 2],
    /// Dilation.
    pub dilation: [usize; 2],
    /// Groups.
    pub groups: usize,
}

/// ABI 1D convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvOptions1 {
    /// Stride.
    pub stride: [usize; 1],
    /// Padding.
    pub padding: [usize; 1],
    /// Dilation.
    pub dilation: [usize; 1],
    /// Groups.
    pub groups: usize,
}

/// ABI 3D convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvOptions3 {
    /// Stride.
    pub stride: [usize; 3],
    /// Padding.
    pub padding: [usize; 3],
    /// Dilation.
    pub dilation: [usize; 3],
    /// Groups.
    pub groups: usize,
}

/// ABI 2D deformable convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiDeformConvOptions2 {
    /// Stride.
    pub stride: [usize; 2],
    /// Padding.
    pub padding: [usize; 2],
    /// Dilation.
    pub dilation: [usize; 2],
    /// Weight groups.
    pub weight_groups: usize,
    /// Offset groups.
    pub offset_groups: usize,
}

/// ABI 2D transposed convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvTransposeOptions2 {
    /// Stride.
    pub stride: [usize; 2],
    /// Padding.
    pub padding: [usize; 2],
    /// Output padding.
    pub padding_out: [usize; 2],
    /// Dilation.
    pub dilation: [usize; 2],
    /// Groups.
    pub groups: usize,
}

/// ABI 1D transposed convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvTransposeOptions1 {
    /// Stride.
    pub stride: [usize; 1],
    /// Padding.
    pub padding: [usize; 1],
    /// Output padding.
    pub padding_out: [usize; 1],
    /// Dilation.
    pub dilation: [usize; 1],
    /// Groups.
    pub groups: usize,
}

/// ABI 3D transposed convolution options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiConvTransposeOptions3 {
    /// Stride.
    pub stride: [usize; 3],
    /// Padding.
    pub padding: [usize; 3],
    /// Output padding.
    pub padding_out: [usize; 3],
    /// Dilation.
    pub dilation: [usize; 3],
    /// Groups.
    pub groups: usize,
}

/// ABI attention module options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiAttentionModuleOptions {
    /// Optional scale value.
    pub scale: f64,
    /// Whether scale is present.
    pub has_scale: u8,
    /// Optional softcap value.
    pub softcap: f64,
    /// Whether softcap is present.
    pub has_softcap: u8,
    /// Whether attention is causal.
    pub is_causal: u8,
}

/// ABI unfold operation options.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiUnfoldOptions {
    /// Stride.
    pub stride: [usize; 2],
    /// Padding.
    pub padding: [usize; 2],
    /// Dilation.
    pub dilation: [usize; 2],
}

/// ABI output payload for max-pool-with-indices.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiMaxPool2dWithIndices {
    /// Output values tensor.
    pub output: TensorHandle,
    /// Output indices tensor.
    pub indices: TensorHandle,
}

/// ABI output payload for max-pool1d-with-indices.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiMaxPool1dWithIndices {
    /// Output values tensor.
    pub output: TensorHandle,
    /// Output indices tensor.
    pub indices: TensorHandle,
}

/// ABI output payload for tensor operations returning values and indices.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiTensorWithIndices {
    /// Output values tensor.
    pub values: TensorHandle,
    /// Output indices tensor.
    pub indices: TensorHandle,
}

/// ABI output payload for deformable-convolution backward.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiDeformConv2dBackward {
    /// Input gradient tensor.
    pub x_grad: TensorHandle,
    /// Offset gradient tensor.
    pub offset_grad: TensorHandle,
    /// Weight gradient tensor.
    pub weight_grad: TensorHandle,
    /// Optional mask gradient tensor.
    pub mask_grad: TensorHandle,
    /// Optional bias gradient tensor.
    pub bias_grad: TensorHandle,
    /// Whether `mask_grad` is present.
    pub has_mask_grad: u8,
    /// Whether `bias_grad` is present.
    pub has_bias_grad: u8,
}

/// ABI output payload for RFFT.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiRfftOutput {
    /// Real output tensor.
    pub real: TensorHandle,
    /// Imaginary output tensor.
    pub imag: TensorHandle,
}

/// Owned quantized tensor data result from a transaction call.
///
/// One item is written per quantized tensor read in a [`TransactionExecuteFn`] call.
/// The `data` buffer is plugin-allocated and must be released via `release_u8_buffer`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OwnedQTransactionItem {
    /// Quantization scheme for this result.
    pub scheme: AbiQuantScheme,
    /// Quantized data bytes (plugin-allocated).
    pub data: OwnedU8Buffer,
}

/// Creates a default backend device and writes its type ID, ordinal, and handle into `out_type_id`, `out_ordinal`, and `out_device`.
pub type BackendCreateDefaultDeviceFn = unsafe extern "C" fn(
    out_type_id: *mut u16,
    out_ordinal: *mut usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus;

/// Creates a backend device and writes its handle into `out_device`.
pub type BackendCreateDeviceFn = unsafe extern "C" fn(
    type_id: u16,
    ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus;

/// Releases a backend device handle.
pub type BackendReleaseDeviceFn = unsafe extern "C" fn(device: DeviceHandle) -> PluginStatus;

/// Creates a tensor from f32 host data.
pub type TensorFromF32DataFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Creates an int tensor from host `u64` data.
pub type TensorFromU64DataFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U64SliceRef,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Creates a bool tensor from host `u8` data.
pub type TensorFromU8DataFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U8SliceRef,
    dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Materializes a tensor into host f32 data.
pub type TensorIntoF32DataFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_data: *mut OwnedF32Buffer) -> PluginStatus;

/// Materializes an int tensor into host `u64` data.
pub type TensorIntoU64DataFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_data: *mut OwnedU64Buffer) -> PluginStatus;

/// Materializes a bool tensor into host `u8` data.
pub type TensorIntoU8DataFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_data: *mut OwnedU8Buffer) -> PluginStatus;

/// Creates a quantized tensor from host `u8` bytes and quantization scheme.
pub type QTensorFromU8DataFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: U8SliceRef,
    scheme: AbiQuantScheme,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Materializes a quantized tensor into host `u8` bytes and quantization scheme.
pub type QTensorIntoU8DataFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_scheme: *mut AbiQuantScheme,
    out_data: *mut OwnedU8Buffer,
) -> PluginStatus;

/// Quantizes a float tensor using a quantization scheme and scales tensor.
pub type QTensorQuantizeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    scheme: AbiQuantScheme,
    scales: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Dequantizes a quantized tensor into a float tensor.
pub type QTensorDequantizeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Fetches the tensor shape.
pub type TensorShapeFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_shape: *mut OwnedUsizeBuffer) -> PluginStatus;

/// Generic unary tensor operation.
pub type TensorUnaryFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_tensor: *mut TensorHandle) -> PluginStatus;

/// Generic binary tensor operation.
pub type TensorBinaryFn = unsafe extern "C" fn(
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Generic scalar tensor operation.
pub type TensorScalarFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    scalar: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor random creation operation.
pub type TensorRandomFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    distribution: AbiDistribution,
    dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Int tensor random creation operation.
pub type TensorRandomIntFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    distribution: AbiDistribution,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor empty creation operation.
pub type TensorEmptyFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Int tensor empty creation operation.
pub type TensorEmptyIntFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Bool tensor empty creation operation.
pub type TensorEmptyBoolFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor cast to int operation.
pub type TensorIntoIntFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor cast to float operation.
pub type TensorCastFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor cast to bool operation.
pub type TensorIntoBoolFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor to-device operation.
pub type TensorToDeviceFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    device: DeviceHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor operation with dimension argument.
pub type TensorDimFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor operation with two dimension arguments.
pub type TensorSwapDimsFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim1: usize,
    dim2: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor operation with axis list argument.
pub type TensorAxesFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    axes: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor operation with shape argument.
pub type TensorReshapeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    shape: TensorShapeRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor cross operation.
pub type TensorCrossFn = unsafe extern "C" fn(
    lhs: TensorHandle,
    rhs: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor gather operation.
pub type TensorGatherFn = unsafe extern "C" fn(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor scatter add operation.
pub type TensorScatterAddFn = unsafe extern "C" fn(
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor select operation.
pub type TensorSelectFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor select add operation.
pub type TensorSelectAddFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor slice operation.
pub type TensorSliceFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor slice assign operation.
pub type TensorSliceAssignFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    slices: AbiSliceRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor mask where operation.
pub type TensorMaskWhereFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor mask fill operation.
pub type TensorMaskFillFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    mask: TensorHandle,
    value: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor clamp operation with scalar min and max.
pub type TensorClampFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    min: AbiScalar,
    max: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor comparison operation.
pub type TensorCompareFn = unsafe extern "C" fn(
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor scalar comparison operation.
pub type TensorCompareScalarFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    rhs: AbiScalar,
    out_dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor comparison operation that keeps bool dtype.
pub type TensorBoolCompareFn = unsafe extern "C" fn(
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor scalar comparison operation that keeps bool dtype.
pub type TensorBoolCompareScalarFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    rhs: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor arg reduction operation.
pub type TensorArgFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor unfold operation.
pub type TensorUnfoldFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    size: usize,
    step: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Float tensor full creation operation.
pub type TensorFullFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    value: AbiScalar,
    dtype: AbiFloatDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Int tensor full creation operation.
pub type TensorFullIntFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    value: AbiScalar,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor repeat-dim operation.
pub type TensorRepeatDimFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    times: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor concatenation operation.
pub type TensorCatFn = unsafe extern "C" fn(
    tensors: TensorHandleRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor boolean reduction operation with output bool dtype.
pub type TensorBoolReduceFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    out_dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor boolean reduction operation along a dimension with output bool dtype.
pub type TensorBoolReduceDimFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_dtype: AbiBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor operation returning values and indices.
pub type TensorWithIndicesFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_dtype: AbiIntDType,
    out_tensors: *mut AbiTensorWithIndices,
) -> PluginStatus;

/// Tensor operation returning values and indices with backend-chosen index dtype.
pub type TensorWithIndicesNoDTypeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_tensors: *mut AbiTensorWithIndices,
) -> PluginStatus;

/// Tensor sort operation.
pub type TensorSortFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    descending: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor sort-with-indices operation.
pub type TensorSortWithIndicesFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    descending: u8,
    out_dtype: AbiIntDType,
    out_tensors: *mut AbiTensorWithIndices,
) -> PluginStatus;

/// Tensor sort-with-indices operation with backend-chosen index dtype.
pub type TensorSortWithIndicesNoDTypeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    descending: u8,
    out_tensors: *mut AbiTensorWithIndices,
) -> PluginStatus;

/// Tensor argsort operation.
pub type TensorArgsortFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    descending: u8,
    out_dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor argsort operation with backend-chosen index dtype.
pub type TensorArgsortNoDTypeFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    descending: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Bool tensor reduction operation along a dimension.
pub type BoolTensorDimFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Int tensor arange operation.
pub type IntTensorArangeFn = unsafe extern "C" fn(
    start: i64,
    end: i64,
    device: DeviceHandle,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Int tensor arange-step operation.
pub type IntTensorArangeStepFn = unsafe extern "C" fn(
    start: i64,
    end: i64,
    step: usize,
    device: DeviceHandle,
    dtype: AbiIntDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module embedding operation.
pub type ModuleEmbeddingFn = unsafe extern "C" fn(
    weights: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module embedding backward operation.
pub type ModuleEmbeddingBackwardFn = unsafe extern "C" fn(
    weights: TensorHandle,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv1d operation.
pub type ModuleConv1dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv1d input-backward operation.
pub type ModuleConv1dXBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv1d weight-backward operation.
pub type ModuleConv1dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv1d bias-backward operation.
pub type ModuleConv1dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv2d input-backward operation.
pub type ModuleConv2dXBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv2d weight-backward operation.
pub type ModuleConv2dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv2d bias-backward operation.
pub type ModuleConv2dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv3d input-backward operation.
pub type ModuleConv3dXBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv3d weight-backward operation.
pub type ModuleConv3dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv3d bias-backward operation.
pub type ModuleConv3dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv1d operation.
pub type ModuleConvTranspose1dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv1d input-backward operation.
pub type ModuleConvTranspose1dXBackwardFn = unsafe extern "C" fn(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv1d weight-backward operation.
pub type ModuleConvTranspose1dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions1,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv1d bias-backward operation.
pub type ModuleConvTranspose1dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv2d input-backward operation.
pub type ModuleConvTranspose2dXBackwardFn = unsafe extern "C" fn(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv2d weight-backward operation.
pub type ModuleConvTranspose2dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv2d bias-backward operation.
pub type ModuleConvTranspose2dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv3d input-backward operation.
pub type ModuleConvTranspose3dXBackwardFn = unsafe extern "C" fn(
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv3d weight-backward operation.
pub type ModuleConvTranspose3dWeightBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    output_grad: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv3d bias-backward operation.
pub type ModuleConvTranspose3dBiasBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module unfold4d operation.
pub type ModuleUnfold4dFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: [usize; 2],
    options: AbiUnfoldOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module avg-pool1d operation.
pub type ModuleAvgPool1dFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module avg-pool1d backward operation.
pub type ModuleAvgPool1dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    grad: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module adaptive avg-pool1d operation.
pub type ModuleAdaptiveAvgPool1dFn = unsafe extern "C" fn(
    x: TensorHandle,
    output_size: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module adaptive avg-pool1d backward operation.
pub type ModuleAdaptiveAvgPool1dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module max-pool1d operation.
pub type ModuleMaxPool1dFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module max-pool1d with indices operation.
pub type ModuleMaxPool1dWithIndicesFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    out_tensors: *mut AbiMaxPool1dWithIndices,
) -> PluginStatus;

/// Module max-pool1d backward operation.
pub type ModuleMaxPool1dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil_mode: u8,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module conv2d operation.
pub type ModuleConv2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module deformable conv2d operation.
pub type ModuleDeformConv2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    offset: TensorHandle,
    weight: TensorHandle,
    mask: TensorHandle,
    bias: TensorHandle,
    options: AbiDeformConvOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module deformable conv2d backward operation.
pub type ModuleDeformConv2dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    offset: TensorHandle,
    weight: TensorHandle,
    mask: TensorHandle,
    bias: TensorHandle,
    output_grad: TensorHandle,
    options: AbiDeformConvOptions2,
    out_tensors: *mut AbiDeformConv2dBackward,
) -> PluginStatus;

/// Module conv3d operation.
pub type ModuleConv3dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv2d operation.
pub type ModuleConvTranspose2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions2,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module transposed conv3d operation.
pub type ModuleConvTranspose3dFn = unsafe extern "C" fn(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    options: AbiConvTransposeOptions3,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module avg-pool2d operation.
pub type ModuleAvgPool2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module avg-pool2d backward operation.
pub type ModuleAvgPool2dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    grad: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: u8,
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module adaptive avg-pool2d operation.
pub type ModuleAdaptiveAvgPool2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    output_size: [usize; 2],
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module adaptive avg-pool2d backward operation.
pub type ModuleAdaptiveAvgPool2dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    grad: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module max-pool2d operation.
pub type ModuleMaxPool2dFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module max-pool2d with indices operation.
pub type ModuleMaxPool2dWithIndicesFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    out_tensors: *mut AbiMaxPool2dWithIndices,
) -> PluginStatus;

/// Module max-pool2d backward operation.
pub type ModuleMaxPool2dBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: u8,
    output_grad: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module interpolate operation.
pub type ModuleInterpolateFn = unsafe extern "C" fn(
    x: TensorHandle,
    output_size: [usize; 2],
    options: AbiInterpolateOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module interpolate backward operation.
pub type ModuleInterpolateBackwardFn = unsafe extern "C" fn(
    x: TensorHandle,
    grad: TensorHandle,
    output_size: [usize; 2],
    options: AbiInterpolateOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module attention operation.
pub type ModuleAttentionFn = unsafe extern "C" fn(
    query: TensorHandle,
    key: TensorHandle,
    value: TensorHandle,
    mask: TensorHandle,
    attn_bias: TensorHandle,
    options: AbiAttentionModuleOptions,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Module RFFT operation.
pub type ModuleRfftFn = unsafe extern "C" fn(
    signal: TensorHandle,
    dim: usize,
    out_tensors: *mut AbiRfftOutput,
) -> PluginStatus;

/// Activation operation with one scalar argument.
pub type ActivationScalarFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    scalar: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Hard-sigmoid activation operation.
pub type ActivationHardSigmoidFn = unsafe extern "C" fn(
    tensor: TensorHandle,
    alpha: AbiScalar,
    beta: AbiScalar,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Tensor addition operation.
pub type TensorAddFn = TensorBinaryFn;

/// Releases a tensor handle.
pub type TensorReleaseFn = unsafe extern "C" fn(tensor: TensorHandle) -> PluginStatus;

/// Releases a plugin-allocated f32 buffer.
pub type ReleaseF32BufferFn = unsafe extern "C" fn(buffer: OwnedF32Buffer) -> PluginStatus;

/// Releases a plugin-allocated `u64` buffer.
pub type ReleaseU64BufferFn = unsafe extern "C" fn(buffer: OwnedU64Buffer) -> PluginStatus;

/// Releases a plugin-allocated `u8` buffer.
pub type ReleaseU8BufferFn = unsafe extern "C" fn(buffer: OwnedU8Buffer) -> PluginStatus;

/// Releases a plugin-allocated shape buffer.
pub type ReleaseUsizeBufferFn = unsafe extern "C" fn(buffer: OwnedUsizeBuffer) -> PluginStatus;

/// Reads multiple tensors in a single plugin call.
///
/// `out_floats`, `out_qfloats`, `out_ints`, and `out_bools` are caller-allocated arrays whose
/// lengths match the corresponding `TensorHandleRef.len` fields. Pass a null pointer when the
/// corresponding count is zero. The plugin fills each slot; data pointers within are
/// plugin-allocated and must be released with the corresponding `release_*_buffer` functions.
pub type TransactionExecuteFn = unsafe extern "C" fn(
    floats: TensorHandleRef,
    qfloats: TensorHandleRef,
    ints: TensorHandleRef,
    bools: TensorHandleRef,
    out_floats: *mut OwnedF32Buffer,
    out_qfloats: *mut OwnedQTransactionItem,
    out_ints: *mut OwnedU64Buffer,
    out_bools: *mut OwnedU8Buffer,
) -> PluginStatus;

/// C ABI table exported by backend plugins.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BackendPluginV1 {
    /// ABI version for compatibility checks.
    pub abi_version: u32,
    /// Backend name function.
    pub backend_name: BackendNameFn,
    /// Backend seed function.
    pub seed: BackendSeedFn,
    /// Backend synchronization function.
    pub sync: BackendSyncFn,
    /// Device count query function.
    pub device_count: BackendDeviceCountFn,
}

/// Entrypoint function exported from plugin dynamic libraries.
pub type BackendPluginEntrypoint = unsafe extern "C" fn() -> *const BackendPluginV1;

/// C ABI table containing tensor and device operations exposed by plugin backends.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BackendTensorOpsV1 {
    /// ABI version for compatibility checks.
    pub abi_version: u32,
    /// Creates a default backend device.
    pub create_default_device: BackendCreateDefaultDeviceFn,
    /// Creates a backend device from `(type_id, ordinal)`.
    pub create_device: BackendCreateDeviceFn,
    /// Releases a backend device handle.
    pub release_device: BackendReleaseDeviceFn,
    /// Creates a tensor from host f32 data.
    pub tensor_from_f32_data: TensorFromF32DataFn,
    /// Materializes a tensor into host f32 data.
    pub tensor_into_f32_data: TensorIntoF32DataFn,
    /// Returns the tensor shape.
    pub tensor_shape: TensorShapeFn,
    /// Creates a random tensor.
    pub tensor_random: TensorRandomFn,
    /// Moves a tensor to a target device.
    pub tensor_to_device: TensorToDeviceFn,
    /// Creates an empty tensor.
    pub tensor_empty: TensorEmptyFn,
    /// Creates a float tensor filled with zeros.
    pub tensor_zeros: TensorEmptyFn,
    /// Creates a float tensor filled with ones.
    pub tensor_ones: TensorEmptyFn,
    /// Creates a float tensor filled with a scalar value.
    pub tensor_full: TensorFullFn,
    /// Casts a float tensor into an int tensor.
    pub tensor_into_int: TensorIntoIntFn,
    /// Dispatches tensor addition.
    pub tensor_add: TensorAddFn,
    /// Dispatches tensor-scalar addition.
    pub tensor_add_scalar: TensorScalarFn,
    /// Dispatches tensor subtraction.
    pub tensor_sub: TensorBinaryFn,
    /// Dispatches tensor-scalar subtraction.
    pub tensor_sub_scalar: TensorScalarFn,
    /// Dispatches tensor multiplication.
    pub tensor_mul: TensorBinaryFn,
    /// Dispatches tensor-scalar multiplication.
    pub tensor_mul_scalar: TensorScalarFn,
    /// Dispatches tensor division.
    pub tensor_div: TensorBinaryFn,
    /// Dispatches tensor-scalar division.
    pub tensor_div_scalar: TensorScalarFn,
    /// Dispatches tensor remainder.
    pub tensor_remainder: TensorBinaryFn,
    /// Dispatches tensor-scalar remainder.
    pub tensor_remainder_scalar: TensorScalarFn,
    /// Dispatches tensor matrix multiplication.
    pub tensor_matmul: TensorBinaryFn,
    /// Dispatches tensor cross product.
    pub tensor_cross: TensorCrossFn,
    /// Dispatches tensor reciprocal.
    pub tensor_recip: TensorUnaryFn,
    /// Dispatches tensor swap dims.
    pub tensor_swap_dims: TensorSwapDimsFn,
    /// Dispatches tensor permute.
    pub tensor_permute: TensorAxesFn,
    /// Dispatches tensor flip.
    pub tensor_flip: TensorAxesFn,
    /// Dispatches tensor reshape.
    pub tensor_reshape: TensorReshapeFn,
    /// Dispatches tensor gather.
    pub tensor_gather: TensorGatherFn,
    /// Dispatches tensor scatter add.
    pub tensor_scatter_add: TensorScatterAddFn,
    /// Dispatches tensor select.
    pub tensor_select: TensorSelectFn,
    /// Dispatches tensor select add.
    pub tensor_select_add: TensorSelectAddFn,
    /// Dispatches tensor slice.
    pub tensor_slice: TensorSliceFn,
    /// Dispatches tensor slice assign.
    pub tensor_slice_assign: TensorSliceAssignFn,
    /// Dispatches tensor mask where.
    pub tensor_mask_where: TensorMaskWhereFn,
    /// Dispatches tensor mask fill.
    pub tensor_mask_fill: TensorMaskFillFn,
    /// Dispatches tensor equality comparison.
    pub tensor_equal: TensorCompareFn,
    /// Dispatches tensor-scalar equality comparison.
    pub tensor_equal_elem: TensorCompareScalarFn,
    /// Dispatches tensor greater comparison.
    pub tensor_greater: TensorCompareFn,
    /// Dispatches tensor-scalar greater comparison.
    pub tensor_greater_elem: TensorCompareScalarFn,
    /// Dispatches tensor greater-equal comparison.
    pub tensor_greater_equal: TensorCompareFn,
    /// Dispatches tensor-scalar greater-equal comparison.
    pub tensor_greater_equal_elem: TensorCompareScalarFn,
    /// Dispatches tensor lower comparison.
    pub tensor_lower: TensorCompareFn,
    /// Dispatches tensor-scalar lower comparison.
    pub tensor_lower_elem: TensorCompareScalarFn,
    /// Dispatches tensor lower-equal comparison.
    pub tensor_lower_equal: TensorCompareFn,
    /// Dispatches tensor-scalar lower-equal comparison.
    pub tensor_lower_equal_elem: TensorCompareScalarFn,
    /// Dispatches tensor sum reduction.
    pub tensor_sum: TensorUnaryFn,
    /// Dispatches tensor sum-dim reduction.
    pub tensor_sum_dim: TensorDimFn,
    /// Dispatches tensor product reduction.
    pub tensor_prod: TensorUnaryFn,
    /// Dispatches tensor product-dim reduction.
    pub tensor_prod_dim: TensorDimFn,
    /// Dispatches tensor mean-dim reduction.
    pub tensor_mean_dim: TensorDimFn,
    /// Dispatches tensor cumsum.
    pub tensor_cumsum: TensorDimFn,
    /// Dispatches tensor cumprod.
    pub tensor_cumprod: TensorDimFn,
    /// Dispatches tensor cummin.
    pub tensor_cummin: TensorDimFn,
    /// Dispatches tensor cummax.
    pub tensor_cummax: TensorDimFn,
    /// Dispatches tensor cast.
    pub tensor_cast: TensorCastFn,
    /// Dispatches tensor exp.
    pub tensor_exp: TensorUnaryFn,
    /// Dispatches tensor log.
    pub tensor_log: TensorUnaryFn,
    /// Dispatches tensor log1p.
    pub tensor_log1p: TensorUnaryFn,
    /// Dispatches tensor power.
    pub tensor_powf: TensorBinaryFn,
    /// Dispatches tensor-scalar power.
    pub tensor_powf_scalar: TensorScalarFn,
    /// Dispatches tensor sqrt.
    pub tensor_sqrt: TensorUnaryFn,
    /// Dispatches tensor abs.
    pub tensor_abs: TensorUnaryFn,
    /// Dispatches tensor cos.
    pub tensor_cos: TensorUnaryFn,
    /// Dispatches tensor sin.
    pub tensor_sin: TensorUnaryFn,
    /// Dispatches tensor tan.
    pub tensor_tan: TensorUnaryFn,
    /// Dispatches tensor cosh.
    pub tensor_cosh: TensorUnaryFn,
    /// Dispatches tensor sinh.
    pub tensor_sinh: TensorUnaryFn,
    /// Dispatches tensor tanh.
    pub tensor_tanh: TensorUnaryFn,
    /// Dispatches tensor acos.
    pub tensor_acos: TensorUnaryFn,
    /// Dispatches tensor acosh.
    pub tensor_acosh: TensorUnaryFn,
    /// Dispatches tensor asin.
    pub tensor_asin: TensorUnaryFn,
    /// Dispatches tensor asinh.
    pub tensor_asinh: TensorUnaryFn,
    /// Dispatches tensor atan.
    pub tensor_atan: TensorUnaryFn,
    /// Dispatches tensor atanh.
    pub tensor_atanh: TensorUnaryFn,
    /// Dispatches tensor atan2.
    pub tensor_atan2: TensorBinaryFn,
    /// Dispatches tensor round.
    pub tensor_round: TensorUnaryFn,
    /// Dispatches tensor floor.
    pub tensor_floor: TensorUnaryFn,
    /// Dispatches tensor ceil.
    pub tensor_ceil: TensorUnaryFn,
    /// Dispatches tensor trunc.
    pub tensor_trunc: TensorUnaryFn,
    /// Dispatches tensor erf.
    pub tensor_erf: TensorUnaryFn,
    /// Dispatches tensor argmax.
    pub tensor_argmax: TensorArgFn,
    /// Dispatches tensor argmin.
    pub tensor_argmin: TensorArgFn,
    /// Dispatches tensor expand.
    pub tensor_expand: TensorReshapeFn,
    /// Dispatches tensor unfold.
    pub tensor_unfold: TensorUnfoldFn,
    /// Dispatches tensor repeat-dim.
    pub tensor_repeat_dim: TensorRepeatDimFn,
    /// Dispatches tensor clamp-min.
    pub tensor_clamp_min: TensorScalarFn,
    /// Dispatches tensor clamp-max.
    pub tensor_clamp_max: TensorScalarFn,
    /// Dispatches tensor clamp.
    pub tensor_clamp: TensorClampFn,
    /// Dispatches tensor negation.
    pub tensor_neg: TensorUnaryFn,
    /// Dispatches tensor transpose.
    pub tensor_transpose: TensorUnaryFn,
    /// Dispatches tensor non-equality comparison.
    pub tensor_not_equal: TensorCompareFn,
    /// Dispatches tensor-scalar non-equality comparison.
    pub tensor_not_equal_elem: TensorCompareScalarFn,
    /// Dispatches tensor mean reduction.
    pub tensor_mean: TensorUnaryFn,
    /// Dispatches tensor integer power.
    pub tensor_powi: TensorBinaryFn,
    /// Dispatches tensor integer-scalar power.
    pub tensor_powi_scalar: TensorScalarFn,
    /// Dispatches tensor concatenation.
    pub tensor_cat: TensorCatFn,
    /// Dispatches tensor max reduction.
    pub tensor_max: TensorUnaryFn,
    /// Dispatches tensor max-dim reduction.
    pub tensor_max_dim: TensorDimFn,
    /// Dispatches tensor max-dim with indices.
    pub tensor_max_dim_with_indices: TensorWithIndicesFn,
    /// Dispatches tensor min reduction.
    pub tensor_min: TensorUnaryFn,
    /// Dispatches tensor min-dim reduction.
    pub tensor_min_dim: TensorDimFn,
    /// Dispatches tensor min-dim with indices.
    pub tensor_min_dim_with_indices: TensorWithIndicesFn,
    /// Dispatches tensor max-abs reduction.
    pub tensor_max_abs: TensorUnaryFn,
    /// Dispatches tensor max-abs-dim reduction.
    pub tensor_max_abs_dim: TensorDimFn,
    /// Dispatches tensor any reduction.
    pub tensor_any: TensorBoolReduceFn,
    /// Dispatches tensor any-dim reduction.
    pub tensor_any_dim: TensorBoolReduceDimFn,
    /// Dispatches tensor all reduction.
    pub tensor_all: TensorBoolReduceFn,
    /// Dispatches tensor all-dim reduction.
    pub tensor_all_dim: TensorBoolReduceDimFn,
    /// Dispatches tensor sign operation.
    pub tensor_sign: TensorUnaryFn,
    /// Dispatches tensor sort operation.
    pub tensor_sort: TensorSortFn,
    /// Dispatches tensor sort-with-indices operation.
    pub tensor_sort_with_indices: TensorSortWithIndicesFn,
    /// Dispatches tensor argsort operation.
    pub tensor_argsort: TensorArgsortFn,
    /// Dispatches tensor is-NaN check.
    pub tensor_is_nan: TensorBoolReduceFn,
    /// Dispatches tensor is-INF check.
    pub tensor_is_inf: TensorBoolReduceFn,
    /// Creates an int tensor from host `u64` data.
    pub int_tensor_from_u64_data: TensorFromU64DataFn,
    /// Materializes an int tensor into host `u64` data.
    pub int_tensor_into_u64_data: TensorIntoU64DataFn,
    /// Moves an int tensor to a target device.
    pub int_tensor_to_device: TensorToDeviceFn,
    /// Creates an empty int tensor.
    pub int_tensor_empty: TensorEmptyIntFn,
    /// Creates a random int tensor.
    pub int_tensor_random: TensorRandomIntFn,
    /// Casts an int tensor into a float tensor.
    pub int_tensor_into_float: TensorCastFn,
    /// Casts an int tensor into another int dtype.
    pub int_tensor_cast: TensorIntoIntFn,
    /// Dispatches int tensor addition.
    pub int_tensor_add: TensorBinaryFn,
    /// Dispatches int tensor-scalar addition.
    pub int_tensor_add_scalar: TensorScalarFn,
    /// Dispatches int tensor subtraction.
    pub int_tensor_sub: TensorBinaryFn,
    /// Dispatches int tensor-scalar subtraction.
    pub int_tensor_sub_scalar: TensorScalarFn,
    /// Dispatches int tensor multiplication.
    pub int_tensor_mul: TensorBinaryFn,
    /// Dispatches int tensor-scalar multiplication.
    pub int_tensor_mul_scalar: TensorScalarFn,
    /// Dispatches int tensor division.
    pub int_tensor_div: TensorBinaryFn,
    /// Dispatches int tensor-scalar division.
    pub int_tensor_div_scalar: TensorScalarFn,
    /// Dispatches int tensor remainder.
    pub int_tensor_remainder: TensorBinaryFn,
    /// Dispatches int tensor-scalar remainder.
    pub int_tensor_remainder_scalar: TensorScalarFn,
    /// Dispatches int tensor matrix multiplication.
    pub int_tensor_matmul: TensorBinaryFn,
    /// Dispatches int tensor absolute value.
    pub int_tensor_abs: TensorUnaryFn,
    /// Dispatches int tensor sum reduction.
    pub int_tensor_sum: TensorUnaryFn,
    /// Dispatches int tensor sum-dim reduction.
    pub int_tensor_sum_dim: TensorDimFn,
    /// Dispatches int tensor product reduction.
    pub int_tensor_prod: TensorUnaryFn,
    /// Dispatches int tensor product-dim reduction.
    pub int_tensor_prod_dim: TensorDimFn,
    /// Dispatches int tensor mean-dim reduction.
    pub int_tensor_mean_dim: TensorDimFn,
    /// Dispatches int tensor cumsum.
    pub int_tensor_cumsum: TensorDimFn,
    /// Dispatches int tensor cumprod.
    pub int_tensor_cumprod: TensorDimFn,
    /// Dispatches int tensor cummin.
    pub int_tensor_cummin: TensorDimFn,
    /// Dispatches int tensor cummax.
    pub int_tensor_cummax: TensorDimFn,
    /// Dispatches int tensor argmax.
    pub int_tensor_argmax: TensorDimFn,
    /// Dispatches int tensor argmin.
    pub int_tensor_argmin: TensorDimFn,
    /// Dispatches int tensor swap dims.
    pub int_tensor_swap_dims: TensorSwapDimsFn,
    /// Dispatches int tensor permute.
    pub int_tensor_permute: TensorAxesFn,
    /// Dispatches int tensor flip.
    pub int_tensor_flip: TensorAxesFn,
    /// Dispatches int tensor reshape.
    pub int_tensor_reshape: TensorReshapeFn,
    /// Dispatches int tensor gather.
    pub int_tensor_gather: TensorGatherFn,
    /// Dispatches int tensor scatter add.
    pub int_tensor_scatter_add: TensorScatterAddFn,
    /// Dispatches int tensor select.
    pub int_tensor_select: TensorSelectFn,
    /// Dispatches int tensor select add.
    pub int_tensor_select_add: TensorSelectAddFn,
    /// Dispatches int tensor slice.
    pub int_tensor_slice: TensorSliceFn,
    /// Dispatches int tensor slice assign.
    pub int_tensor_slice_assign: TensorSliceAssignFn,
    /// Dispatches int tensor mask where.
    pub int_tensor_mask_where: TensorMaskWhereFn,
    /// Dispatches int tensor mask fill.
    pub int_tensor_mask_fill: TensorMaskFillFn,
    /// Dispatches int tensor equality comparison.
    pub int_tensor_equal: TensorCompareFn,
    /// Dispatches int tensor-scalar equality comparison.
    pub int_tensor_equal_elem: TensorCompareScalarFn,
    /// Dispatches int tensor greater comparison.
    pub int_tensor_greater: TensorCompareFn,
    /// Dispatches int tensor-scalar greater comparison.
    pub int_tensor_greater_elem: TensorCompareScalarFn,
    /// Dispatches int tensor greater-equal comparison.
    pub int_tensor_greater_equal: TensorCompareFn,
    /// Dispatches int tensor-scalar greater-equal comparison.
    pub int_tensor_greater_equal_elem: TensorCompareScalarFn,
    /// Dispatches int tensor lower comparison.
    pub int_tensor_lower: TensorCompareFn,
    /// Dispatches int tensor-scalar lower comparison.
    pub int_tensor_lower_elem: TensorCompareScalarFn,
    /// Dispatches int tensor lower-equal comparison.
    pub int_tensor_lower_equal: TensorCompareFn,
    /// Dispatches int tensor-scalar lower-equal comparison.
    pub int_tensor_lower_equal_elem: TensorCompareScalarFn,
    /// Dispatches int tensor bitwise and.
    pub int_tensor_bitwise_and: TensorBinaryFn,
    /// Dispatches int tensor-scalar bitwise and.
    pub int_tensor_bitwise_and_scalar: TensorScalarFn,
    /// Dispatches int tensor bitwise or.
    pub int_tensor_bitwise_or: TensorBinaryFn,
    /// Dispatches int tensor-scalar bitwise or.
    pub int_tensor_bitwise_or_scalar: TensorScalarFn,
    /// Dispatches int tensor bitwise xor.
    pub int_tensor_bitwise_xor: TensorBinaryFn,
    /// Dispatches int tensor-scalar bitwise xor.
    pub int_tensor_bitwise_xor_scalar: TensorScalarFn,
    /// Dispatches int tensor bitwise not.
    pub int_tensor_bitwise_not: TensorUnaryFn,
    /// Dispatches int tensor bitwise left shift.
    pub int_tensor_bitwise_left_shift: TensorBinaryFn,
    /// Dispatches int tensor-scalar bitwise left shift.
    pub int_tensor_bitwise_left_shift_scalar: TensorScalarFn,
    /// Dispatches int tensor bitwise right shift.
    pub int_tensor_bitwise_right_shift: TensorBinaryFn,
    /// Dispatches int tensor-scalar bitwise right shift.
    pub int_tensor_bitwise_right_shift_scalar: TensorScalarFn,
    /// Dispatches int tensor expand.
    pub int_tensor_expand: TensorReshapeFn,
    /// Dispatches int tensor unfold.
    pub int_tensor_unfold: TensorUnfoldFn,
    /// Dispatches int tensor repeat-dim.
    pub int_tensor_repeat_dim: TensorRepeatDimFn,
    /// Dispatches int tensor concatenation.
    pub int_tensor_cat: TensorCatFn,
    /// Dispatches int tensor non-equality comparison.
    pub int_tensor_not_equal: TensorCompareFn,
    /// Dispatches int tensor-scalar non-equality comparison.
    pub int_tensor_not_equal_elem: TensorCompareScalarFn,
    /// Dispatches int tensor integer power.
    pub int_tensor_powi: TensorBinaryFn,
    /// Dispatches int tensor integer-scalar power.
    pub int_tensor_powi_scalar: TensorScalarFn,
    /// Dispatches int tensor clamp-min.
    pub int_tensor_clamp_min: TensorScalarFn,
    /// Dispatches int tensor clamp-max.
    pub int_tensor_clamp_max: TensorScalarFn,
    /// Dispatches int tensor clamp.
    pub int_tensor_clamp: TensorClampFn,
    /// Dispatches int tensor negation.
    pub int_tensor_neg: TensorUnaryFn,
    /// Creates an int tensor filled with zeros.
    pub int_tensor_zeros: TensorEmptyIntFn,
    /// Creates an int tensor filled with ones.
    pub int_tensor_ones: TensorEmptyIntFn,
    /// Creates an int tensor filled with a scalar value.
    pub int_tensor_full: TensorFullIntFn,
    /// Dispatches int tensor mean reduction.
    pub int_tensor_mean: TensorUnaryFn,
    /// Dispatches int tensor max reduction.
    pub int_tensor_max: TensorUnaryFn,
    /// Dispatches int tensor max-dim reduction.
    pub int_tensor_max_dim: TensorDimFn,
    /// Dispatches int tensor max-dim with indices.
    pub int_tensor_max_dim_with_indices: TensorWithIndicesNoDTypeFn,
    /// Dispatches int tensor max-abs reduction.
    pub int_tensor_max_abs: TensorUnaryFn,
    /// Dispatches int tensor max-abs-dim reduction.
    pub int_tensor_max_abs_dim: TensorDimFn,
    /// Dispatches int tensor min reduction.
    pub int_tensor_min: TensorUnaryFn,
    /// Dispatches int tensor min-dim reduction.
    pub int_tensor_min_dim: TensorDimFn,
    /// Dispatches int tensor min-dim with indices.
    pub int_tensor_min_dim_with_indices: TensorWithIndicesNoDTypeFn,
    /// Dispatches int tensor transpose.
    pub int_tensor_transpose: TensorUnaryFn,
    /// Creates an int range tensor with step.
    pub int_tensor_arange_step: IntTensorArangeStepFn,
    /// Creates an int range tensor.
    pub int_tensor_arange: IntTensorArangeFn,
    /// Dispatches int tensor any reduction.
    pub int_tensor_any: TensorBoolReduceFn,
    /// Dispatches int tensor any-dim reduction.
    pub int_tensor_any_dim: TensorBoolReduceDimFn,
    /// Dispatches int tensor all reduction.
    pub int_tensor_all: TensorBoolReduceFn,
    /// Dispatches int tensor all-dim reduction.
    pub int_tensor_all_dim: TensorBoolReduceDimFn,
    /// Dispatches int tensor sign operation.
    pub int_tensor_sign: TensorUnaryFn,
    /// Dispatches int tensor sort operation.
    pub int_tensor_sort: TensorSortFn,
    /// Dispatches int tensor sort-with-indices operation.
    pub int_tensor_sort_with_indices: TensorSortWithIndicesNoDTypeFn,
    /// Dispatches int tensor argsort operation.
    pub int_tensor_argsort: TensorArgsortNoDTypeFn,
    /// Creates a bool tensor from host `u8` data.
    pub bool_tensor_from_u8_data: TensorFromU8DataFn,
    /// Materializes a bool tensor into host `u8` data.
    pub bool_tensor_into_u8_data: TensorIntoU8DataFn,
    /// Casts a bool tensor into an int tensor.
    pub bool_tensor_into_int: TensorIntoIntFn,
    /// Casts a bool tensor into a float tensor.
    pub bool_tensor_into_float: TensorCastFn,
    /// Moves a bool tensor to a target device.
    pub bool_tensor_to_device: TensorToDeviceFn,
    /// Creates an empty bool tensor.
    pub bool_tensor_empty: TensorEmptyBoolFn,
    /// Creates a bool tensor filled with zeros.
    pub bool_tensor_zeros: TensorEmptyBoolFn,
    /// Creates a bool tensor filled with ones.
    pub bool_tensor_ones: TensorEmptyBoolFn,
    /// Dispatches bool tensor reshape.
    pub bool_tensor_reshape: TensorReshapeFn,
    /// Dispatches bool tensor gather.
    pub bool_tensor_gather: TensorGatherFn,
    /// Dispatches bool tensor scatter or.
    pub bool_tensor_scatter_or: TensorScatterAddFn,
    /// Dispatches bool tensor select.
    pub bool_tensor_select: TensorSelectFn,
    /// Dispatches bool tensor select or.
    pub bool_tensor_select_or: TensorSelectAddFn,
    /// Dispatches bool tensor slice.
    pub bool_tensor_slice: TensorSliceFn,
    /// Dispatches bool tensor slice assign.
    pub bool_tensor_slice_assign: TensorSliceAssignFn,
    /// Dispatches bool tensor mask where.
    pub bool_tensor_mask_where: TensorMaskWhereFn,
    /// Dispatches bool tensor mask fill.
    pub bool_tensor_mask_fill: TensorMaskFillFn,
    /// Dispatches bool tensor equality comparison.
    pub bool_tensor_equal: TensorBoolCompareFn,
    /// Dispatches bool tensor-scalar equality comparison.
    pub bool_tensor_equal_elem: TensorBoolCompareScalarFn,
    /// Dispatches bool tensor logical not.
    pub bool_tensor_not: TensorUnaryFn,
    /// Dispatches bool tensor logical and.
    pub bool_tensor_and: TensorBinaryFn,
    /// Dispatches bool tensor logical or.
    pub bool_tensor_or: TensorBinaryFn,
    /// Dispatches bool tensor swap dims.
    pub bool_tensor_swap_dims: TensorSwapDimsFn,
    /// Dispatches bool tensor permute.
    pub bool_tensor_permute: TensorAxesFn,
    /// Dispatches bool tensor flip.
    pub bool_tensor_flip: TensorAxesFn,
    /// Dispatches bool tensor expand.
    pub bool_tensor_expand: TensorReshapeFn,
    /// Dispatches bool tensor unfold.
    pub bool_tensor_unfold: TensorUnfoldFn,
    /// Dispatches bool tensor repeat-dim.
    pub bool_tensor_repeat_dim: TensorRepeatDimFn,
    /// Dispatches bool tensor concatenation.
    pub bool_tensor_cat: TensorCatFn,
    /// Dispatches bool tensor non-equality comparison.
    pub bool_tensor_not_equal: TensorBoolCompareFn,
    /// Dispatches bool tensor-scalar non-equality comparison.
    pub bool_tensor_not_equal_elem: TensorBoolCompareScalarFn,
    /// Dispatches bool tensor xor.
    pub bool_tensor_xor: TensorBinaryFn,
    /// Dispatches bool tensor transpose.
    pub bool_tensor_transpose: TensorUnaryFn,
    /// Dispatches bool tensor any reduction.
    pub bool_tensor_any: TensorUnaryFn,
    /// Dispatches bool tensor any-dim reduction.
    pub bool_tensor_any_dim: BoolTensorDimFn,
    /// Dispatches bool tensor all reduction.
    pub bool_tensor_all: TensorUnaryFn,
    /// Dispatches bool tensor all-dim reduction.
    pub bool_tensor_all_dim: BoolTensorDimFn,
    /// Creates a quantized tensor from host `u8` bytes.
    pub q_tensor_from_u8_data: QTensorFromU8DataFn,
    /// Materializes a quantized tensor into host `u8` bytes.
    pub q_tensor_into_u8_data: QTensorIntoU8DataFn,
    /// Quantizes a float tensor.
    pub q_tensor_quantize: QTensorQuantizeFn,
    /// Dequantizes a quantized tensor.
    pub q_tensor_dequantize: QTensorDequantizeFn,
    /// Moves a quantized tensor to a target device.
    pub q_tensor_to_device: TensorToDeviceFn,
    /// Dispatches quantized tensor reshape.
    pub q_tensor_reshape: TensorReshapeFn,
    /// Dispatches quantized tensor expand.
    pub q_tensor_expand: TensorReshapeFn,
    /// Dispatches quantized tensor swap dims.
    pub q_tensor_swap_dims: TensorSwapDimsFn,
    /// Dispatches quantized tensor permute.
    pub q_tensor_permute: TensorAxesFn,
    /// Dispatches quantized tensor flip.
    pub q_tensor_flip: TensorAxesFn,
    /// Dispatches quantized tensor select.
    pub q_tensor_select: TensorSelectFn,
    /// Dispatches quantized tensor slice.
    pub q_tensor_slice: TensorSliceFn,
    /// Dispatches module embedding.
    pub module_embedding: ModuleEmbeddingFn,
    /// Dispatches module embedding backward.
    pub module_embedding_backward: ModuleEmbeddingBackwardFn,
    /// Dispatches module conv1d.
    pub module_conv1d: ModuleConv1dFn,
    /// Dispatches module conv1d input-backward.
    pub module_conv1d_x_backward: ModuleConv1dXBackwardFn,
    /// Dispatches module conv1d weight-backward.
    pub module_conv1d_weight_backward: ModuleConv1dWeightBackwardFn,
    /// Dispatches module conv1d bias-backward.
    pub module_conv1d_bias_backward: ModuleConv1dBiasBackwardFn,
    /// Dispatches module conv2d input-backward.
    pub module_conv2d_x_backward: ModuleConv2dXBackwardFn,
    /// Dispatches module conv2d weight-backward.
    pub module_conv2d_weight_backward: ModuleConv2dWeightBackwardFn,
    /// Dispatches module conv2d bias-backward.
    pub module_conv2d_bias_backward: ModuleConv2dBiasBackwardFn,
    /// Dispatches module conv3d input-backward.
    pub module_conv3d_x_backward: ModuleConv3dXBackwardFn,
    /// Dispatches module conv3d weight-backward.
    pub module_conv3d_weight_backward: ModuleConv3dWeightBackwardFn,
    /// Dispatches module conv3d bias-backward.
    pub module_conv3d_bias_backward: ModuleConv3dBiasBackwardFn,
    /// Dispatches module transposed conv1d.
    pub module_conv_transpose1d: ModuleConvTranspose1dFn,
    /// Dispatches module transposed conv1d input-backward.
    pub module_conv_transpose1d_x_backward: ModuleConvTranspose1dXBackwardFn,
    /// Dispatches module transposed conv1d weight-backward.
    pub module_conv_transpose1d_weight_backward: ModuleConvTranspose1dWeightBackwardFn,
    /// Dispatches module transposed conv1d bias-backward.
    pub module_conv_transpose1d_bias_backward: ModuleConvTranspose1dBiasBackwardFn,
    /// Dispatches module transposed conv2d input-backward.
    pub module_conv_transpose2d_x_backward: ModuleConvTranspose2dXBackwardFn,
    /// Dispatches module transposed conv2d weight-backward.
    pub module_conv_transpose2d_weight_backward: ModuleConvTranspose2dWeightBackwardFn,
    /// Dispatches module transposed conv2d bias-backward.
    pub module_conv_transpose2d_bias_backward: ModuleConvTranspose2dBiasBackwardFn,
    /// Dispatches module transposed conv3d input-backward.
    pub module_conv_transpose3d_x_backward: ModuleConvTranspose3dXBackwardFn,
    /// Dispatches module transposed conv3d weight-backward.
    pub module_conv_transpose3d_weight_backward: ModuleConvTranspose3dWeightBackwardFn,
    /// Dispatches module transposed conv3d bias-backward.
    pub module_conv_transpose3d_bias_backward: ModuleConvTranspose3dBiasBackwardFn,
    /// Dispatches module unfold4d.
    pub module_unfold4d: ModuleUnfold4dFn,
    /// Dispatches module avg-pool1d.
    pub module_avg_pool1d: ModuleAvgPool1dFn,
    /// Dispatches module avg-pool1d backward.
    pub module_avg_pool1d_backward: ModuleAvgPool1dBackwardFn,
    /// Dispatches module adaptive avg-pool1d.
    pub module_adaptive_avg_pool1d: ModuleAdaptiveAvgPool1dFn,
    /// Dispatches module adaptive avg-pool1d backward.
    pub module_adaptive_avg_pool1d_backward: ModuleAdaptiveAvgPool1dBackwardFn,
    /// Dispatches module max-pool1d.
    pub module_max_pool1d: ModuleMaxPool1dFn,
    /// Dispatches module max-pool1d with indices.
    pub module_max_pool1d_with_indices: ModuleMaxPool1dWithIndicesFn,
    /// Dispatches module max-pool1d with indices backward.
    pub module_max_pool1d_with_indices_backward: ModuleMaxPool1dBackwardFn,
    /// Dispatches module conv2d.
    pub module_conv2d: ModuleConv2dFn,
    /// Dispatches module deformable conv2d.
    pub module_deform_conv2d: ModuleDeformConv2dFn,
    /// Dispatches module deformable conv2d backward.
    pub module_deform_conv2d_backward: ModuleDeformConv2dBackwardFn,
    /// Dispatches module conv3d.
    pub module_conv3d: ModuleConv3dFn,
    /// Dispatches module transposed conv2d.
    pub module_conv_transpose2d: ModuleConvTranspose2dFn,
    /// Dispatches module transposed conv3d.
    pub module_conv_transpose3d: ModuleConvTranspose3dFn,
    /// Dispatches module avg-pool2d.
    pub module_avg_pool2d: ModuleAvgPool2dFn,
    /// Dispatches module avg-pool2d backward.
    pub module_avg_pool2d_backward: ModuleAvgPool2dBackwardFn,
    /// Dispatches module adaptive avg-pool2d.
    pub module_adaptive_avg_pool2d: ModuleAdaptiveAvgPool2dFn,
    /// Dispatches module adaptive avg-pool2d backward.
    pub module_adaptive_avg_pool2d_backward: ModuleAdaptiveAvgPool2dBackwardFn,
    /// Dispatches module max-pool2d.
    pub module_max_pool2d: ModuleMaxPool2dFn,
    /// Dispatches module max-pool2d with indices.
    pub module_max_pool2d_with_indices: ModuleMaxPool2dWithIndicesFn,
    /// Dispatches module max-pool2d with indices backward.
    pub module_max_pool2d_with_indices_backward: ModuleMaxPool2dBackwardFn,
    /// Dispatches module interpolate.
    pub module_interpolate: ModuleInterpolateFn,
    /// Dispatches module interpolate backward.
    pub module_interpolate_backward: ModuleInterpolateBackwardFn,
    /// Dispatches module attention.
    pub module_attention: ModuleAttentionFn,
    /// Dispatches module RFFT.
    pub module_rfft: ModuleRfftFn,
    /// Dispatches leaky ReLU activation.
    pub activation_leaky_relu: ActivationScalarFn,
    /// Dispatches ReLU activation.
    pub activation_relu: TensorUnaryFn,
    /// Dispatches ReLU backward activation.
    pub activation_relu_backward: TensorBinaryFn,
    /// Dispatches GELU activation.
    pub activation_gelu: TensorUnaryFn,
    /// Dispatches PReLU activation.
    pub activation_prelu: TensorBinaryFn,
    /// Dispatches GELU backward activation.
    pub activation_gelu_backward: TensorBinaryFn,
    /// Dispatches sigmoid activation.
    pub activation_sigmoid: TensorUnaryFn,
    /// Dispatches sigmoid backward activation.
    pub activation_sigmoid_backward: TensorBinaryFn,
    /// Dispatches hard-sigmoid activation.
    pub activation_hard_sigmoid: ActivationHardSigmoidFn,
    /// Dispatches log-sigmoid activation.
    pub activation_log_sigmoid: TensorUnaryFn,
    /// Dispatches log-sigmoid backward activation.
    pub activation_log_sigmoid_backward: TensorBinaryFn,
    /// Executes a read transaction for all tensor types in a single plugin call.
    pub transaction_execute: TransactionExecuteFn,
    /// Releases a tensor handle.
    pub release_tensor: TensorReleaseFn,
    /// Releases a plugin-allocated f32 buffer.
    pub release_f32_buffer: ReleaseF32BufferFn,
    /// Releases a plugin-allocated `u64` buffer.
    pub release_u64_buffer: ReleaseU64BufferFn,
    /// Releases a plugin-allocated `u8` buffer.
    pub release_u8_buffer: ReleaseU8BufferFn,
    /// Releases a plugin-allocated shape buffer.
    pub release_usize_buffer: ReleaseUsizeBufferFn,
}

/// Entrypoint function exported from plugin dynamic libraries for tensor operations.
pub type BackendTensorOpsEntrypoint = unsafe extern "C" fn() -> *const BackendTensorOpsV1;

/// Export the required plugin symbol.
///
/// # Example
///
/// ```ignore
/// use burn_dylib::{BackendPluginV1, BACKEND_PLUGIN_ABI_VERSION, export_backend_plugin_v1};
///
/// unsafe extern "C" fn backend_name() -> *const core::ffi::c_char {
///     b"my-backend\0".as_ptr().cast()
/// }
///
/// unsafe extern "C" fn seed(_seed: u64) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn sync() -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn device_count(_type_id: u16) -> usize {
///     1
/// }
///
/// static PLUGIN: BackendPluginV1 = BackendPluginV1 {
///     abi_version: BACKEND_PLUGIN_ABI_VERSION,
///     backend_name,
///     seed,
///     sync,
///     device_count,
/// };
///
/// export_backend_plugin_v1!(PLUGIN);
/// ```
#[macro_export]
macro_rules! export_backend_plugin_v1 {
    ($plugin:path) => {
        #[unsafe(no_mangle)]
        #[doc = "Exports the backend plugin descriptor symbol."]
        pub unsafe extern "C" fn burn_backend_plugin_v1() -> *const $crate::BackendPluginV1 {
            core::ptr::addr_of!($plugin)
        }
    };
}

/// Export the required tensor operations symbol.
///
/// # Example
///
/// ```ignore
/// use burn_dylib::{
///     BACKEND_TENSOR_OPS_ABI_VERSION, BackendTensorOpsV1, export_backend_tensor_ops_v1,
/// };
///
/// unsafe extern "C" fn create_device(
///     _type_id: u16,
///     _ordinal: usize,
///     _out_device: *mut burn_dylib::DeviceHandle,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn release_device(
///     _device: burn_dylib::DeviceHandle,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn tensor_from_f32_data(
///     _device: burn_dylib::DeviceHandle,
///     _shape: burn_dylib::TensorShapeRef,
///     _data: burn_dylib::F32SliceRef,
///     _out_tensor: *mut burn_dylib::TensorHandle,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn tensor_into_f32_data(
///     _tensor: burn_dylib::TensorHandle,
///     _out_data: *mut burn_dylib::OwnedF32Buffer,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn tensor_shape(
///     _tensor: burn_dylib::TensorHandle,
///     _out_shape: *mut burn_dylib::OwnedUsizeBuffer,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn tensor_add(
///     _lhs: burn_dylib::TensorHandle,
///     _rhs: burn_dylib::TensorHandle,
///     _out_tensor: *mut burn_dylib::TensorHandle,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn release_tensor(
///     _tensor: burn_dylib::TensorHandle,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn release_f32_buffer(
///     _buffer: burn_dylib::OwnedF32Buffer,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// unsafe extern "C" fn release_usize_buffer(
///     _buffer: burn_dylib::OwnedUsizeBuffer,
/// ) -> burn_dylib::PluginStatus {
///     burn_dylib::PluginStatus::ok()
/// }
///
/// static OPS: BackendTensorOpsV1 = BackendTensorOpsV1 {
///     abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
///     create_device,
///     release_device,
///     tensor_from_f32_data,
///     tensor_into_f32_data,
///     tensor_shape,
///     tensor_add,
///     release_tensor,
///     release_f32_buffer,
///     release_usize_buffer,
/// };
///
/// export_backend_tensor_ops_v1!(OPS);
/// ```
#[macro_export]
macro_rules! export_backend_tensor_ops_v1 {
    ($ops:path) => {
        #[unsafe(no_mangle)]
        #[doc = "Exports the backend tensor-operations descriptor symbol."]
        pub unsafe extern "C" fn burn_backend_tensor_ops_v1() -> *const $crate::BackendTensorOpsV1 {
            core::ptr::addr_of!($ops)
        }
    };
}

/// Export a backend plugin implementation.
///
/// The backend type must implement [`burn_backend::Backend`].
///
/// # Example
///
/// ```ignore
/// burn_dylib::export_plugin_api!(MyBackend, b"my-backend\0");
/// ```
#[cfg(feature = "std")]
#[macro_export]
macro_rules! export_plugin_api {
    ($backend:path, $name:expr) => {
        #[doc(hidden)]
        unsafe extern "C" fn __burn_dylib_backend_name() -> *const core::ffi::c_char {
            $name.as_ptr().cast()
        }

        static BURN_DYLIB_PLUGIN_V1: $crate::BackendPluginV1 =
            $crate::adapter::backend_plugin_v1::<$backend>(__burn_dylib_backend_name);
        static BURN_DYLIB_TENSOR_OPS_V1: $crate::BackendTensorOpsV1 =
            $crate::adapter::backend_tensor_ops_v1::<$backend>();

        $crate::export_backend_plugin_v1!(BURN_DYLIB_PLUGIN_V1);
        $crate::export_backend_tensor_ops_v1!(BURN_DYLIB_TENSOR_OPS_V1);
    };
}

/// Runtime dynamic library loader for plugin backends.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod loader {
    use super::{
        ABI_QUANT_BLOCK_MAX_DIMS, AbiAttentionModuleOptions, AbiBoolDType, AbiConvOptions1,
        AbiConvOptions2, AbiConvOptions3, AbiConvTransposeOptions1, AbiConvTransposeOptions2,
        AbiConvTransposeOptions3, AbiDeformConv2dBackward, AbiDeformConvOptions2, AbiDistribution,
        AbiDistributionKind, AbiFloatDType, AbiIntDType, AbiInterpolateMode, AbiInterpolateOptions,
        AbiMaxPool1dWithIndices, AbiMaxPool2dWithIndices, AbiQuantLevel, AbiQuantMode,
        AbiQuantParam, AbiQuantScheme, AbiQuantStore, AbiQuantValue, AbiRfftOutput, AbiScalar,
        AbiScalarKind, AbiSlice, AbiSliceRef, AbiTensorWithIndices, AbiUnfoldOptions,
        BACKEND_PLUGIN_ABI_VERSION, BACKEND_PLUGIN_SYMBOL, BACKEND_TENSOR_OPS_ABI_VERSION,
        BACKEND_TENSOR_OPS_SYMBOL, BackendPluginEntrypoint, BackendPluginV1,
        BackendTensorOpsEntrypoint, BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer,
        OwnedQTransactionItem, OwnedU8Buffer, OwnedU64Buffer, OwnedUsizeBuffer, PluginStatus,
        PluginStatusCode, TensorHandle, TensorHandleRef, TensorShapeRef, U8SliceRef, U64SliceRef,
    };
    use burn_backend::ops::{
        AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConvOptions,
        InterpolateMode, InterpolateOptions, UnfoldOptions,
    };
    use burn_backend::quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
    };
    use burn_backend::{BoolDType, Distribution, FloatDType, IntDType, Scalar, Slice};
    use core::ffi::c_char;
    use libloading::{Library, Symbol};
    use std::error::Error;
    use std::ffi::CStr;
    use std::fmt::{Display, Formatter};
    use std::path::Path;

    /// Errors while loading a backend plugin shared library.
    #[derive(Debug)]
    pub enum LoadError {
        /// Failed to load the shared library.
        Library(libloading::Error),
        /// The backend metadata entry symbol is missing or has an incompatible type.
        PluginSymbol(libloading::Error),
        /// The tensor operation entry symbol is missing or has an incompatible type.
        TensorOpsSymbol(libloading::Error),
        /// Entrypoint returned a null plugin pointer.
        NullPlugin,
        /// Tensor ops entrypoint returned a null pointer.
        NullTensorOps,
        /// Plugin ABI version does not match the host ABI version.
        AbiVersionMismatch {
            /// Host ABI version.
            expected: u32,
            /// Plugin ABI version.
            found: u32,
        },
        /// Tensor ops ABI version does not match the host ABI version.
        TensorOpsAbiVersionMismatch {
            /// Host ABI version.
            expected: u32,
            /// Plugin ABI version.
            found: u32,
        },
    }

    impl Display for LoadError {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::Library(err) => write!(f, "Failed to load shared library: {err}"),
                Self::PluginSymbol(err) => {
                    write!(f, "Failed to resolve plugin metadata symbol: {err}")
                }
                Self::TensorOpsSymbol(err) => {
                    write!(f, "Failed to resolve tensor ops symbol: {err}")
                }
                Self::NullPlugin => {
                    write!(f, "Plugin entrypoint returned a null backend descriptor")
                }
                Self::NullTensorOps => {
                    write!(f, "Tensor ops entrypoint returned a null descriptor")
                }
                Self::AbiVersionMismatch { expected, found } => {
                    write!(
                        f,
                        "Plugin ABI mismatch, expected version {expected} but found {found}",
                    )
                }
                Self::TensorOpsAbiVersionMismatch { expected, found } => {
                    write!(
                        f,
                        "Tensor ops ABI mismatch, expected version {expected} but found {found}",
                    )
                }
            }
        }
    }

    impl Error for LoadError {}

    /// Errors while invoking plugin functions.
    #[derive(Debug)]
    pub enum PluginCallError {
        /// Plugin returned a null C string pointer.
        NullPointer(&'static str),
        /// Plugin returned an invalid device or tensor handle.
        InvalidHandle(&'static str),
        /// Plugin returned an invalid UTF-8 string.
        InvalidUtf8(std::str::Utf8Error),
        /// Plugin reported a failing status.
        Failure {
            /// Plugin status code.
            code: PluginStatusCode,
            /// Error message.
            message: String,
        },
    }

    impl Display for PluginCallError {
        fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::NullPointer(context) => {
                    write!(f, "Plugin returned a null pointer for {context}")
                }
                Self::InvalidHandle(context) => {
                    write!(f, "Plugin returned an invalid handle for {context}")
                }
                Self::InvalidUtf8(err) => write!(f, "Plugin returned invalid UTF-8: {err}"),
                Self::Failure { code, message } => {
                    write!(f, "Plugin call failed with code {code:?}: {message}")
                }
            }
        }
    }

    impl Error for PluginCallError {}

    /// Handles produced by deformable-convolution backward.
    #[derive(Debug, Clone, Copy)]
    pub struct DeformConv2dBackwardHandles {
        /// Gradient for the input tensor.
        pub x_grad: TensorHandle,
        /// Gradient for the offset tensor.
        pub offset_grad: TensorHandle,
        /// Gradient for the weight tensor.
        pub weight_grad: TensorHandle,
        /// Optional gradient for the mask tensor.
        pub mask_grad: Option<TensorHandle>,
        /// Optional gradient for the bias tensor.
        pub bias_grad: Option<TensorHandle>,
    }

    /// Handles produced by max-pool2d-with-indices.
    #[derive(Debug, Clone, Copy)]
    pub struct MaxPool2dWithIndicesHandles {
        /// Output tensor handle.
        pub output: TensorHandle,
        /// Indices tensor handle.
        pub indices: TensorHandle,
    }

    /// Handles produced by max-pool1d-with-indices.
    #[derive(Debug, Clone, Copy)]
    pub struct MaxPool1dWithIndicesHandles {
        /// Output tensor handle.
        pub output: TensorHandle,
        /// Indices tensor handle.
        pub indices: TensorHandle,
    }

    /// Handles produced by operations returning values and indices.
    #[derive(Debug, Clone, Copy)]
    pub struct TensorWithIndicesHandles {
        /// Values tensor handle.
        pub values: TensorHandle,
        /// Indices tensor handle.
        pub indices: TensorHandle,
    }

    /// Handles produced by RFFT.
    #[derive(Debug, Clone, Copy)]
    pub struct RfftHandles {
        /// Real output tensor handle.
        pub real: TensorHandle,
        /// Imaginary output tensor handle.
        pub imag: TensorHandle,
    }

    /// A loaded backend plugin.
    ///
    /// The underlying dynamic library handle is retained for the whole lifetime of this object,
    /// ensuring plugin function pointers remain valid.
    pub struct LoadedBackendPlugin {
        library: Library,
        plugin: *const BackendPluginV1,
        tensor_ops: *const BackendTensorOpsV1,
    }

    // Safety: The plugin ABI contract requires callback tables and symbols to be immutable and
    // process-wide for the full lifetime of the loaded library. Calls are delegated through
    // function pointers and the library handle remains alive in this struct.
    unsafe impl Send for LoadedBackendPlugin {}
    unsafe impl Sync for LoadedBackendPlugin {}

    macro_rules! loader_unary_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(&self, tensor: TensorHandle) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, out)
                })
            }
        };
    }

    macro_rules! loader_binary_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                lhs: TensorHandle,
                rhs: TensorHandle,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(lhs, rhs, out)
                })
            }
        };
    }

    macro_rules! loader_scalar_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                scalar: Scalar,
            ) -> Result<TensorHandle, PluginCallError> {
                let scalar = scalar_to_abi(scalar);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, scalar, out)
                })
            }
        };
    }

    macro_rules! loader_dim_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, out)
                })
            }
        };
    }

    macro_rules! loader_compare_binary_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                lhs: TensorHandle,
                rhs: TensorHandle,
                out_dtype: BoolDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let out_dtype = bool_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(lhs, rhs, out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_compare_scalar_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                rhs: Scalar,
                out_dtype: BoolDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let rhs = scalar_to_abi(rhs);
                let out_dtype = bool_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, rhs, out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_arg_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                out_dtype: IntDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let out_dtype = int_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_clamp_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                min: Scalar,
                max: Scalar,
            ) -> Result<TensorHandle, PluginCallError> {
                let min = scalar_to_abi(min);
                let max = scalar_to_abi(max);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, min, max, out)
                })
            }
        };
    }

    macro_rules! loader_repeat_dim_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                times: usize,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, times, out)
                })
            }
        };
    }

    macro_rules! loader_cat_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensors: &[TensorHandle],
                dim: usize,
            ) -> Result<TensorHandle, PluginCallError> {
                let tensors_ref = TensorHandleRef {
                    ptr: tensors.as_ptr(),
                    len: tensors.len(),
                };
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensors_ref, dim, out)
                })
            }
        };
    }

    macro_rules! loader_bool_reduce_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                out_dtype: BoolDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let out_dtype = bool_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_bool_reduce_dim_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                out_dtype: BoolDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let out_dtype = bool_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_bool_dim_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, out)
                })
            }
        };
    }

    macro_rules! loader_with_indices_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                out_dtype: IntDType,
            ) -> Result<TensorWithIndicesHandles, PluginCallError> {
                let out_dtype = int_dtype_to_abi(out_dtype);
                let mut out = AbiTensorWithIndices {
                    values: TensorHandle::INVALID,
                    indices: TensorHandle::INVALID,
                };
                let status =
                    unsafe { (self.tensor_ops().$field)(tensor, dim, out_dtype, &mut out) };
                check_status(status)?;

                if !out.values.is_valid() || !out.indices.is_valid() {
                    return Err(PluginCallError::InvalidHandle("values_with_indices"));
                }

                Ok(TensorWithIndicesHandles {
                    values: out.values,
                    indices: out.indices,
                })
            }
        };
    }

    macro_rules! loader_with_indices_no_dtype_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
            ) -> Result<TensorWithIndicesHandles, PluginCallError> {
                let mut out = AbiTensorWithIndices {
                    values: TensorHandle::INVALID,
                    indices: TensorHandle::INVALID,
                };
                let status = unsafe { (self.tensor_ops().$field)(tensor, dim, &mut out) };
                check_status(status)?;

                if !out.values.is_valid() || !out.indices.is_valid() {
                    return Err(PluginCallError::InvalidHandle("values_with_indices"));
                }

                Ok(TensorWithIndicesHandles {
                    values: out.values,
                    indices: out.indices,
                })
            }
        };
    }

    macro_rules! loader_sort_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                descending: bool,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, u8::from(descending), out)
                })
            }
        };
    }

    macro_rules! loader_sort_with_indices_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                descending: bool,
                out_dtype: IntDType,
            ) -> Result<TensorWithIndicesHandles, PluginCallError> {
                let out_dtype = int_dtype_to_abi(out_dtype);
                let mut out = AbiTensorWithIndices {
                    values: TensorHandle::INVALID,
                    indices: TensorHandle::INVALID,
                };
                let status = unsafe {
                    (self.tensor_ops().$field)(
                        tensor,
                        dim,
                        u8::from(descending),
                        out_dtype,
                        &mut out,
                    )
                };
                check_status(status)?;

                if !out.values.is_valid() || !out.indices.is_valid() {
                    return Err(PluginCallError::InvalidHandle("sort_with_indices"));
                }

                Ok(TensorWithIndicesHandles {
                    values: out.values,
                    indices: out.indices,
                })
            }
        };
    }

    macro_rules! loader_sort_with_indices_no_dtype_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                descending: bool,
            ) -> Result<TensorWithIndicesHandles, PluginCallError> {
                let mut out = AbiTensorWithIndices {
                    values: TensorHandle::INVALID,
                    indices: TensorHandle::INVALID,
                };
                let status = unsafe {
                    (self.tensor_ops().$field)(tensor, dim, u8::from(descending), &mut out)
                };
                check_status(status)?;

                if !out.values.is_valid() || !out.indices.is_valid() {
                    return Err(PluginCallError::InvalidHandle("sort_with_indices"));
                }

                Ok(TensorWithIndicesHandles {
                    values: out.values,
                    indices: out.indices,
                })
            }
        };
    }

    macro_rules! loader_argsort_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                descending: bool,
                out_dtype: IntDType,
            ) -> Result<TensorHandle, PluginCallError> {
                let out_dtype = int_dtype_to_abi(out_dtype);
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, u8::from(descending), out_dtype, out)
                })
            }
        };
    }

    macro_rules! loader_argsort_no_dtype_method {
        ($name:ident, $field:ident) => {
            #[allow(missing_docs)]
            pub fn $name(
                &self,
                tensor: TensorHandle,
                dim: usize,
                descending: bool,
            ) -> Result<TensorHandle, PluginCallError> {
                self.call_with_out_handle("tensor", |out| unsafe {
                    (self.tensor_ops().$field)(tensor, dim, u8::from(descending), out)
                })
            }
        };
    }

    impl LoadedBackendPlugin {
        /// Loads a backend plugin from a shared library file.
        ///
        /// # Safety
        ///
        /// The library must export `burn_backend_plugin_v1` with the expected ABI and all callback
        /// functions in the table must uphold their own contracts.
        pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
            let library = unsafe { Library::new(path.as_ref()) }.map_err(LoadError::Library)?;
            let entrypoint: Symbol<'_, BackendPluginEntrypoint> =
                unsafe { library.get(BACKEND_PLUGIN_SYMBOL) }.map_err(LoadError::PluginSymbol)?;

            let tensor_ops_entrypoint: Symbol<'_, BackendTensorOpsEntrypoint> =
                unsafe { library.get(BACKEND_TENSOR_OPS_SYMBOL) }
                    .map_err(LoadError::TensorOpsSymbol)?;

            let plugin = unsafe { entrypoint() };
            if plugin.is_null() {
                return Err(LoadError::NullPlugin);
            }

            let tensor_ops = unsafe { tensor_ops_entrypoint() };
            if tensor_ops.is_null() {
                return Err(LoadError::NullTensorOps);
            }

            // Safety: checked for null above and kept valid by owning `library` in this struct.
            let api = unsafe { &*plugin };
            if api.abi_version != BACKEND_PLUGIN_ABI_VERSION {
                return Err(LoadError::AbiVersionMismatch {
                    expected: BACKEND_PLUGIN_ABI_VERSION,
                    found: api.abi_version,
                });
            }

            // Safety: checked for null above and kept valid by owning `library` in this struct.
            let tensor_ops_api = unsafe { &*tensor_ops };
            if tensor_ops_api.abi_version != BACKEND_TENSOR_OPS_ABI_VERSION {
                return Err(LoadError::TensorOpsAbiVersionMismatch {
                    expected: BACKEND_TENSOR_OPS_ABI_VERSION,
                    found: tensor_ops_api.abi_version,
                });
            }

            Ok(Self {
                library,
                plugin,
                tensor_ops,
            })
        }

        /// Returns the backend name from the plugin.
        pub fn backend_name(&self) -> Result<String, PluginCallError> {
            let ptr = unsafe { (self.api().backend_name)() };
            read_c_string(ptr, "backend_name")
        }

        /// Forwards a seed value to the loaded backend.
        pub fn backend_seed(&self, seed: u64) -> Result<(), PluginCallError> {
            let status = unsafe { (self.api().seed)(seed) };
            check_status(status)
        }

        /// Synchronizes all pending operations on the backend.
        pub fn backend_sync(&self) -> Result<(), PluginCallError> {
            let status = unsafe { (self.api().sync)() };
            check_status(status)
        }

        /// Returns the number of devices for the provided backend type identifier.
        pub fn device_count(&self, type_id: u16) -> usize {
            unsafe { (self.api().device_count)(type_id) }
        }

        /// Creates a default backend device handle.
        pub fn create_default_device(&self) -> Result<(u16, usize, DeviceHandle), PluginCallError> {
            let mut type_id = 0;
            let mut ordinal = 0;
            let mut handle = DeviceHandle::INVALID;
            let status = unsafe {
                (self.tensor_ops().create_default_device)(&mut type_id, &mut ordinal, &mut handle)
            };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("device"));
            }
            Ok((type_id, ordinal, handle))
        }

        /// Creates a backend device handle.
        pub fn create_device(
            &self,
            type_id: u16,
            ordinal: usize,
        ) -> Result<DeviceHandle, PluginCallError> {
            let mut handle = DeviceHandle::INVALID;
            let status =
                unsafe { (self.tensor_ops().create_device)(type_id, ordinal, &mut handle) };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("device"));
            }
            Ok(handle)
        }

        /// Releases a backend device handle.
        pub fn release_device(&self, device: DeviceHandle) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_device)(device) };
            check_status(status)
        }

        /// Creates a float tensor from f32 data and shape.
        pub fn float_tensor_from_f32_data(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            data: &[f32],
        ) -> Result<TensorHandle, PluginCallError> {
            let mut handle = TensorHandle::INVALID;
            let shape_ref = TensorShapeRef {
                dims: shape.as_ptr(),
                rank: shape.len(),
            };
            let data_ref = F32SliceRef {
                ptr: data.as_ptr(),
                len: data.len(),
            };
            let status = unsafe {
                (self.tensor_ops().tensor_from_f32_data)(device, shape_ref, data_ref, &mut handle)
            };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("tensor"));
            }
            Ok(handle)
        }

        /// Reads a float tensor as a host f32 vector.
        pub fn float_tensor_into_f32_data(
            &self,
            tensor: TensorHandle,
        ) -> Result<Vec<f32>, PluginCallError> {
            let mut buffer = OwnedF32Buffer::empty();
            let status = unsafe { (self.tensor_ops().tensor_into_f32_data)(tensor, &mut buffer) };
            check_status(status)?;

            if buffer.len == 0 {
                return Ok(Vec::new());
            }
            if buffer.ptr.is_null() {
                return Err(PluginCallError::NullPointer("tensor_into_f32_data"));
            }

            let values = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
            self.release_f32_buffer(buffer)?;
            Ok(values)
        }

        /// Reads the tensor shape into a host vector.
        pub fn float_tensor_shape(
            &self,
            tensor: TensorHandle,
        ) -> Result<Vec<usize>, PluginCallError> {
            let mut buffer = OwnedUsizeBuffer::empty();
            let status = unsafe { (self.tensor_ops().tensor_shape)(tensor, &mut buffer) };
            check_status(status)?;

            if buffer.len == 0 {
                return Ok(Vec::new());
            }
            if buffer.ptr.is_null() {
                return Err(PluginCallError::NullPointer("tensor_shape"));
            }

            let shape = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
            self.release_usize_buffer(buffer)?;
            Ok(shape)
        }

        /// Creates a random float tensor and returns a newly allocated tensor handle.
        pub fn float_tensor_random(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            distribution: Distribution,
            dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let distribution = distribution_to_abi(distribution);
            let dtype = float_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_random)(device, shape_ref, distribution, dtype, out)
            })
        }

        /// Moves a tensor to a different backend device.
        pub fn float_tensor_to_device(
            &self,
            tensor: TensorHandle,
            device: DeviceHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_to_device)(tensor, device, out)
            })
        }

        /// Creates an empty float tensor.
        pub fn float_tensor_empty(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = float_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_empty)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a float tensor filled with zeros.
        pub fn float_tensor_zeros(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = float_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_zeros)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a float tensor filled with ones.
        pub fn float_tensor_ones(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = float_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_ones)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a float tensor filled with a scalar value.
        pub fn float_tensor_full(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            value: Scalar,
            dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let value = scalar_to_abi(value);
            let dtype = float_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_full)(device, shape_ref, value, dtype, out)
            })
        }

        /// Casts a float tensor into an int tensor.
        pub fn float_tensor_into_int(
            &self,
            tensor: TensorHandle,
            out_dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = int_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_into_int)(tensor, out_dtype, out)
            })
        }

        /// Adds two float tensors and returns a newly allocated tensor handle.
        pub fn float_tensor_add(
            &self,
            lhs: TensorHandle,
            rhs: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_add)(lhs, rhs, out)
            })
        }

        loader_scalar_method!(float_tensor_add_scalar, tensor_add_scalar);
        loader_binary_method!(float_tensor_sub, tensor_sub);
        loader_scalar_method!(float_tensor_sub_scalar, tensor_sub_scalar);
        loader_binary_method!(float_tensor_mul, tensor_mul);
        loader_scalar_method!(float_tensor_mul_scalar, tensor_mul_scalar);
        loader_binary_method!(float_tensor_div, tensor_div);
        loader_scalar_method!(float_tensor_div_scalar, tensor_div_scalar);
        loader_binary_method!(float_tensor_remainder, tensor_remainder);
        loader_scalar_method!(float_tensor_remainder_scalar, tensor_remainder_scalar);
        loader_binary_method!(float_tensor_matmul, tensor_matmul);

        /// Computes the cross product over `dim`.
        pub fn float_tensor_cross(
            &self,
            lhs: TensorHandle,
            rhs: TensorHandle,
            dim: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_cross)(lhs, rhs, dim, out)
            })
        }

        loader_unary_method!(float_tensor_recip, tensor_recip);

        /// Swaps two dimensions on a tensor.
        pub fn float_tensor_swap_dims(
            &self,
            tensor: TensorHandle,
            dim1: usize,
            dim2: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_swap_dims)(tensor, dim1, dim2, out)
            })
        }

        /// Permutes tensor dimensions using `axes`.
        pub fn float_tensor_permute(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_permute)(tensor, axes_ref, out)
            })
        }

        /// Flips tensor dimensions listed in `axes`.
        pub fn float_tensor_flip(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_flip)(tensor, axes_ref, out)
            })
        }

        /// Reshapes a tensor.
        pub fn float_tensor_reshape(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_reshape)(tensor, shape_ref, out)
            })
        }

        /// Gathers values from a tensor using index tensor.
        pub fn float_tensor_gather(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_gather)(dim, tensor, indices, out)
            })
        }

        /// Adds `value` into `tensor` at indexed locations.
        pub fn float_tensor_scatter_add(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_scatter_add)(dim, tensor, indices, value, out)
            })
        }

        /// Selects values from a tensor using rank-1 indices.
        pub fn float_tensor_select(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_select)(tensor, dim, indices, out)
            })
        }

        /// Adds selected values into a tensor.
        pub fn float_tensor_select_add(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_select_add)(tensor, dim, indices, value, out)
            })
        }

        /// Slices a tensor.
        pub fn float_tensor_slice(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_slice)(tensor, slices_ref, out)
            })
        }

        /// Assigns a tensor into a slice view.
        pub fn float_tensor_slice_assign(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_slice_assign)(tensor, slices_ref, value, out)
            })
        }

        /// Selects values from `tensor` where `mask` is true.
        pub fn float_tensor_mask_where(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_mask_where)(tensor, mask, value, out)
            })
        }

        /// Fills values in `tensor` where `mask` is true.
        pub fn float_tensor_mask_fill(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: Scalar,
        ) -> Result<TensorHandle, PluginCallError> {
            let value = scalar_to_abi(value);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_mask_fill)(tensor, mask, value, out)
            })
        }

        loader_compare_binary_method!(float_tensor_equal, tensor_equal);
        loader_compare_scalar_method!(float_tensor_equal_elem, tensor_equal_elem);
        loader_compare_binary_method!(float_tensor_greater, tensor_greater);
        loader_compare_scalar_method!(float_tensor_greater_elem, tensor_greater_elem);
        loader_compare_binary_method!(float_tensor_greater_equal, tensor_greater_equal);
        loader_compare_scalar_method!(float_tensor_greater_equal_elem, tensor_greater_equal_elem);
        loader_compare_binary_method!(float_tensor_lower, tensor_lower);
        loader_compare_scalar_method!(float_tensor_lower_elem, tensor_lower_elem);
        loader_compare_binary_method!(float_tensor_lower_equal, tensor_lower_equal);
        loader_compare_scalar_method!(float_tensor_lower_equal_elem, tensor_lower_equal_elem);
        loader_compare_binary_method!(float_tensor_not_equal, tensor_not_equal);
        loader_compare_scalar_method!(float_tensor_not_equal_elem, tensor_not_equal_elem);

        loader_unary_method!(float_tensor_sum, tensor_sum);
        loader_dim_method!(float_tensor_sum_dim, tensor_sum_dim);
        loader_unary_method!(float_tensor_prod, tensor_prod);
        loader_dim_method!(float_tensor_prod_dim, tensor_prod_dim);
        loader_unary_method!(float_tensor_mean, tensor_mean);
        loader_dim_method!(float_tensor_mean_dim, tensor_mean_dim);
        loader_dim_method!(float_tensor_cumsum, tensor_cumsum);
        loader_dim_method!(float_tensor_cumprod, tensor_cumprod);
        loader_dim_method!(float_tensor_cummin, tensor_cummin);
        loader_dim_method!(float_tensor_cummax, tensor_cummax);

        /// Casts a tensor to a different float dtype.
        pub fn float_tensor_cast(
            &self,
            tensor: TensorHandle,
            out_dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = float_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_cast)(tensor, out_dtype, out)
            })
        }

        loader_unary_method!(float_tensor_exp, tensor_exp);
        loader_unary_method!(float_tensor_log, tensor_log);
        loader_unary_method!(float_tensor_log1p, tensor_log1p);
        loader_binary_method!(float_tensor_powf, tensor_powf);
        loader_scalar_method!(float_tensor_powf_scalar, tensor_powf_scalar);
        loader_binary_method!(float_tensor_powi, tensor_powi);
        loader_scalar_method!(float_tensor_powi_scalar, tensor_powi_scalar);
        loader_unary_method!(float_tensor_sqrt, tensor_sqrt);
        loader_unary_method!(float_tensor_abs, tensor_abs);
        loader_unary_method!(float_tensor_cos, tensor_cos);
        loader_unary_method!(float_tensor_sin, tensor_sin);
        loader_unary_method!(float_tensor_tan, tensor_tan);
        loader_unary_method!(float_tensor_cosh, tensor_cosh);
        loader_unary_method!(float_tensor_sinh, tensor_sinh);
        loader_unary_method!(float_tensor_tanh, tensor_tanh);
        loader_unary_method!(float_tensor_acos, tensor_acos);
        loader_unary_method!(float_tensor_acosh, tensor_acosh);
        loader_unary_method!(float_tensor_asin, tensor_asin);
        loader_unary_method!(float_tensor_asinh, tensor_asinh);
        loader_unary_method!(float_tensor_atan, tensor_atan);
        loader_unary_method!(float_tensor_atanh, tensor_atanh);
        loader_binary_method!(float_tensor_atan2, tensor_atan2);
        loader_unary_method!(float_tensor_round, tensor_round);
        loader_unary_method!(float_tensor_floor, tensor_floor);
        loader_unary_method!(float_tensor_ceil, tensor_ceil);
        loader_unary_method!(float_tensor_trunc, tensor_trunc);
        loader_unary_method!(float_tensor_erf, tensor_erf);

        loader_arg_method!(float_tensor_argmax, tensor_argmax);
        loader_arg_method!(float_tensor_argmin, tensor_argmin);

        /// Expands a tensor to a broadcast-compatible shape.
        pub fn float_tensor_expand(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_expand)(tensor, shape_ref, out)
            })
        }

        /// Unfolds a tensor along one dimension.
        pub fn float_tensor_unfold(
            &self,
            tensor: TensorHandle,
            dim: usize,
            size: usize,
            step: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().tensor_unfold)(tensor, dim, size, step, out)
            })
        }

        loader_repeat_dim_method!(float_tensor_repeat_dim, tensor_repeat_dim);
        loader_scalar_method!(float_tensor_clamp_min, tensor_clamp_min);
        loader_scalar_method!(float_tensor_clamp_max, tensor_clamp_max);
        loader_clamp_method!(float_tensor_clamp, tensor_clamp);
        loader_unary_method!(float_tensor_neg, tensor_neg);
        loader_unary_method!(float_tensor_transpose, tensor_transpose);
        loader_cat_method!(float_tensor_cat, tensor_cat);
        loader_unary_method!(float_tensor_max, tensor_max);
        loader_dim_method!(float_tensor_max_dim, tensor_max_dim);
        loader_with_indices_method!(
            float_tensor_max_dim_with_indices,
            tensor_max_dim_with_indices
        );
        loader_unary_method!(float_tensor_min, tensor_min);
        loader_dim_method!(float_tensor_min_dim, tensor_min_dim);
        loader_with_indices_method!(
            float_tensor_min_dim_with_indices,
            tensor_min_dim_with_indices
        );
        loader_unary_method!(float_tensor_max_abs, tensor_max_abs);
        loader_dim_method!(float_tensor_max_abs_dim, tensor_max_abs_dim);
        loader_bool_reduce_method!(float_tensor_any, tensor_any);
        loader_bool_reduce_dim_method!(float_tensor_any_dim, tensor_any_dim);
        loader_bool_reduce_method!(float_tensor_all, tensor_all);
        loader_bool_reduce_dim_method!(float_tensor_all_dim, tensor_all_dim);
        loader_unary_method!(float_tensor_sign, tensor_sign);
        loader_sort_method!(float_tensor_sort, tensor_sort);
        loader_sort_with_indices_method!(float_tensor_sort_with_indices, tensor_sort_with_indices);
        loader_argsort_method!(float_tensor_argsort, tensor_argsort);
        loader_bool_reduce_method!(float_tensor_is_nan, tensor_is_nan);
        loader_bool_reduce_method!(float_tensor_is_inf, tensor_is_inf);

        /// Creates an int tensor from host `u64` data and shape.
        pub fn int_tensor_from_u64_data(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            data: &[u64],
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let mut handle = TensorHandle::INVALID;
            let shape_ref = shape_ref(shape);
            let data_ref = U64SliceRef {
                ptr: data.as_ptr(),
                len: data.len(),
            };
            let dtype = int_dtype_to_abi(dtype);
            let status = unsafe {
                (self.tensor_ops().int_tensor_from_u64_data)(
                    device,
                    shape_ref,
                    data_ref,
                    dtype,
                    &mut handle,
                )
            };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("tensor"));
            }
            Ok(handle)
        }

        /// Reads an int tensor as a host `u64` vector.
        pub fn int_tensor_into_u64_data(
            &self,
            tensor: TensorHandle,
        ) -> Result<Vec<u64>, PluginCallError> {
            let mut buffer = OwnedU64Buffer::empty();
            let status =
                unsafe { (self.tensor_ops().int_tensor_into_u64_data)(tensor, &mut buffer) };
            check_status(status)?;

            if buffer.len == 0 {
                return Ok(Vec::new());
            }
            if buffer.ptr.is_null() {
                return Err(PluginCallError::NullPointer("int_tensor_into_u64_data"));
            }

            let values = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
            self.release_u64_buffer(buffer)?;
            Ok(values)
        }

        /// Creates a bool tensor from host `u8` data and shape.
        pub fn bool_tensor_from_u8_data(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            data: &[u8],
            dtype: BoolDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let mut handle = TensorHandle::INVALID;
            let shape_ref = shape_ref(shape);
            let data_ref = U8SliceRef {
                ptr: data.as_ptr(),
                len: data.len(),
            };
            let dtype = bool_dtype_to_abi(dtype);
            let status = unsafe {
                (self.tensor_ops().bool_tensor_from_u8_data)(
                    device,
                    shape_ref,
                    data_ref,
                    dtype,
                    &mut handle,
                )
            };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("tensor"));
            }
            Ok(handle)
        }

        /// Reads a bool tensor as a host `u8` vector.
        pub fn bool_tensor_into_u8_data(
            &self,
            tensor: TensorHandle,
        ) -> Result<Vec<u8>, PluginCallError> {
            let mut buffer = OwnedU8Buffer::empty();
            let status =
                unsafe { (self.tensor_ops().bool_tensor_into_u8_data)(tensor, &mut buffer) };
            check_status(status)?;

            if buffer.len == 0 {
                return Ok(Vec::new());
            }
            if buffer.ptr.is_null() {
                return Err(PluginCallError::NullPointer("bool_tensor_into_u8_data"));
            }

            let values = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
            self.release_u8_buffer(buffer)?;
            Ok(values)
        }

        /// Moves an int tensor to a different backend device.
        pub fn int_tensor_to_device(
            &self,
            tensor: TensorHandle,
            device: DeviceHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_to_device)(tensor, device, out)
            })
        }

        /// Creates an empty int tensor.
        pub fn int_tensor_empty(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_empty)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a random int tensor.
        pub fn int_tensor_random(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            distribution: Distribution,
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let distribution = distribution_to_abi(distribution);
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_random)(device, shape_ref, distribution, dtype, out)
            })
        }

        /// Casts an int tensor into a float tensor.
        pub fn int_tensor_into_float(
            &self,
            tensor: TensorHandle,
            out_dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = float_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_into_float)(tensor, out_dtype, out)
            })
        }

        /// Casts an int tensor to another int dtype.
        pub fn int_tensor_cast(
            &self,
            tensor: TensorHandle,
            out_dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = int_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_cast)(tensor, out_dtype, out)
            })
        }

        loader_binary_method!(int_tensor_add, int_tensor_add);
        loader_scalar_method!(int_tensor_add_scalar, int_tensor_add_scalar);
        loader_binary_method!(int_tensor_sub, int_tensor_sub);
        loader_scalar_method!(int_tensor_sub_scalar, int_tensor_sub_scalar);
        loader_binary_method!(int_tensor_mul, int_tensor_mul);
        loader_scalar_method!(int_tensor_mul_scalar, int_tensor_mul_scalar);
        loader_binary_method!(int_tensor_div, int_tensor_div);
        loader_scalar_method!(int_tensor_div_scalar, int_tensor_div_scalar);
        loader_binary_method!(int_tensor_remainder, int_tensor_remainder);
        loader_scalar_method!(int_tensor_remainder_scalar, int_tensor_remainder_scalar);
        loader_binary_method!(int_tensor_matmul, int_tensor_matmul);
        loader_unary_method!(int_tensor_abs, int_tensor_abs);
        loader_unary_method!(int_tensor_sum, int_tensor_sum);
        loader_dim_method!(int_tensor_sum_dim, int_tensor_sum_dim);
        loader_unary_method!(int_tensor_prod, int_tensor_prod);
        loader_dim_method!(int_tensor_prod_dim, int_tensor_prod_dim);
        loader_dim_method!(int_tensor_mean_dim, int_tensor_mean_dim);
        loader_dim_method!(int_tensor_cumsum, int_tensor_cumsum);
        loader_dim_method!(int_tensor_cumprod, int_tensor_cumprod);
        loader_dim_method!(int_tensor_cummin, int_tensor_cummin);
        loader_dim_method!(int_tensor_cummax, int_tensor_cummax);
        loader_dim_method!(int_tensor_argmax, int_tensor_argmax);
        loader_dim_method!(int_tensor_argmin, int_tensor_argmin);

        /// Swaps two dimensions on an int tensor.
        pub fn int_tensor_swap_dims(
            &self,
            tensor: TensorHandle,
            dim1: usize,
            dim2: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_swap_dims)(tensor, dim1, dim2, out)
            })
        }

        /// Permutes int tensor dimensions using `axes`.
        pub fn int_tensor_permute(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_permute)(tensor, axes_ref, out)
            })
        }

        /// Flips int tensor dimensions listed in `axes`.
        pub fn int_tensor_flip(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_flip)(tensor, axes_ref, out)
            })
        }

        /// Reshapes an int tensor.
        pub fn int_tensor_reshape(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_reshape)(tensor, shape_ref, out)
            })
        }

        /// Gathers values from an int tensor using index tensor.
        pub fn int_tensor_gather(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_gather)(dim, tensor, indices, out)
            })
        }

        /// Adds `value` into an int tensor at indexed locations.
        pub fn int_tensor_scatter_add(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_scatter_add)(dim, tensor, indices, value, out)
            })
        }

        /// Selects values from an int tensor using rank-1 indices.
        pub fn int_tensor_select(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_select)(tensor, dim, indices, out)
            })
        }

        /// Adds selected values into an int tensor.
        pub fn int_tensor_select_add(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_select_add)(tensor, dim, indices, value, out)
            })
        }

        /// Slices an int tensor.
        pub fn int_tensor_slice(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_slice)(tensor, slices_ref, out)
            })
        }

        /// Assigns an int tensor into a slice view.
        pub fn int_tensor_slice_assign(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_slice_assign)(tensor, slices_ref, value, out)
            })
        }

        /// Selects values from int tensor where `mask` is true.
        pub fn int_tensor_mask_where(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_mask_where)(tensor, mask, value, out)
            })
        }

        /// Fills values in int tensor where `mask` is true.
        pub fn int_tensor_mask_fill(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: Scalar,
        ) -> Result<TensorHandle, PluginCallError> {
            let value = scalar_to_abi(value);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_mask_fill)(tensor, mask, value, out)
            })
        }

        loader_compare_binary_method!(int_tensor_equal, int_tensor_equal);
        loader_compare_scalar_method!(int_tensor_equal_elem, int_tensor_equal_elem);
        loader_compare_binary_method!(int_tensor_greater, int_tensor_greater);
        loader_compare_scalar_method!(int_tensor_greater_elem, int_tensor_greater_elem);
        loader_compare_binary_method!(int_tensor_greater_equal, int_tensor_greater_equal);
        loader_compare_scalar_method!(int_tensor_greater_equal_elem, int_tensor_greater_equal_elem);
        loader_compare_binary_method!(int_tensor_lower, int_tensor_lower);
        loader_compare_scalar_method!(int_tensor_lower_elem, int_tensor_lower_elem);
        loader_compare_binary_method!(int_tensor_lower_equal, int_tensor_lower_equal);
        loader_compare_scalar_method!(int_tensor_lower_equal_elem, int_tensor_lower_equal_elem);

        loader_binary_method!(int_tensor_bitwise_and, int_tensor_bitwise_and);
        loader_scalar_method!(int_tensor_bitwise_and_scalar, int_tensor_bitwise_and_scalar);
        loader_binary_method!(int_tensor_bitwise_or, int_tensor_bitwise_or);
        loader_scalar_method!(int_tensor_bitwise_or_scalar, int_tensor_bitwise_or_scalar);
        loader_binary_method!(int_tensor_bitwise_xor, int_tensor_bitwise_xor);
        loader_scalar_method!(int_tensor_bitwise_xor_scalar, int_tensor_bitwise_xor_scalar);
        loader_unary_method!(int_tensor_bitwise_not, int_tensor_bitwise_not);
        loader_binary_method!(int_tensor_bitwise_left_shift, int_tensor_bitwise_left_shift);
        loader_scalar_method!(
            int_tensor_bitwise_left_shift_scalar,
            int_tensor_bitwise_left_shift_scalar
        );
        loader_binary_method!(
            int_tensor_bitwise_right_shift,
            int_tensor_bitwise_right_shift
        );
        loader_scalar_method!(
            int_tensor_bitwise_right_shift_scalar,
            int_tensor_bitwise_right_shift_scalar
        );

        /// Expands an int tensor to a broadcast-compatible shape.
        pub fn int_tensor_expand(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_expand)(tensor, shape_ref, out)
            })
        }

        /// Unfolds an int tensor along one dimension.
        pub fn int_tensor_unfold(
            &self,
            tensor: TensorHandle,
            dim: usize,
            size: usize,
            step: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_unfold)(tensor, dim, size, step, out)
            })
        }

        loader_repeat_dim_method!(int_tensor_repeat_dim, int_tensor_repeat_dim);
        loader_cat_method!(int_tensor_cat, int_tensor_cat);
        loader_compare_binary_method!(int_tensor_not_equal, int_tensor_not_equal);
        loader_compare_scalar_method!(int_tensor_not_equal_elem, int_tensor_not_equal_elem);
        loader_binary_method!(int_tensor_powi, int_tensor_powi);
        loader_scalar_method!(int_tensor_powi_scalar, int_tensor_powi_scalar);
        loader_scalar_method!(int_tensor_clamp_min, int_tensor_clamp_min);
        loader_scalar_method!(int_tensor_clamp_max, int_tensor_clamp_max);
        loader_clamp_method!(int_tensor_clamp, int_tensor_clamp);
        loader_unary_method!(int_tensor_neg, int_tensor_neg);

        /// Creates an int tensor filled with zeros.
        pub fn int_tensor_zeros(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_zeros)(device, shape_ref, dtype, out)
            })
        }

        /// Creates an int tensor filled with ones.
        pub fn int_tensor_ones(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_ones)(device, shape_ref, dtype, out)
            })
        }

        /// Creates an int tensor filled with a scalar value.
        pub fn int_tensor_full(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            value: Scalar,
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let value = scalar_to_abi(value);
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_full)(device, shape_ref, value, dtype, out)
            })
        }

        loader_unary_method!(int_tensor_mean, int_tensor_mean);
        loader_unary_method!(int_tensor_max, int_tensor_max);
        loader_dim_method!(int_tensor_max_dim, int_tensor_max_dim);
        loader_with_indices_no_dtype_method!(
            int_tensor_max_dim_with_indices,
            int_tensor_max_dim_with_indices
        );
        loader_unary_method!(int_tensor_max_abs, int_tensor_max_abs);
        loader_dim_method!(int_tensor_max_abs_dim, int_tensor_max_abs_dim);
        loader_unary_method!(int_tensor_min, int_tensor_min);
        loader_dim_method!(int_tensor_min_dim, int_tensor_min_dim);
        loader_with_indices_no_dtype_method!(
            int_tensor_min_dim_with_indices,
            int_tensor_min_dim_with_indices
        );
        loader_unary_method!(int_tensor_transpose, int_tensor_transpose);

        /// Creates an int range tensor with a custom step.
        pub fn int_tensor_arange_step(
            &self,
            start: i64,
            end: i64,
            step: usize,
            device: DeviceHandle,
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_arange_step)(start, end, step, device, dtype, out)
            })
        }

        /// Creates an int range tensor with step 1.
        pub fn int_tensor_arange(
            &self,
            start: i64,
            end: i64,
            device: DeviceHandle,
            dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let dtype = int_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().int_tensor_arange)(start, end, device, dtype, out)
            })
        }

        loader_bool_reduce_method!(int_tensor_any, int_tensor_any);
        loader_bool_reduce_dim_method!(int_tensor_any_dim, int_tensor_any_dim);
        loader_bool_reduce_method!(int_tensor_all, int_tensor_all);
        loader_bool_reduce_dim_method!(int_tensor_all_dim, int_tensor_all_dim);
        loader_unary_method!(int_tensor_sign, int_tensor_sign);
        loader_sort_method!(int_tensor_sort, int_tensor_sort);
        loader_sort_with_indices_no_dtype_method!(
            int_tensor_sort_with_indices,
            int_tensor_sort_with_indices
        );
        loader_argsort_no_dtype_method!(int_tensor_argsort, int_tensor_argsort);

        /// Casts a bool tensor into an int tensor.
        pub fn bool_tensor_into_int(
            &self,
            tensor: TensorHandle,
            out_dtype: IntDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = int_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_into_int)(tensor, out_dtype, out)
            })
        }

        /// Casts a bool tensor into a float tensor.
        pub fn bool_tensor_into_float(
            &self,
            tensor: TensorHandle,
            out_dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = float_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_into_float)(tensor, out_dtype, out)
            })
        }

        /// Moves a bool tensor to a different backend device.
        pub fn bool_tensor_to_device(
            &self,
            tensor: TensorHandle,
            device: DeviceHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_to_device)(tensor, device, out)
            })
        }

        /// Creates an empty bool tensor.
        pub fn bool_tensor_empty(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: BoolDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = bool_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_empty)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a bool tensor filled with zeros.
        pub fn bool_tensor_zeros(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: BoolDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = bool_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_zeros)(device, shape_ref, dtype, out)
            })
        }

        /// Creates a bool tensor filled with ones.
        pub fn bool_tensor_ones(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            dtype: BoolDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            let dtype = bool_dtype_to_abi(dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_ones)(device, shape_ref, dtype, out)
            })
        }

        /// Reshapes a bool tensor.
        pub fn bool_tensor_reshape(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_reshape)(tensor, shape_ref, out)
            })
        }

        /// Gathers values from a bool tensor using index tensor.
        pub fn bool_tensor_gather(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_gather)(dim, tensor, indices, out)
            })
        }

        /// Scatters bool values into a tensor using OR reduction.
        pub fn bool_tensor_scatter_or(
            &self,
            dim: usize,
            tensor: TensorHandle,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_scatter_or)(dim, tensor, indices, value, out)
            })
        }

        /// Selects values from a bool tensor using rank-1 indices.
        pub fn bool_tensor_select(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_select)(tensor, dim, indices, out)
            })
        }

        /// Writes selected bool values into a tensor using OR reduction.
        pub fn bool_tensor_select_or(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_select_or)(tensor, dim, indices, value, out)
            })
        }

        /// Slices a bool tensor.
        pub fn bool_tensor_slice(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_slice)(tensor, slices_ref, out)
            })
        }

        /// Assigns a bool tensor into a slice view.
        pub fn bool_tensor_slice_assign(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_slice_assign)(tensor, slices_ref, value, out)
            })
        }

        /// Selects values from bool tensor where `mask` is true.
        pub fn bool_tensor_mask_where(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_mask_where)(tensor, mask, value, out)
            })
        }

        /// Fills values in bool tensor where `mask` is true.
        pub fn bool_tensor_mask_fill(
            &self,
            tensor: TensorHandle,
            mask: TensorHandle,
            value: Scalar,
        ) -> Result<TensorHandle, PluginCallError> {
            let value = scalar_to_abi(value);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_mask_fill)(tensor, mask, value, out)
            })
        }

        loader_binary_method!(bool_tensor_equal, bool_tensor_equal);
        loader_scalar_method!(bool_tensor_equal_elem, bool_tensor_equal_elem);
        loader_unary_method!(bool_tensor_not, bool_tensor_not);
        loader_binary_method!(bool_tensor_and, bool_tensor_and);
        loader_binary_method!(bool_tensor_or, bool_tensor_or);

        /// Swaps two dimensions on a bool tensor.
        pub fn bool_tensor_swap_dims(
            &self,
            tensor: TensorHandle,
            dim1: usize,
            dim2: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_swap_dims)(tensor, dim1, dim2, out)
            })
        }

        /// Permutes bool tensor dimensions using `axes`.
        pub fn bool_tensor_permute(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_permute)(tensor, axes_ref, out)
            })
        }

        /// Flips bool tensor dimensions listed in `axes`.
        pub fn bool_tensor_flip(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_flip)(tensor, axes_ref, out)
            })
        }

        /// Expands a bool tensor to a broadcast-compatible shape.
        pub fn bool_tensor_expand(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_expand)(tensor, shape_ref, out)
            })
        }

        /// Unfolds a bool tensor along one dimension.
        pub fn bool_tensor_unfold(
            &self,
            tensor: TensorHandle,
            dim: usize,
            size: usize,
            step: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().bool_tensor_unfold)(tensor, dim, size, step, out)
            })
        }

        loader_repeat_dim_method!(bool_tensor_repeat_dim, bool_tensor_repeat_dim);
        loader_cat_method!(bool_tensor_cat, bool_tensor_cat);
        loader_binary_method!(bool_tensor_not_equal, bool_tensor_not_equal);
        loader_scalar_method!(bool_tensor_not_equal_elem, bool_tensor_not_equal_elem);
        loader_binary_method!(bool_tensor_xor, bool_tensor_xor);
        loader_unary_method!(bool_tensor_transpose, bool_tensor_transpose);
        loader_unary_method!(bool_tensor_any, bool_tensor_any);
        loader_bool_dim_method!(bool_tensor_any_dim, bool_tensor_any_dim);
        loader_unary_method!(bool_tensor_all, bool_tensor_all);
        loader_bool_dim_method!(bool_tensor_all_dim, bool_tensor_all_dim);

        /// Creates a quantized tensor from host bytes, shape and quantization scheme.
        pub fn q_tensor_from_u8_data(
            &self,
            device: DeviceHandle,
            shape: &[usize],
            data: &[u8],
            scheme: QuantScheme,
        ) -> Result<TensorHandle, PluginCallError> {
            let mut handle = TensorHandle::INVALID;
            let shape_ref = shape_ref(shape);
            let data_ref = U8SliceRef {
                ptr: data.as_ptr(),
                len: data.len(),
            };
            let scheme = quant_scheme_to_abi(scheme);
            let status = unsafe {
                (self.tensor_ops().q_tensor_from_u8_data)(
                    device,
                    shape_ref,
                    data_ref,
                    scheme,
                    &mut handle,
                )
            };
            check_status(status)?;
            if !handle.is_valid() {
                return Err(PluginCallError::InvalidHandle("tensor"));
            }
            Ok(handle)
        }

        /// Reads a quantized tensor as host bytes plus quantization scheme.
        pub fn q_tensor_into_u8_data(
            &self,
            tensor: TensorHandle,
        ) -> Result<(Vec<u8>, QuantScheme), PluginCallError> {
            let mut scheme = AbiQuantScheme {
                value: AbiQuantValue::Q8F,
                param: AbiQuantParam::F32,
                store: AbiQuantStore::PackedU32,
                store_packed_dim: 0,
                level: AbiQuantLevel::Tensor,
                block_dims: [1; ABI_QUANT_BLOCK_MAX_DIMS],
                block_rank: 0,
                mode: AbiQuantMode::Symmetric,
            };
            let mut buffer = OwnedU8Buffer::empty();
            let status = unsafe {
                (self.tensor_ops().q_tensor_into_u8_data)(tensor, &mut scheme, &mut buffer)
            };
            check_status(status)?;

            let quant_scheme = quant_scheme_from_abi(scheme)?;

            if buffer.len == 0 {
                return Ok((Vec::new(), quant_scheme));
            }
            if buffer.ptr.is_null() {
                return Err(PluginCallError::NullPointer("q_tensor_into_u8_data"));
            }

            let values = unsafe { std::slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec();
            self.release_u8_buffer(buffer)?;
            Ok((values, quant_scheme))
        }

        /// Quantizes a float tensor using the given quantization scheme and scales tensor.
        pub fn q_tensor_quantize(
            &self,
            tensor: TensorHandle,
            scheme: QuantScheme,
            scales: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            let scheme = quant_scheme_to_abi(scheme);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_quantize)(tensor, scheme, scales, out)
            })
        }

        /// Dequantizes a quantized tensor into a float tensor.
        pub fn q_tensor_dequantize(
            &self,
            tensor: TensorHandle,
            out_dtype: FloatDType,
        ) -> Result<TensorHandle, PluginCallError> {
            let out_dtype = float_dtype_to_abi(out_dtype);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_dequantize)(tensor, out_dtype, out)
            })
        }

        /// Moves a quantized tensor to a different backend device.
        pub fn q_tensor_to_device(
            &self,
            tensor: TensorHandle,
            device: DeviceHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_to_device)(tensor, device, out)
            })
        }

        /// Reshapes a quantized tensor.
        pub fn q_tensor_reshape(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_reshape)(tensor, shape_ref, out)
            })
        }

        /// Expands a quantized tensor to a broadcast-compatible shape.
        pub fn q_tensor_expand(
            &self,
            tensor: TensorHandle,
            shape: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let shape_ref = shape_ref(shape);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_expand)(tensor, shape_ref, out)
            })
        }

        /// Swaps two dimensions on a quantized tensor.
        pub fn q_tensor_swap_dims(
            &self,
            tensor: TensorHandle,
            dim1: usize,
            dim2: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_swap_dims)(tensor, dim1, dim2, out)
            })
        }

        /// Permutes quantized tensor dimensions using `axes`.
        pub fn q_tensor_permute(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_permute)(tensor, axes_ref, out)
            })
        }

        /// Flips quantized tensor dimensions listed in `axes`.
        pub fn q_tensor_flip(
            &self,
            tensor: TensorHandle,
            axes: &[usize],
        ) -> Result<TensorHandle, PluginCallError> {
            let axes_ref = shape_ref(axes);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_flip)(tensor, axes_ref, out)
            })
        }

        /// Selects values from a quantized tensor using rank-1 indices.
        pub fn q_tensor_select(
            &self,
            tensor: TensorHandle,
            dim: usize,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_select)(tensor, dim, indices, out)
            })
        }

        /// Slices a quantized tensor.
        pub fn q_tensor_slice(
            &self,
            tensor: TensorHandle,
            slices: &[Slice],
        ) -> Result<TensorHandle, PluginCallError> {
            let slice_items = encode_slices(slices);
            let slices_ref = AbiSliceRef {
                ptr: slice_items.as_ptr(),
                len: slice_items.len(),
            };
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().q_tensor_slice)(tensor, slices_ref, out)
            })
        }

        /// Dispatches `embedding` module operation.
        pub fn module_embedding(
            &self,
            weights: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_embedding)(weights, indices, out)
            })
        }

        /// Dispatches `embedding_backward` module operation.
        pub fn module_embedding_backward(
            &self,
            weights: TensorHandle,
            output_grad: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_embedding_backward)(weights, output_grad, indices, out)
            })
        }

        /// Dispatches `conv1d` module operation.
        pub fn module_conv1d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv1d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `conv1d_x_backward` module operation.
        pub fn module_conv1d_x_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv1d_x_backward)(x, weight, output_grad, options, out)
            })
        }

        /// Dispatches `conv1d_weight_backward` module operation.
        pub fn module_conv1d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv1d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv1d_bias_backward` module operation.
        pub fn module_conv1d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv1d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `conv2d_x_backward` module operation.
        pub fn module_conv2d_x_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv2d_x_backward)(x, weight, output_grad, options, out)
            })
        }

        /// Dispatches `conv2d_weight_backward` module operation.
        pub fn module_conv2d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv2d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv2d_bias_backward` module operation.
        pub fn module_conv2d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv2d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `conv3d_x_backward` module operation.
        pub fn module_conv3d_x_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv3d_x_backward)(x, weight, output_grad, options, out)
            })
        }

        /// Dispatches `conv3d_weight_backward` module operation.
        pub fn module_conv3d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv3d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv3d_bias_backward` module operation.
        pub fn module_conv3d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv3d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `conv_transpose1d` module operation.
        pub fn module_conv_transpose1d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvTransposeOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_transpose_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose1d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `conv_transpose1d_x_backward` module operation.
        pub fn module_conv_transpose1d_x_backward(
            &self,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose1d_x_backward)(
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose1d_weight_backward` module operation.
        pub fn module_conv_transpose1d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<1>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_1_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose1d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose1d_bias_backward` module operation.
        pub fn module_conv_transpose1d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose1d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `conv_transpose2d_x_backward` module operation.
        pub fn module_conv_transpose2d_x_backward(
            &self,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose2d_x_backward)(
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose2d_weight_backward` module operation.
        pub fn module_conv_transpose2d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose2d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose2d_bias_backward` module operation.
        pub fn module_conv_transpose2d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose2d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `conv_transpose3d_x_backward` module operation.
        pub fn module_conv_transpose3d_x_backward(
            &self,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose3d_x_backward)(
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose3d_weight_backward` module operation.
        pub fn module_conv_transpose3d_weight_backward(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            output_grad: TensorHandle,
            options: ConvTransposeOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = conv_transpose_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose3d_weight_backward)(
                    x,
                    weight,
                    output_grad,
                    options,
                    out,
                )
            })
        }

        /// Dispatches `conv_transpose3d_bias_backward` module operation.
        pub fn module_conv_transpose3d_bias_backward(
            &self,
            x: TensorHandle,
            bias: TensorHandle,
            output_grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose3d_bias_backward)(x, bias, output_grad, out)
            })
        }

        /// Dispatches `unfold4d` module operation.
        pub fn module_unfold4d(
            &self,
            x: TensorHandle,
            kernel_size: [usize; 2],
            options: UnfoldOptions,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = unfold_options_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_unfold4d)(x, kernel_size, options, out)
            })
        }

        /// Dispatches `avg_pool1d` module operation.
        pub fn module_avg_pool1d(
            &self,
            x: TensorHandle,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_avg_pool1d)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    u8::from(count_include_pad),
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `avg_pool1d_backward` module operation.
        pub fn module_avg_pool1d_backward(
            &self,
            x: TensorHandle,
            grad: TensorHandle,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_avg_pool1d_backward)(
                    x,
                    grad,
                    kernel_size,
                    stride,
                    padding,
                    u8::from(count_include_pad),
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `adaptive_avg_pool1d` module operation.
        pub fn module_adaptive_avg_pool1d(
            &self,
            x: TensorHandle,
            output_size: usize,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_adaptive_avg_pool1d)(x, output_size, out)
            })
        }

        /// Dispatches `adaptive_avg_pool1d_backward` module operation.
        pub fn module_adaptive_avg_pool1d_backward(
            &self,
            x: TensorHandle,
            grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_adaptive_avg_pool1d_backward)(x, grad, out)
            })
        }

        /// Dispatches `max_pool1d` module operation.
        pub fn module_max_pool1d(
            &self,
            x: TensorHandle,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_max_pool1d)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `max_pool1d_with_indices` module operation.
        pub fn module_max_pool1d_with_indices(
            &self,
            x: TensorHandle,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            ceil_mode: bool,
        ) -> Result<MaxPool1dWithIndicesHandles, PluginCallError> {
            let mut out = AbiMaxPool1dWithIndices {
                output: TensorHandle::INVALID,
                indices: TensorHandle::INVALID,
            };
            let status = unsafe {
                (self.tensor_ops().module_max_pool1d_with_indices)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    &mut out,
                )
            };
            check_status(status)?;

            if !out.output.is_valid() || !out.indices.is_valid() {
                return Err(PluginCallError::InvalidHandle("max_pool1d_with_indices"));
            }

            Ok(MaxPool1dWithIndicesHandles {
                output: out.output,
                indices: out.indices,
            })
        }

        /// Dispatches `max_pool1d_with_indices_backward` module operation.
        pub fn module_max_pool1d_with_indices_backward(
            &self,
            x: TensorHandle,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            ceil_mode: bool,
            output_grad: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_max_pool1d_with_indices_backward)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    output_grad,
                    indices,
                    out,
                )
            })
        }

        /// Dispatches `conv2d` module operation.
        pub fn module_conv2d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv2d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `deform_conv2d` module operation.
        pub fn module_deform_conv2d(
            &self,
            x: TensorHandle,
            offset: TensorHandle,
            weight: TensorHandle,
            mask: Option<TensorHandle>,
            bias: Option<TensorHandle>,
            options: DeformConvOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let mask = mask.unwrap_or(TensorHandle::INVALID);
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = deform_conv_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_deform_conv2d)(
                    x, offset, weight, mask, bias, options, out,
                )
            })
        }

        /// Dispatches `deform_conv2d_backward` module operation.
        pub fn module_deform_conv2d_backward(
            &self,
            x: TensorHandle,
            offset: TensorHandle,
            weight: TensorHandle,
            mask: Option<TensorHandle>,
            bias: Option<TensorHandle>,
            output_grad: TensorHandle,
            options: DeformConvOptions<2>,
        ) -> Result<DeformConv2dBackwardHandles, PluginCallError> {
            let mask = mask.unwrap_or(TensorHandle::INVALID);
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = deform_conv_options_2_to_abi(options);

            let mut out = AbiDeformConv2dBackward {
                x_grad: TensorHandle::INVALID,
                offset_grad: TensorHandle::INVALID,
                weight_grad: TensorHandle::INVALID,
                mask_grad: TensorHandle::INVALID,
                bias_grad: TensorHandle::INVALID,
                has_mask_grad: 0,
                has_bias_grad: 0,
            };
            let status = unsafe {
                (self.tensor_ops().module_deform_conv2d_backward)(
                    x,
                    offset,
                    weight,
                    mask,
                    bias,
                    output_grad,
                    options,
                    &mut out,
                )
            };
            check_status(status)?;

            if !out.x_grad.is_valid() || !out.offset_grad.is_valid() || !out.weight_grad.is_valid()
            {
                return Err(PluginCallError::InvalidHandle("deform_conv2d_backward"));
            }

            let mask_grad = if out.has_mask_grad == 0 {
                None
            } else if out.mask_grad.is_valid() {
                Some(out.mask_grad)
            } else {
                return Err(PluginCallError::InvalidHandle(
                    "deform_conv2d_backward.mask_grad",
                ));
            };

            let bias_grad = if out.has_bias_grad == 0 {
                None
            } else if out.bias_grad.is_valid() {
                Some(out.bias_grad)
            } else {
                return Err(PluginCallError::InvalidHandle(
                    "deform_conv2d_backward.bias_grad",
                ));
            };

            Ok(DeformConv2dBackwardHandles {
                x_grad: out.x_grad,
                offset_grad: out.offset_grad,
                weight_grad: out.weight_grad,
                mask_grad,
                bias_grad,
            })
        }

        /// Dispatches `conv3d` module operation.
        pub fn module_conv3d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv3d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `conv_transpose2d` module operation.
        pub fn module_conv_transpose2d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvTransposeOptions<2>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_transpose_options_2_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose2d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `conv_transpose3d` module operation.
        pub fn module_conv_transpose3d(
            &self,
            x: TensorHandle,
            weight: TensorHandle,
            bias: Option<TensorHandle>,
            options: ConvTransposeOptions<3>,
        ) -> Result<TensorHandle, PluginCallError> {
            let bias = bias.unwrap_or(TensorHandle::INVALID);
            let options = conv_transpose_options_3_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_conv_transpose3d)(x, weight, bias, options, out)
            })
        }

        /// Dispatches `avg_pool2d` module operation.
        pub fn module_avg_pool2d(
            &self,
            x: TensorHandle,
            kernel_size: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_avg_pool2d)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    u8::from(count_include_pad),
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `avg_pool2d_backward` module operation.
        pub fn module_avg_pool2d_backward(
            &self,
            x: TensorHandle,
            grad: TensorHandle,
            kernel_size: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_avg_pool2d_backward)(
                    x,
                    grad,
                    kernel_size,
                    stride,
                    padding,
                    u8::from(count_include_pad),
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `adaptive_avg_pool2d` module operation.
        pub fn module_adaptive_avg_pool2d(
            &self,
            x: TensorHandle,
            output_size: [usize; 2],
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_adaptive_avg_pool2d)(x, output_size, out)
            })
        }

        /// Dispatches `adaptive_avg_pool2d_backward` module operation.
        pub fn module_adaptive_avg_pool2d_backward(
            &self,
            x: TensorHandle,
            grad: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_adaptive_avg_pool2d_backward)(x, grad, out)
            })
        }

        /// Dispatches `max_pool2d` module operation.
        pub fn module_max_pool2d(
            &self,
            x: TensorHandle,
            kernel_size: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            dilation: [usize; 2],
            ceil_mode: bool,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_max_pool2d)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    out,
                )
            })
        }

        /// Dispatches `max_pool2d_with_indices` module operation.
        pub fn module_max_pool2d_with_indices(
            &self,
            x: TensorHandle,
            kernel_size: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            dilation: [usize; 2],
            ceil_mode: bool,
        ) -> Result<MaxPool2dWithIndicesHandles, PluginCallError> {
            let mut out = AbiMaxPool2dWithIndices {
                output: TensorHandle::INVALID,
                indices: TensorHandle::INVALID,
            };
            let status = unsafe {
                (self.tensor_ops().module_max_pool2d_with_indices)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    &mut out,
                )
            };
            check_status(status)?;

            if !out.output.is_valid() || !out.indices.is_valid() {
                return Err(PluginCallError::InvalidHandle("max_pool2d_with_indices"));
            }

            Ok(MaxPool2dWithIndicesHandles {
                output: out.output,
                indices: out.indices,
            })
        }

        /// Dispatches `max_pool2d_with_indices_backward` module operation.
        pub fn module_max_pool2d_with_indices_backward(
            &self,
            x: TensorHandle,
            kernel_size: [usize; 2],
            stride: [usize; 2],
            padding: [usize; 2],
            dilation: [usize; 2],
            ceil_mode: bool,
            output_grad: TensorHandle,
            indices: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_max_pool2d_with_indices_backward)(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    u8::from(ceil_mode),
                    output_grad,
                    indices,
                    out,
                )
            })
        }

        /// Dispatches `interpolate` module operation.
        pub fn module_interpolate(
            &self,
            x: TensorHandle,
            output_size: [usize; 2],
            options: InterpolateOptions,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = interpolate_options_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_interpolate)(x, output_size, options, out)
            })
        }

        /// Dispatches `interpolate_backward` module operation.
        pub fn module_interpolate_backward(
            &self,
            x: TensorHandle,
            grad: TensorHandle,
            output_size: [usize; 2],
            options: InterpolateOptions,
        ) -> Result<TensorHandle, PluginCallError> {
            let options = interpolate_options_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_interpolate_backward)(x, grad, output_size, options, out)
            })
        }

        /// Dispatches `attention` module operation.
        pub fn module_attention(
            &self,
            query: TensorHandle,
            key: TensorHandle,
            value: TensorHandle,
            mask: Option<TensorHandle>,
            attn_bias: Option<TensorHandle>,
            options: AttentionModuleOptions,
        ) -> Result<TensorHandle, PluginCallError> {
            let mask = mask.unwrap_or(TensorHandle::INVALID);
            let attn_bias = attn_bias.unwrap_or(TensorHandle::INVALID);
            let options = attention_options_to_abi(options);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().module_attention)(
                    query, key, value, mask, attn_bias, options, out,
                )
            })
        }

        /// Dispatches `rfft` module operation.
        pub fn module_rfft(
            &self,
            signal: TensorHandle,
            dim: usize,
        ) -> Result<RfftHandles, PluginCallError> {
            let mut out = AbiRfftOutput {
                real: TensorHandle::INVALID,
                imag: TensorHandle::INVALID,
            };
            let status = unsafe { (self.tensor_ops().module_rfft)(signal, dim, &mut out) };
            check_status(status)?;

            if !out.real.is_valid() || !out.imag.is_valid() {
                return Err(PluginCallError::InvalidHandle("rfft"));
            }

            Ok(RfftHandles {
                real: out.real,
                imag: out.imag,
            })
        }

        /// Dispatches `leaky_relu` activation operation.
        pub fn activation_leaky_relu(
            &self,
            tensor: TensorHandle,
            negative_slope: Scalar,
        ) -> Result<TensorHandle, PluginCallError> {
            let negative_slope = scalar_to_abi(negative_slope);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().activation_leaky_relu)(tensor, negative_slope, out)
            })
        }

        loader_unary_method!(activation_relu, activation_relu);
        loader_binary_method!(activation_relu_backward, activation_relu_backward);
        loader_unary_method!(activation_gelu, activation_gelu);
        loader_binary_method!(activation_prelu, activation_prelu);
        loader_binary_method!(activation_gelu_backward, activation_gelu_backward);
        loader_unary_method!(activation_sigmoid, activation_sigmoid);
        loader_binary_method!(activation_sigmoid_backward, activation_sigmoid_backward);

        /// Dispatches `hard_sigmoid` activation operation.
        pub fn activation_hard_sigmoid(
            &self,
            tensor: TensorHandle,
            alpha: Scalar,
            beta: Scalar,
        ) -> Result<TensorHandle, PluginCallError> {
            let alpha = scalar_to_abi(alpha);
            let beta = scalar_to_abi(beta);
            self.call_with_out_handle("tensor", |out| unsafe {
                (self.tensor_ops().activation_hard_sigmoid)(tensor, alpha, beta, out)
            })
        }

        loader_unary_method!(activation_log_sigmoid, activation_log_sigmoid);
        loader_binary_method!(
            activation_log_sigmoid_backward,
            activation_log_sigmoid_backward
        );

        /// Releases a tensor handle.
        pub fn release_tensor(&self, tensor: TensorHandle) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_tensor)(tensor) };
            check_status(status)
        }

        /// Executes a read transaction, returning raw data vectors for each tensor type.
        ///
        /// Returns `(floats, qfloats, ints, bools)` where:
        /// - `floats`: one `Vec<f32>` per float tensor
        /// - `qfloats`: one `(Vec<u8>, QuantScheme)` per quantized tensor
        /// - `ints`: one `Vec<u64>` per int tensor
        /// - `bools`: one `Vec<u8>` per bool tensor
        pub fn transaction_execute(
            &self,
            floats: &[TensorHandle],
            qfloats: &[TensorHandle],
            ints: &[TensorHandle],
            bools: &[TensorHandle],
        ) -> Result<
            (
                Vec<Vec<f32>>,
                Vec<(Vec<u8>, QuantScheme)>,
                Vec<Vec<u64>>,
                Vec<Vec<u8>>,
            ),
            PluginCallError,
        > {
            let mut out_floats: Vec<OwnedF32Buffer> =
                vec![OwnedF32Buffer::empty(); floats.len()];
            let mut out_qfloats: Vec<OwnedQTransactionItem> = (0..qfloats.len())
                .map(|_| OwnedQTransactionItem {
                    scheme: AbiQuantScheme {
                        value: AbiQuantValue::Q8F,
                        param: AbiQuantParam::F32,
                        store: AbiQuantStore::PackedU32,
                        store_packed_dim: 0,
                        level: AbiQuantLevel::Tensor,
                        block_dims: [1; ABI_QUANT_BLOCK_MAX_DIMS],
                        block_rank: 0,
                        mode: AbiQuantMode::Symmetric,
                    },
                    data: OwnedU8Buffer::empty(),
                })
                .collect();
            let mut out_ints: Vec<OwnedU64Buffer> =
                vec![OwnedU64Buffer::empty(); ints.len()];
            let mut out_bools: Vec<OwnedU8Buffer> =
                vec![OwnedU8Buffer::empty(); bools.len()];

            let floats_ref = TensorHandleRef {
                ptr: floats.as_ptr(),
                len: floats.len(),
            };
            let qfloats_ref = TensorHandleRef {
                ptr: qfloats.as_ptr(),
                len: qfloats.len(),
            };
            let ints_ref = TensorHandleRef {
                ptr: ints.as_ptr(),
                len: ints.len(),
            };
            let bools_ref = TensorHandleRef {
                ptr: bools.as_ptr(),
                len: bools.len(),
            };

            let status = unsafe {
                (self.tensor_ops().transaction_execute)(
                    floats_ref,
                    qfloats_ref,
                    ints_ref,
                    bools_ref,
                    if floats.is_empty() {
                        std::ptr::null_mut()
                    } else {
                        out_floats.as_mut_ptr()
                    },
                    if qfloats.is_empty() {
                        std::ptr::null_mut()
                    } else {
                        out_qfloats.as_mut_ptr()
                    },
                    if ints.is_empty() {
                        std::ptr::null_mut()
                    } else {
                        out_ints.as_mut_ptr()
                    },
                    if bools.is_empty() {
                        std::ptr::null_mut()
                    } else {
                        out_bools.as_mut_ptr()
                    },
                )
            };
            check_status(status)?;

            let mut float_result = Vec::with_capacity(floats.len());
            for buf in out_floats {
                if buf.len == 0 {
                    float_result.push(Vec::new());
                    continue;
                }
                if buf.ptr.is_null() {
                    return Err(PluginCallError::NullPointer("transaction_execute/float"));
                }
                let values =
                    unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) }.to_vec();
                self.release_f32_buffer(buf)?;
                float_result.push(values);
            }

            let mut qfloat_result = Vec::with_capacity(qfloats.len());
            for item in out_qfloats {
                let scheme = quant_scheme_from_abi(item.scheme)?;
                let data = if item.data.len == 0 {
                    Vec::new()
                } else {
                    if item.data.ptr.is_null() {
                        return Err(PluginCallError::NullPointer(
                            "transaction_execute/qfloat",
                        ));
                    }
                    let values = unsafe {
                        std::slice::from_raw_parts(item.data.ptr, item.data.len)
                    }
                    .to_vec();
                    self.release_u8_buffer(item.data)?;
                    values
                };
                qfloat_result.push((data, scheme));
            }

            let mut int_result = Vec::with_capacity(ints.len());
            for buf in out_ints {
                if buf.len == 0 {
                    int_result.push(Vec::new());
                    continue;
                }
                if buf.ptr.is_null() {
                    return Err(PluginCallError::NullPointer("transaction_execute/int"));
                }
                let values =
                    unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) }.to_vec();
                self.release_u64_buffer(buf)?;
                int_result.push(values);
            }

            let mut bool_result = Vec::with_capacity(bools.len());
            for buf in out_bools {
                if buf.len == 0 {
                    bool_result.push(Vec::new());
                    continue;
                }
                if buf.ptr.is_null() {
                    return Err(PluginCallError::NullPointer("transaction_execute/bool"));
                }
                let values =
                    unsafe { std::slice::from_raw_parts(buf.ptr, buf.len) }.to_vec();
                self.release_u8_buffer(buf)?;
                bool_result.push(values);
            }

            Ok((float_result, qfloat_result, int_result, bool_result))
        }

        fn api(&self) -> &BackendPluginV1 {
            // Safety: `plugin` was checked for null at load time and the library stays loaded
            // for as long as this struct exists.
            unsafe { &*self.plugin }
        }

        fn tensor_ops(&self) -> &BackendTensorOpsV1 {
            // Safety: `tensor_ops` was checked for null at load time and the library stays loaded
            // for as long as this struct exists.
            unsafe { &*self.tensor_ops }
        }

        fn call_with_out_handle(
            &self,
            context: &'static str,
            call: impl FnOnce(*mut TensorHandle) -> PluginStatus,
        ) -> Result<TensorHandle, PluginCallError> {
            let mut out = TensorHandle::INVALID;
            let status = call(&mut out);
            check_status(status)?;
            if !out.is_valid() {
                return Err(PluginCallError::InvalidHandle(context));
            }
            Ok(out)
        }

        fn release_f32_buffer(&self, buffer: OwnedF32Buffer) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_f32_buffer)(buffer) };
            check_status(status)
        }

        fn release_u64_buffer(&self, buffer: OwnedU64Buffer) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_u64_buffer)(buffer) };
            check_status(status)
        }

        fn release_u8_buffer(&self, buffer: OwnedU8Buffer) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_u8_buffer)(buffer) };
            check_status(status)
        }

        fn release_usize_buffer(&self, buffer: OwnedUsizeBuffer) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_usize_buffer)(buffer) };
            check_status(status)
        }

        /// Returns true while the plugin library is loaded.
        pub fn is_loaded(&self) -> bool {
            let _ = &self.library;
            true
        }
    }

    fn shape_ref(shape: &[usize]) -> TensorShapeRef {
        TensorShapeRef {
            dims: shape.as_ptr(),
            rank: shape.len(),
        }
    }

    fn float_dtype_to_abi(dtype: FloatDType) -> AbiFloatDType {
        match dtype {
            FloatDType::F64 => AbiFloatDType::F64,
            FloatDType::F32 => AbiFloatDType::F32,
            FloatDType::Flex32 => AbiFloatDType::Flex32,
            FloatDType::F16 => AbiFloatDType::F16,
            FloatDType::BF16 => AbiFloatDType::BF16,
        }
    }

    fn int_dtype_to_abi(dtype: IntDType) -> AbiIntDType {
        match dtype {
            IntDType::I64 => AbiIntDType::I64,
            IntDType::I32 => AbiIntDType::I32,
            IntDType::I16 => AbiIntDType::I16,
            IntDType::I8 => AbiIntDType::I8,
            IntDType::U64 => AbiIntDType::U64,
            IntDType::U32 => AbiIntDType::U32,
            IntDType::U16 => AbiIntDType::U16,
            IntDType::U8 => AbiIntDType::U8,
        }
    }

    fn bool_dtype_to_abi(dtype: BoolDType) -> AbiBoolDType {
        match dtype {
            BoolDType::Native => AbiBoolDType::Native,
            BoolDType::U8 => AbiBoolDType::U8,
            BoolDType::U32 => AbiBoolDType::U32,
        }
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

    fn quant_param_from_abi(param: AbiQuantParam) -> QuantParam {
        match param {
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

    fn quant_scheme_from_abi(scheme: AbiQuantScheme) -> Result<QuantScheme, PluginCallError> {
        let level = match scheme.level {
            AbiQuantLevel::Tensor => QuantLevel::Tensor,
            AbiQuantLevel::Block => {
                if scheme.block_rank == 0 || scheme.block_rank > ABI_QUANT_BLOCK_MAX_DIMS {
                    return Err(PluginCallError::Failure {
                        code: PluginStatusCode::Failed,
                        message: format!(
                            "invalid quantized block rank '{}' in plugin response",
                            scheme.block_rank
                        ),
                    });
                }
                QuantLevel::Block(BlockSize::new(&scheme.block_dims[..scheme.block_rank]))
            }
        };

        Ok(QuantScheme {
            value: quant_value_from_abi(scheme.value),
            param: quant_param_from_abi(scheme.param),
            store: quant_store_from_abi(scheme.store, scheme.store_packed_dim),
            level,
            mode: quant_mode_from_abi(scheme.mode),
        })
    }

    fn scalar_to_abi(scalar: Scalar) -> AbiScalar {
        match scalar {
            Scalar::Float(value) => AbiScalar {
                kind: AbiScalarKind::Float,
                payload: value.to_bits(),
            },
            Scalar::Int(value) => AbiScalar {
                kind: AbiScalarKind::Int,
                payload: value as u64,
            },
            Scalar::UInt(value) => AbiScalar {
                kind: AbiScalarKind::UInt,
                payload: value,
            },
            Scalar::Bool(value) => AbiScalar {
                kind: AbiScalarKind::Bool,
                payload: u64::from(value),
            },
        }
    }

    fn distribution_to_abi(distribution: Distribution) -> AbiDistribution {
        match distribution {
            Distribution::Default => AbiDistribution {
                kind: AbiDistributionKind::Default,
                param0: 0.0,
                param1: 0.0,
            },
            Distribution::Bernoulli(probability) => AbiDistribution {
                kind: AbiDistributionKind::Bernoulli,
                param0: probability,
                param1: 0.0,
            },
            Distribution::Uniform(low, high) => AbiDistribution {
                kind: AbiDistributionKind::Uniform,
                param0: low,
                param1: high,
            },
            Distribution::Normal(mean, std) => AbiDistribution {
                kind: AbiDistributionKind::Normal,
                param0: mean,
                param1: std,
            },
        }
    }

    fn conv_options_2_to_abi(options: ConvOptions<2>) -> AbiConvOptions2 {
        AbiConvOptions2 {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn conv_options_1_to_abi(options: ConvOptions<1>) -> AbiConvOptions1 {
        AbiConvOptions1 {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn conv_options_3_to_abi(options: ConvOptions<3>) -> AbiConvOptions3 {
        AbiConvOptions3 {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn deform_conv_options_2_to_abi(options: DeformConvOptions<2>) -> AbiDeformConvOptions2 {
        AbiDeformConvOptions2 {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            weight_groups: options.weight_groups,
            offset_groups: options.offset_groups,
        }
    }

    fn conv_transpose_options_2_to_abi(
        options: ConvTransposeOptions<2>,
    ) -> AbiConvTransposeOptions2 {
        AbiConvTransposeOptions2 {
            stride: options.stride,
            padding: options.padding,
            padding_out: options.padding_out,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn conv_transpose_options_1_to_abi(
        options: ConvTransposeOptions<1>,
    ) -> AbiConvTransposeOptions1 {
        AbiConvTransposeOptions1 {
            stride: options.stride,
            padding: options.padding,
            padding_out: options.padding_out,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn conv_transpose_options_3_to_abi(
        options: ConvTransposeOptions<3>,
    ) -> AbiConvTransposeOptions3 {
        AbiConvTransposeOptions3 {
            stride: options.stride,
            padding: options.padding,
            padding_out: options.padding_out,
            dilation: options.dilation,
            groups: options.groups,
        }
    }

    fn interpolate_mode_to_abi(mode: InterpolateMode) -> AbiInterpolateMode {
        match mode {
            InterpolateMode::Nearest => AbiInterpolateMode::Nearest,
            InterpolateMode::Bilinear => AbiInterpolateMode::Bilinear,
            InterpolateMode::Bicubic => AbiInterpolateMode::Bicubic,
            InterpolateMode::Lanczos3 => AbiInterpolateMode::Lanczos3,
        }
    }

    fn unfold_options_to_abi(options: UnfoldOptions) -> AbiUnfoldOptions {
        AbiUnfoldOptions {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        }
    }

    fn interpolate_options_to_abi(options: InterpolateOptions) -> AbiInterpolateOptions {
        AbiInterpolateOptions {
            mode: interpolate_mode_to_abi(options.mode),
            align_corners: u8::from(options.align_corners),
        }
    }

    fn attention_options_to_abi(options: AttentionModuleOptions) -> AbiAttentionModuleOptions {
        AbiAttentionModuleOptions {
            scale: options.scale.unwrap_or(0.0),
            has_scale: u8::from(options.scale.is_some()),
            softcap: options.softcap.unwrap_or(0.0),
            has_softcap: u8::from(options.softcap.is_some()),
            is_causal: u8::from(options.is_causal),
        }
    }

    fn encode_slices(slices: &[Slice]) -> Vec<AbiSlice> {
        slices
            .iter()
            .map(|slice| AbiSlice {
                start: slice.start,
                end: slice.end.unwrap_or(0),
                step: slice.step,
                has_end: u8::from(slice.end.is_some()),
            })
            .collect()
    }

    fn check_status(status: PluginStatus) -> Result<(), PluginCallError> {
        if status.code == PluginStatusCode::Ok {
            return Ok(());
        }

        let message = if status.message.is_null() {
            String::from("<no message>")
        } else {
            read_c_string(status.message, "status.message")?
        };

        Err(PluginCallError::Failure {
            code: status.code,
            message,
        })
    }

    fn read_c_string(ptr: *const c_char, context: &'static str) -> Result<String, PluginCallError> {
        if ptr.is_null() {
            return Err(PluginCallError::NullPointer(context));
        }

        // Safety: pointer validity and null termination are guaranteed by plugin contract.
        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str()
            .map(str::to_owned)
            .map_err(PluginCallError::InvalidUtf8)
    }
}
