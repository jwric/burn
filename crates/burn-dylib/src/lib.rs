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

pub use runtime::{create_device_from_path, device_from_registry};

/// Symbol name that backend plugins must export.
pub const BACKEND_PLUGIN_SYMBOL: &[u8] = b"burn_backend_plugin_v1\0";

/// Symbol name that backend tensor operation table exports must use.
pub const BACKEND_TENSOR_OPS_SYMBOL: &[u8] = b"burn_backend_tensor_ops_v1\0";

/// Current plugin ABI version.
pub const BACKEND_PLUGIN_ABI_VERSION: u32 = 1;

/// Current tensor operations ABI version.
pub const BACKEND_TENSOR_OPS_ABI_VERSION: u32 = 2;

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

/// Borrowed f32 data slice passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct F32SliceRef {
    /// Pointer to contiguous f32 data.
    pub ptr: *const f32,
    /// Number of f32 elements.
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

/// Materializes a tensor into host f32 data.
pub type TensorIntoF32DataFn =
    unsafe extern "C" fn(tensor: TensorHandle, out_data: *mut OwnedF32Buffer) -> PluginStatus;

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

/// Tensor empty creation operation.
pub type TensorEmptyFn = unsafe extern "C" fn(
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: AbiFloatDType,
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

/// Tensor addition operation.
pub type TensorAddFn = TensorBinaryFn;

/// Releases a tensor handle.
pub type TensorReleaseFn = unsafe extern "C" fn(tensor: TensorHandle) -> PluginStatus;

/// Releases a plugin-allocated f32 buffer.
pub type ReleaseF32BufferFn = unsafe extern "C" fn(buffer: OwnedF32Buffer) -> PluginStatus;

/// Releases a plugin-allocated shape buffer.
pub type ReleaseUsizeBufferFn = unsafe extern "C" fn(buffer: OwnedUsizeBuffer) -> PluginStatus;

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
    /// Releases a tensor handle.
    pub release_tensor: TensorReleaseFn,
    /// Releases a plugin-allocated f32 buffer.
    pub release_f32_buffer: ReleaseF32BufferFn,
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
        AbiBoolDType, AbiDistribution, AbiDistributionKind, AbiFloatDType, AbiIntDType, AbiScalar,
        AbiScalarKind, AbiSlice, AbiSliceRef, BACKEND_PLUGIN_ABI_VERSION, BACKEND_PLUGIN_SYMBOL,
        BACKEND_TENSOR_OPS_ABI_VERSION, BACKEND_TENSOR_OPS_SYMBOL, BackendPluginEntrypoint,
        BackendPluginV1, BackendTensorOpsEntrypoint, BackendTensorOpsV1, DeviceHandle, F32SliceRef,
        OwnedF32Buffer, OwnedUsizeBuffer, PluginStatus, PluginStatusCode, TensorHandle,
        TensorShapeRef,
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

        loader_unary_method!(float_tensor_sum, tensor_sum);
        loader_dim_method!(float_tensor_sum_dim, tensor_sum_dim);
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

        /// Releases a tensor handle.
        pub fn release_tensor(&self, tensor: TensorHandle) -> Result<(), PluginCallError> {
            let status = unsafe { (self.tensor_ops().release_tensor)(tensor) };
            check_status(status)
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
