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
pub const BACKEND_TENSOR_OPS_ABI_VERSION: u32 = 1;

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

/// Tensor addition operation.
pub type TensorAddFn = unsafe extern "C" fn(
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

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
    /// Dispatches tensor addition.
    pub tensor_add: TensorAddFn,
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
        BACKEND_PLUGIN_ABI_VERSION, BACKEND_PLUGIN_SYMBOL, BACKEND_TENSOR_OPS_ABI_VERSION,
        BACKEND_TENSOR_OPS_SYMBOL, BackendPluginEntrypoint, BackendPluginV1,
        BackendTensorOpsEntrypoint, BackendTensorOpsV1, DeviceHandle, F32SliceRef, OwnedF32Buffer,
        OwnedUsizeBuffer, PluginStatus, PluginStatusCode, TensorHandle, TensorShapeRef,
    };
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

        /// Adds two float tensors and returns a newly allocated tensor handle.
        pub fn float_tensor_add(
            &self,
            lhs: TensorHandle,
            rhs: TensorHandle,
        ) -> Result<TensorHandle, PluginCallError> {
            let mut out = TensorHandle::INVALID;
            let status = unsafe { (self.tensor_ops().tensor_add)(lhs, rhs, &mut out) };
            check_status(status)?;
            if !out.is_valid() {
                return Err(PluginCallError::InvalidHandle("tensor"));
            }
            Ok(out)
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
