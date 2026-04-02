#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn dynamic backend plugin ABI.
//!
//! This crate provides two layers:
//! - A versioned C ABI (`BackendPluginV1`) for backend plugins.
//! - A runtime loader (`loader`) to load a backend from a shared library.
//!
//! # Design Goal
//!
//! Compile application code without linking any heavy backend, then load a backend plugin (`.so`,
//! `.dylib`, `.dll`) at runtime.

use core::ffi::c_char;

/// Symbol name that backend plugins must export.
pub const BACKEND_PLUGIN_SYMBOL: &[u8] = b"burn_backend_plugin_v1\0";

/// Current plugin ABI version.
pub const BACKEND_PLUGIN_ABI_VERSION: u32 = 1;

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

/// Runtime dynamic library loader for plugin backends.
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod loader {
    use super::{
        BACKEND_PLUGIN_ABI_VERSION, BACKEND_PLUGIN_SYMBOL, BackendPluginEntrypoint,
        BackendPluginV1, PluginStatus, PluginStatusCode,
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
        /// The entry symbol is missing or has an incompatible type.
        Symbol(libloading::Error),
        /// Entrypoint returned a null plugin pointer.
        NullPlugin,
        /// Plugin ABI version does not match the host ABI version.
        AbiVersionMismatch {
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
                Self::Symbol(err) => write!(f, "Failed to resolve plugin symbol: {err}"),
                Self::NullPlugin => {
                    write!(f, "Plugin entrypoint returned a null backend descriptor")
                }
                Self::AbiVersionMismatch { expected, found } => {
                    write!(
                        f,
                        "Plugin ABI mismatch, expected version {expected} but found {found}",
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
                unsafe { library.get(BACKEND_PLUGIN_SYMBOL) }.map_err(LoadError::Symbol)?;

            let plugin = unsafe { entrypoint() };
            if plugin.is_null() {
                return Err(LoadError::NullPlugin);
            }

            // Safety: checked for null above and kept valid by owning `library` in this struct.
            let api = unsafe { &*plugin };
            if api.abi_version != BACKEND_PLUGIN_ABI_VERSION {
                return Err(LoadError::AbiVersionMismatch {
                    expected: BACKEND_PLUGIN_ABI_VERSION,
                    found: api.abi_version,
                });
            }

            Ok(Self { library, plugin })
        }

        /// Returns the backend name from the plugin.
        pub fn name(&self) -> Result<String, PluginCallError> {
            let ptr = unsafe { (self.api().backend_name)() };
            read_c_string(ptr, "backend_name")
        }

        /// Forwards a seed value to the loaded backend.
        pub fn seed(&self, seed: u64) -> Result<(), PluginCallError> {
            let status = unsafe { (self.api().seed)(seed) };
            check_status(status)
        }

        /// Synchronizes all pending operations on the backend.
        pub fn sync(&self) -> Result<(), PluginCallError> {
            let status = unsafe { (self.api().sync)() };
            check_status(status)
        }

        /// Returns the number of devices for the provided backend type identifier.
        pub fn device_count(&self, type_id: u16) -> usize {
            unsafe { (self.api().device_count)(type_id) }
        }

        fn api(&self) -> &BackendPluginV1 {
            // Safety: `plugin` was checked for null at load time and the library stays loaded
            // for as long as this struct exists.
            unsafe { &*self.plugin }
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
