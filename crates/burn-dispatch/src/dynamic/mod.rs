//! Runtime-loaded backend support for dispatch.

mod backend;
mod device;
mod ops;
mod runtime;
mod tensor;

pub use backend::Dylib;
pub use device::DylibDevice;
pub use runtime::DylibError;

pub(crate) use device::{create_device_from_path, device_from_registry};
