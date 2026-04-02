#![cfg(feature = "dylib")]

use burn_backend::DeviceId;
use burn_dylib::DeviceHandle;

use super::runtime::{self, DylibError};

/// Runtime-loaded dispatch device.
#[derive(Clone, PartialEq, Eq)]
pub struct DylibDevice {
    pub(crate) registry_index: u32,
    pub(crate) runtime_id: u64,
    pub(crate) backend_type_id: u16,
    pub(crate) ordinal: u32,
    pub(crate) handle: DeviceHandle,
}

impl core::fmt::Debug for DylibDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DylibDevice")
            .field("registry_index", &self.registry_index)
            .field("runtime_id", &self.runtime_id)
            .field("backend_type_id", &self.backend_type_id)
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl Default for DylibDevice {
    fn default() -> Self {
        panic!(
            "DylibDevice::default() is not available. Use DispatchDevice::dylib(path, type_id, ordinal)."
        )
    }
}

impl burn_backend::Device for DylibDevice {
    fn from_id(device_id: DeviceId) -> Self {
        runtime::device_from_registry(device_id.index_id).unwrap_or_else(|err| panic!("{err}"))
    }

    fn to_id(&self) -> DeviceId {
        DeviceId::new(self.backend_type_id, self.registry_index)
    }
}

impl burn_backend::DeviceOps for DylibDevice {}

pub fn create_device_from_path(
    path: impl AsRef<std::path::Path>,
    backend_type_id: u16,
    ordinal: usize,
) -> Result<DylibDevice, DylibError> {
    runtime::create_device_from_path(path, backend_type_id, ordinal)
}

pub fn device_from_registry(index_id: u32) -> Result<DylibDevice, DylibError> {
    runtime::device_from_registry(index_id)
}
