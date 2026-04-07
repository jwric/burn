use burn_backend::DeviceId;

use super::runtime::{self, DeviceSnapshot, DylibError};

/// Runtime-loaded dispatch device.
///
/// The actual plugin handle and backend metadata live in the runtime registry.
/// This wrapper keeps only the registry index and relies on the runtime to
/// retain and release the underlying plugin device.
#[derive(PartialEq, Eq)]
pub struct DylibDevice {
    pub(crate) registry_index: u32,
}

impl DylibDevice {
    pub(crate) fn from_registry_index(registry_index: u32) -> Self {
        Self { registry_index }
    }

    fn snapshot(&self) -> Result<DeviceSnapshot, DylibError> {
        runtime::device_snapshot(self.registry_index)
    }
}

impl Clone for DylibDevice {
    fn clone(&self) -> Self {
        runtime::retain_device(self.registry_index).unwrap_or_else(|err| panic!("{err}"));
        Self::from_registry_index(self.registry_index)
    }
}

impl Drop for DylibDevice {
    fn drop(&mut self) {
        runtime::release_device(self.registry_index);
    }
}

impl core::fmt::Debug for DylibDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("DylibDevice");
        debug.field("registry_index", &self.registry_index);

        match self.snapshot() {
            Ok(snapshot) => {
                debug
                    .field("runtime_id", &snapshot.runtime_id)
                    .field("backend_type_id", &snapshot.backend_type_id)
                    .field("ordinal", &snapshot.ordinal);
            }
            Err(err) => {
                debug.field("error", &err.to_string());
            }
        }

        debug.finish()
    }
}

impl Default for DylibDevice {
    fn default() -> Self {
        runtime::create_default_device().unwrap_or_else(|err| panic!("{err}"))
    }
}

impl burn_backend::Device for DylibDevice {
    fn from_id(device_id: DeviceId) -> Self {
        runtime::device_from_registry(device_id.index_id).unwrap_or_else(|err| panic!("{err}"))
    }

    fn to_id(&self) -> DeviceId {
        let snapshot = self.snapshot().unwrap_or_else(|err| panic!("{err}"));
        DeviceId::new(snapshot.backend_type_id, self.registry_index)
    }
}

impl burn_backend::DeviceOps for DylibDevice {}
