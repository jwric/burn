use super::{RemoteDevice, service::RemoteService};
use burn_backend::{DeviceHandle, backend::Device};

/// A thin handle to a [`RemoteService`] running on its own device-runner thread.
///
/// Every `RouterClient` method delegates to `handle.submit` / `submit_blocking`, so all
/// connection state, the tokio runtime, the response-demux task, and the op batch buffer
/// live on the service side.
pub struct RemoteClient {
    pub(crate) device: RemoteDevice,
    pub(crate) handle: DeviceHandle<RemoteService>,
}

impl RemoteClient {
    pub fn init(device: RemoteDevice) -> Self {
        // `DeviceHandle::new` initializes the service the first time it's called for a given
        // device id. `RemoteService::init` is deliberately cheap — it records the endpoint but
        // does NOT connect, because cubecl holds a process-global lock across it; the actual
        // connect + handshake happens lazily on first use (or via `ensure_connected`).
        // Subsequent calls return a handle to the existing service.
        let handle = DeviceHandle::<RemoteService>::new(device.to_id());
        Self { device, handle }
    }

    /// Force the lazily-established connection to be opened now, populating the device's
    /// settings/device-count cells. Used by the settings path (`RemoteDevice::defaults` /
    /// `enumerate`), which needs the handshake reply before any op has flushed. Runs the
    /// connect on the service's runner thread, so it can't sit under cubecl's global lock.
    pub(crate) fn ensure_connected(&self) {
        self.handle
            .submit_blocking(|s| s.ensure_connected())
            .expect("Service call failed");
    }
}

impl Clone for RemoteClient {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            handle: self.handle.clone(),
        }
    }
}
