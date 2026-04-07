#![cfg(feature = "plugin")]
//! Export helpers for building `burn-dispatch` as a `burn-dylib` backend plugin.
//!
//! The current plugin ABI can represent child backends that are instantiable from
//! `type_id + ordinal`. That includes built-in backends like `ndarray`, `cuda`,
//! `wgpu`, and `tch`. Nested `dylib` children are not instantiable because they
//! require backend-specific device descriptors such as a child plugin path.
//!
//! Autodiff metadata is not preserved across the plugin boundary; exported tensor
//! ops use the eager `Dispatch` backend surface.

use burn_dylib::{BackendPluginV1, BackendTensorOpsV1};

use crate::Dispatch;

#[cfg(test)]
use crate::DispatchDevice;

const NAME: &[u8] = b"dispatch\0";

/// Stable backend ids understood by the dispatch plugin ABI.
///
/// The discriminants are part of the plugin contract and match the dispatch
/// device encoding used internally by [`DispatchDevice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum PluginBackendId {
    #[cfg(feature = "cpu")]
    /// CPU backend.
    Cpu = 0,
    #[cfg(feature = "cuda")]
    /// CUDA backend.
    Cuda = 1,
    #[cfg(wgpu_metal)]
    /// Metal backend.
    Metal = 2,
    #[cfg(feature = "rocm")]
    /// ROCm backend.
    Rocm = 3,
    #[cfg(wgpu_vulkan)]
    /// Vulkan backend.
    Vulkan = 4,
    #[cfg(wgpu_webgpu)]
    /// WebGPU backend.
    Wgpu = 5,
    #[cfg(feature = "ndarray")]
    /// NdArray backend.
    NdArray = 6,
    #[cfg(feature = "tch")]
    /// LibTorch backend.
    LibTorch = 7,
    #[cfg(feature = "dylib")]
    /// Nested dylib backend.
    ///
    /// Device creation is currently rejected because the outer plugin ABI does
    /// not yet provide a backend-specific device descriptor.
    Dylib = 8,
}

impl PluginBackendId {
    /// Encodes a backend-local `backend_type_id` into the dispatch plugin type id.
    pub const fn encode(self, backend_type_id: u16) -> u16 {
        (self as u16) * crate::device::TYPE_ID_BASE + backend_type_id
    }
}

/// Static plugin descriptor for exporting the dispatch backend as a dylib plugin.
pub static DISPATCH_PLUGIN_V1: BackendPluginV1 =
    burn_dylib::adapter::backend_plugin_v1::<Dispatch>(backend_name);

/// Static tensor-op descriptor for exporting the dispatch backend as a dylib plugin.
pub static DISPATCH_TENSOR_OPS_V1: BackendTensorOpsV1 =
    burn_dylib::adapter::backend_tensor_ops_v1::<Dispatch>();

unsafe extern "C" fn backend_name() -> *const core::ffi::c_char {
    NAME.as_ptr().cast()
}

/// Export the dispatch backend plugin symbols from a final `cdylib` crate.
#[macro_export]
macro_rules! export_dispatch_plugin_v1 {
    () => {
        $crate::burn_dylib::export_backend_plugin_v1!($crate::plugin::DISPATCH_PLUGIN_V1);
        $crate::burn_dylib::export_backend_tensor_ops_v1!($crate::plugin::DISPATCH_TENSOR_OPS_V1);
    };
}

export_dispatch_plugin_v1!();

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::slice;

    use burn_backend::DeviceOps;
    use burn_dylib::{
        DeviceHandle, F32SliceRef, OwnedF32Buffer, PluginStatus, PluginStatusCode, TensorHandle,
        TensorShapeRef,
    };

    use super::*;

    fn clear_state() {
        burn_dylib::adapter::reset_state::<Dispatch>();
    }

    fn status_message(status: PluginStatus) -> String {
        if status.message.is_null() {
            return String::new();
        }

        unsafe { CStr::from_ptr(status.message) }
            .to_str()
            .expect("status message should be valid utf-8")
            .to_owned()
    }

    fn create_test_tensor(device: DeviceHandle, shape: &[usize], data: &[f32]) -> TensorHandle {
        let mut out = TensorHandle::INVALID;
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.tensor_from_f32_data)(
                device,
                TensorShapeRef {
                    dims: shape.as_ptr(),
                    rank: shape.len(),
                },
                F32SliceRef {
                    ptr: data.as_ptr(),
                    len: data.len(),
                },
                &mut out,
            )
        };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );
        out
    }

    fn read_tensor(handle: TensorHandle) -> Vec<f32> {
        let mut buffer = OwnedF32Buffer::empty();
        let status = unsafe { (DISPATCH_TENSOR_OPS_V1.tensor_into_f32_data)(handle, &mut buffer) };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );

        let values = if buffer.len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(buffer.ptr, buffer.len) }.to_vec()
        };

        let status = unsafe { (DISPATCH_TENSOR_OPS_V1.release_f32_buffer)(buffer) };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );
        values
    }

    #[test]
    #[cfg(feature = "ndarray")]
    fn plugin_runs_ndarray_backend() {
        clear_state();

        let inner_device = DispatchDevice::NdArray(Default::default());
        let inner_id = inner_device.id();

        let mut device = DeviceHandle::INVALID;
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.create_device)(
                inner_id.type_id,
                inner_id.index_id as usize,
                &mut device,
            )
        };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );

        let lhs = create_test_tensor(device, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rhs = create_test_tensor(device, &[2, 2], &[5.0, 6.0, 7.0, 8.0]);

        let mut add = TensorHandle::INVALID;
        let status = unsafe { (DISPATCH_TENSOR_OPS_V1.tensor_add)(lhs, rhs, &mut add) };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );
        assert_eq!(read_tensor(add), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[cfg(feature = "dylib")]
    fn nested_dylib_requires_device_descriptor() {
        clear_state();

        let mut device = DeviceHandle::INVALID;
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.create_device)(PluginBackendId::Dylib.encode(0), 0, &mut device)
        };

        assert_eq!(status.code, PluginStatusCode::InvalidArgument);
    }
}
