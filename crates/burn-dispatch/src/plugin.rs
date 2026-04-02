#![cfg(feature = "plugin")]
//! Export helpers for building `burn-dispatch` as a `burn-dylib` backend plugin.
//!
//! The current plugin ABI can represent child backends that are instantiable from
//! `type_id + ordinal`. That includes built-in backends like `ndarray`, `cuda`,
//! `wgpu`, and `tch`. Nested `dylib` children are rejected for now because they
//! require backend-specific device descriptors such as a child plugin path.
//!
//! Autodiff metadata is not preserved across the plugin boundary; exported tensor
//! ops use the eager `Dispatch` backend surface.

use burn_backend::ops::FloatTensorOps;
use burn_backend::{
    Backend, DType, Device as BurnDevice, DeviceId, Shape, TensorData, TensorMetadata,
};
use burn_dylib::adapter::{
    DenseTensorData, FloatTensorPlugin, PluginError, PluginMetadata, PluginResult,
};
use burn_dylib::{BackendPluginV1, BackendTensorOpsV1, DenseTensorBinaryOp, DenseTensorDType};

#[cfg(feature = "dylib")]
use crate::BackendId;
use crate::{Dispatch, DispatchDevice, DispatchTensor};

const NAME: &[u8] = b"dispatch\0";
const ERR_EXECUTION: &[u8] = b"dispatch execution failed\0";
const ERR_UNSUPPORTED_NESTED_DYLIB: &[u8] =
    b"nested dylib devices require a backend-specific device descriptor; the current ABI only supports type_id + ordinal\0";

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

/// Trait-backed dispatch plugin implementation.
pub struct DispatchPlugin;

/// Static plugin descriptor for exporting the dispatch backend as a dylib plugin.
pub static DISPATCH_PLUGIN_V1: BackendPluginV1 =
    burn_dylib::adapter::backend_plugin_v1::<DispatchPlugin>();

/// Static tensor-op descriptor for exporting the dispatch backend as a dylib plugin.
pub static DISPATCH_TENSOR_OPS_V1: BackendTensorOpsV1 =
    burn_dylib::adapter::backend_tensor_ops_v1::<DispatchPlugin>();

fn execution_failed() -> PluginError {
    PluginError::failed(ERR_EXECUTION)
}

#[cfg(feature = "dylib")]
fn is_nested_dylib(type_id: u16) -> bool {
    matches!(DispatchDevice::decode_type_id(type_id).0, BackendId::Dylib)
}

#[cfg(not(feature = "dylib"))]
fn is_nested_dylib(_type_id: u16) -> bool {
    false
}

impl PluginMetadata for DispatchPlugin {
    type Device = DispatchDevice;

    fn backend_name() -> &'static [u8] {
        NAME
    }

    fn seed(seed: u64, devices: &[Self::Device]) -> PluginResult<()> {
        for device in devices {
            Dispatch::seed(device, seed);
        }
        Ok(())
    }

    fn sync(devices: &[Self::Device]) -> PluginResult<()> {
        for device in devices {
            Dispatch::sync(device).map_err(|_| execution_failed())?;
        }
        Ok(())
    }

    fn device_count(type_id: u16) -> usize {
        if is_nested_dylib(type_id) {
            return 0;
        }

        Dispatch::device_count(type_id)
    }

    fn create_device(type_id: u16, ordinal: usize) -> PluginResult<Self::Device> {
        if is_nested_dylib(type_id) {
            return Err(PluginError::unsupported(ERR_UNSUPPORTED_NESTED_DYLIB));
        }

        let ordinal = u32::try_from(ordinal)
            .map_err(|_| PluginError::invalid_argument(b"invalid argument\0"))?;

        Ok(DispatchDevice::from_id(DeviceId::new(type_id, ordinal)))
    }
}

impl FloatTensorPlugin for DispatchPlugin {
    type FloatTensor = DispatchTensor;
    type IntTensor = DispatchTensor;
    type BoolTensor = DispatchTensor;

    fn dense_float_from_data(
        device: &Self::Device,
        data: DenseTensorData,
    ) -> PluginResult<Self::FloatTensor> {
        if data.dtype != DenseTensorDType::F32 {
            return Err(PluginError::unsupported(
                b"only f32 dense tensors are supported\0",
            ));
        }

        Ok(<Dispatch as FloatTensorOps<Dispatch>>::float_from_data(
            TensorData::from_bytes_vec(data.bytes, Shape::new_raw(data.shape.into()), DType::F32),
            device,
        ))
    }

    fn dense_float_into_data(tensor: &Self::FloatTensor) -> PluginResult<DenseTensorData> {
        let data = burn_backend::read_sync(
            <Dispatch as FloatTensorOps<Dispatch>>::float_into_data(tensor.clone()),
        )
        .map_err(|_| execution_failed())?;

        if data.dtype != DType::F32 {
            return Err(execution_failed());
        }

        Ok(DenseTensorData {
            dtype: DenseTensorDType::F32,
            shape: data.shape.as_slice().to_vec(),
            bytes: data.as_bytes().to_vec(),
        })
    }

    fn float_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>> {
        Ok(tensor.shape().as_slice().to_vec())
    }

    fn float_binary(
        op: DenseTensorBinaryOp,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        Ok(match op {
            DenseTensorBinaryOp::Add => {
                <Dispatch as FloatTensorOps<Dispatch>>::float_add(lhs.clone(), rhs.clone())
            }
            DenseTensorBinaryOp::Matmul => {
                <Dispatch as FloatTensorOps<Dispatch>>::float_matmul(lhs.clone(), rhs.clone())
            }
            _ => return Err(PluginError::unsupported(b"float op not implemented\0")),
        })
    }
}

/// Export the dispatch backend plugin symbols from a final `cdylib` crate.
#[macro_export]
macro_rules! export_dispatch_plugin_v1 {
    () => {
        $crate::burn_dylib::export_backend_plugin_v1!($crate::plugin::DISPATCH_PLUGIN_V1);
        $crate::burn_dylib::export_backend_tensor_ops_v1!($crate::plugin::DISPATCH_TENSOR_OPS_V1);
    };
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::slice;

    use burn_backend::DeviceOps;
    use burn_dylib::{
        ByteSliceRef, DenseTensorBinaryOp, DenseTensorDType, DenseTensorDataRef, DenseTensorKind,
        DeviceHandle, OwnedDenseTensorData, PluginStatus, PluginStatusCode, TensorHandle,
        TensorShapeRef,
    };

    use super::*;

    fn clear_state() {
        burn_dylib::adapter::reset_state::<DispatchPlugin>();
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
        let bytes = data
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.dense_tensor_from_data)(
                DenseTensorKind::Float,
                device,
                DenseTensorDataRef {
                    dtype: DenseTensorDType::F32,
                    shape: TensorShapeRef {
                        dims: shape.as_ptr(),
                        rank: shape.len(),
                    },
                    bytes: ByteSliceRef {
                        ptr: bytes.as_ptr(),
                        len: bytes.len(),
                    },
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
        let mut buffer = OwnedDenseTensorData::empty(DenseTensorDType::F32);
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.dense_tensor_into_data)(
                DenseTensorKind::Float,
                handle,
                &mut buffer,
            )
        };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );

        assert_eq!(buffer.dtype, DenseTensorDType::F32);

        let values = if buffer.bytes.len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(buffer.bytes.ptr, buffer.bytes.len) }
                .chunks_exact(core::mem::size_of::<f32>())
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk size should match")))
                .collect()
        };

        let status = unsafe { (DISPATCH_TENSOR_OPS_V1.release_byte_buffer)(buffer.bytes) };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );

        let status = unsafe { (DISPATCH_TENSOR_OPS_V1.release_usize_buffer)(buffer.shape) };
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
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.dense_tensor_binary)(
                DenseTensorKind::Float,
                DenseTensorBinaryOp::Add,
                lhs,
                rhs,
                &mut add,
            )
        };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );
        assert_eq!(read_tensor(add), vec![6.0, 8.0, 10.0, 12.0]);

        let lhs = create_test_tensor(device, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rhs = create_test_tensor(device, &[2, 2], &[2.0, 0.0, 1.0, 2.0]);

        let mut matmul = TensorHandle::INVALID;
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.dense_tensor_binary)(
                DenseTensorKind::Float,
                DenseTensorBinaryOp::Matmul,
                lhs,
                rhs,
                &mut matmul,
            )
        };
        assert_eq!(
            status.code,
            PluginStatusCode::Ok,
            "{:?}",
            status_message(status)
        );
        assert_eq!(read_tensor(matmul), vec![4.0, 4.0, 10.0, 8.0]);
    }

    #[test]
    #[cfg(feature = "dylib")]
    fn nested_dylib_requires_device_descriptor() {
        clear_state();

        let mut device = DeviceHandle::INVALID;
        let status = unsafe {
            (DISPATCH_TENSOR_OPS_V1.create_device)(PluginBackendId::Dylib.encode(0), 0, &mut device)
        };

        assert_eq!(status.code, PluginStatusCode::Unsupported);
        assert!(
            status_message(status)
                .contains("nested dylib devices require a backend-specific device descriptor")
        );
    }
}
