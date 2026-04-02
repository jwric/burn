#![cfg(feature = "dylib")]

use alloc::string::String;
use core::marker::PhantomData;

use burn_backend::{Backend, DType, DTypeUsageSet, ExecutionError};

use super::device::DylibDevice;
use super::runtime;
use super::tensor::DylibTensor;

/// Runtime-loaded backend modeled as a concrete Burn backend type.
pub struct Dylib<E: Send + Sync + 'static = f32>(PhantomData<E>);

impl<E: Send + Sync + 'static> Clone for Dylib<E> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<E: Send + Sync + 'static> Default for Dylib<E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<E: Send + Sync + 'static> core::fmt::Debug for Dylib<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Dylib")
    }
}

impl<E: Send + Sync + 'static> Backend for Dylib<E> {
    type Device = DylibDevice;
    type FloatTensorPrimitive = DylibTensor;
    type FloatElem = f32;

    type IntTensorPrimitive = DylibTensor;
    type IntElem = i32;

    type BoolTensorPrimitive = DylibTensor;
    type BoolElem = u8;

    type QuantizedTensorPrimitive = DylibTensor;

    fn name(device: &Self::Device) -> String {
        runtime::backend_name(device)
    }

    fn seed(device: &Self::Device, seed: u64) {
        runtime::backend_seed(device, seed)
    }

    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        runtime::backend_sync(device)
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        runtime::dtype_usage(dtype)
    }

    fn device_count(_type_id: u16) -> usize {
        // Runtime-loaded backends are created from explicit device handles.
        0
    }
}
