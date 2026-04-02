#![cfg(feature = "dylib")]

use std::sync::Arc;

use burn_backend::{DType, QTensorPrimitive, Shape, TensorMetadata, quantization::QuantScheme};
use burn_dylib::TensorHandle;

use super::device::DylibDevice;
use super::runtime;

#[derive(Debug)]
struct DylibTensorDropGuard {
    runtime_id: u64,
    handle: TensorHandle,
}

impl Drop for DylibTensorDropGuard {
    fn drop(&mut self) {
        runtime::release_tensor(self.runtime_id, self.handle);
    }
}

/// Tensor managed by a runtime-loaded backend.
#[derive(Clone, Debug)]
pub struct DylibTensor {
    pub(crate) runtime_id: u64,
    pub(crate) device: DylibDevice,
    pub(crate) handle: TensorHandle,
    _guard: Arc<DylibTensorDropGuard>,
    pub(crate) dtype: DType,
    pub(crate) shape: Shape,
}

impl DylibTensor {
    pub(crate) fn new(
        runtime_id: u64,
        device: DylibDevice,
        handle: TensorHandle,
        dtype: DType,
        shape: Shape,
    ) -> Self {
        let guard = Arc::new(DylibTensorDropGuard { runtime_id, handle });

        Self {
            runtime_id,
            device,
            handle,
            _guard: guard,
            dtype,
            shape,
        }
    }
}

impl TensorMetadata for DylibTensor {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

impl QTensorPrimitive for DylibTensor {
    fn scheme(&self) -> &QuantScheme {
        panic!("Quantized operations are not supported for dylib backend.")
    }
}
