#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::{ActivationOps, QTensorOps, TransactionOps};
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::tensor::{Device, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{ExecutionError, FloatDType, Shape, Slice, TensorData};

use super::super::backend::Dylib;
use super::unsupported_op;

impl<E: Send + Sync + 'static> QTensorOps<Dylib<E>> for Dylib<E> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        unsupported_op("q_from_data");
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        unsupported_op("quantize");
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        unsupported_op("dequantize");
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        unsupported_op("q_device");
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        unsupported_op("q_to_device");
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        unsupported_op("q_reshape");
    }

    fn q_into_data(
        tensor: QuantizedTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { unsupported_op("q_into_data") }
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        unsupported_op("q_expand");
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        unsupported_op("q_swap_dims");
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unsupported_op("q_permute");
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        unsupported_op("q_flip");
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unsupported_op("q_select");
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        unsupported_op("q_slice");
    }
}

impl<E: Send + Sync + 'static> ActivationOps<Dylib<E>> for Dylib<E> {}

impl<E: Send + Sync + 'static> TransactionOps<Dylib<E>> for Dylib<E> {}
