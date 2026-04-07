#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::{ActivationOps, QTensorOps, TransactionOps};
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::tensor::{Device, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{ExecutionError, FloatDType, Shape, Slice, TensorData};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> QTensorOps<Dylib<E>> for Dylib<E> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        runtime::q_tensor_from_data(data, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        runtime::q_tensor_quantize(tensor, *scheme, qparams.scales)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        runtime::q_tensor_dequantize(tensor, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        runtime::q_tensor_to_device(tensor, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        runtime::q_tensor_reshape(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_into_data(
        tensor: QuantizedTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        core::future::ready(
            runtime::q_tensor_into_data(tensor).map_err(runtime::to_execution_error),
        )
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        runtime::q_tensor_expand(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        runtime::q_tensor_swap_dims(tensor, dim1, dim2).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        runtime::q_tensor_permute(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        runtime::q_tensor_flip(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        runtime::q_tensor_select(tensor, dim, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        runtime::q_tensor_slice(tensor, slices).unwrap_or_else(|err| panic!("{err}"))
    }
}

impl<E: Send + Sync + 'static> ActivationOps<Dylib<E>> for Dylib<E> {}

impl<E: Send + Sync + 'static> TransactionOps<Dylib<E>> for Dylib<E> {}
