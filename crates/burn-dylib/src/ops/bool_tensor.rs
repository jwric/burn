#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::BoolTensorOps;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{
    BoolDType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> BoolTensorOps<Dylib<E>> for Dylib<E> {
    fn bool_empty(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        runtime::bool_tensor_empty(shape, device, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        runtime::bool_tensor_zeros(shape, device, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_ones(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        runtime::bool_tensor_ones(shape, device, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_into_data(
        tensor: BoolTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        core::future::ready(
            runtime::bool_tensor_into_data(tensor).map_err(runtime::to_execution_error),
        )
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_from_data(data, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        runtime::bool_tensor_into_int(tensor, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        runtime::bool_tensor_into_float(tensor, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_to_device(tensor, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        runtime::bool_tensor_reshape(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        runtime::bool_tensor_slice(tensor, slices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_slice_assign(tensor, slices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_mask_where(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_mask_fill(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_gather(dim, tensor, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_scatter_or(dim, tensor, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_select(tensor, dim, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_select_or(tensor, dim, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_equal(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_and(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_or(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        runtime::bool_tensor_not(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        runtime::bool_tensor_swap_dims(tensor, dim1, dim2).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        runtime::bool_tensor_permute(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        runtime::bool_tensor_flip(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        runtime::bool_tensor_expand(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        runtime::bool_tensor_equal_elem(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        runtime::bool_tensor_unfold(tensor, dim, size, step).unwrap_or_else(|err| panic!("{err}"))
    }
}
