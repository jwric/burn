#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::IntTensorOps;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{
    BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> IntTensorOps<Dylib<E>> for Dylib<E> {
    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        runtime::int_tensor_empty(shape, device, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_into_data(
        tensor: IntTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        core::future::ready(
            runtime::int_tensor_into_data(tensor).map_err(runtime::to_execution_error),
        )
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        runtime::int_tensor_from_data(data, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        runtime::int_tensor_to_device(tensor, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        runtime::int_tensor_reshape(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        runtime::int_tensor_slice(tensor, slices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_slice_assign(tensor, slices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        runtime::int_tensor_into_float(tensor, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_mask_where(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> IntTensor<Self> {
        runtime::int_tensor_mask_fill(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_gather(dim, tensor, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_scatter_add(dim, tensor, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_select(tensor, dim, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        runtime::int_tensor_select_add(tensor, dim, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_equal(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        runtime::int_tensor_equal_elem(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_greater(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_greater(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_greater_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_greater_elem(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_greater_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_greater_equal(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_greater_equal_elem(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_lower(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_lower(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        runtime::int_tensor_lower_elem(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_lower_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_lower_equal(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::int_tensor_lower_equal_elem(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_add(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_add_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_sub(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_sub_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_mul(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_mul_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_div(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_div_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_remainder(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_remainder_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_matmul(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_sum(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_sum_dim(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_prod(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_prod_dim(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_mean_dim(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_cumsum(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_cumprod(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_cummin(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_cummax(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_argmax(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        runtime::int_tensor_argmin(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_abs(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        runtime::int_tensor_swap_dims(tensor, dim1, dim2).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        runtime::int_tensor_permute(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        runtime::int_tensor_flip(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        runtime::int_tensor_random(shape, distribution, device, dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        runtime::int_tensor_expand(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_and(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_and_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_or(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_or_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_xor(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_xor_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_not(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_left_shift(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_left_shift_scalar(lhs, rhs)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_right_shift(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        runtime::int_tensor_bitwise_right_shift_scalar(lhs, rhs)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        runtime::int_tensor_cast(tensor, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        runtime::int_tensor_unfold(tensor, dim, size, step).unwrap_or_else(|err| panic!("{err}"))
    }
}
