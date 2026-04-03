#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{
    BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> FloatTensorOps<Dylib<E>> for Dylib<E> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_from_data(data, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_random(shape, distribution, device, dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_into_data(
        tensor: FloatTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        core::future::ready(
            runtime::float_tensor_into_data(tensor).map_err(runtime::to_execution_error),
        )
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_to_device(tensor, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        runtime::float_tensor_into_int(tensor, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        runtime::float_tensor_empty(shape, device, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_add(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_add_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_sub(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_sub_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_mul(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_mul_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_div(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_div_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_remainder(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_remainder_scalar(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_matmul(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_cross(lhs, rhs, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_recip(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        runtime::float_tensor_swap_dims(tensor, dim1, dim2).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        runtime::float_tensor_permute(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        runtime::float_tensor_flip(tensor, axes).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        runtime::float_tensor_reshape(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_gather(dim, tensor, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_scatter_add(dim, tensor, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_select(tensor, dim, indices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_select_add(tensor, dim, indices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        runtime::float_tensor_slice(tensor, slices).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_slice_assign(tensor, slices, value)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_mask_where(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_mask_fill(tensor, mask, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_equal(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_equal_elem(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_greater(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_greater_elem(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_greater_equal(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_greater_equal_elem(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_lower(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_lower_elem(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_lower_equal(lhs, rhs, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        runtime::float_tensor_lower_equal_elem(lhs, rhs, out_dtype)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_sum(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_sum_dim(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_mean_dim(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_cumsum(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_cumprod(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_cummin(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        runtime::float_tensor_cummax(tensor, dim).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        runtime::float_tensor_cast(tensor, dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_exp(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_log(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_log1p(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_powf(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        runtime::float_tensor_powf_scalar(tensor, value).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_sqrt(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_abs(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_cos(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_sin(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_tan(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_cosh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_sinh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_tanh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_acos(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_acosh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_asin(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_asinh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_atan(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_atanh(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_atan2(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_round(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_floor(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_ceil(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_trunc(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_tensor_erf(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        runtime::float_tensor_argmax(tensor, dim, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        runtime::float_tensor_argmin(tensor, dim, out_dtype).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        runtime::float_tensor_expand(tensor, shape).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        runtime::float_tensor_unfold(tensor, dim, size, step).unwrap_or_else(|err| panic!("{err}"))
    }
}
