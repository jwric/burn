#![allow(unused_variables)]

use core::future::Future;

use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{
    BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

use super::super::backend::Dylib;
use super::super::runtime;
use super::unsupported_op;

impl<E: Send + Sync + 'static> FloatTensorOps<Dylib<E>> for Dylib<E> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        runtime::float_from_data(data, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        unsupported_op("float_random");
    }

    fn float_into_data(
        tensor: FloatTensor<Self>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        core::future::ready(runtime::float_into_data(tensor).map_err(runtime::to_execution_error))
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        runtime::float_to_device(tensor, device).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_into_int(tensor: FloatTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        unsupported_op("float_into_int");
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        if dtype != FloatDType::F32 {
            unsupported_op("float_empty_non_f32");
        }
        Self::float_from_data(TensorData::zeros::<f32, _>(shape), device)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::float_add(lhs, rhs).unwrap_or_else(|err| panic!("{err}"))
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_add_scalar");
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_sub");
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_sub_scalar");
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_mul");
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_mul_scalar");
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_div");
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_div_scalar");
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_remainder");
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_remainder_scalar");
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let _ = (lhs, rhs);
        unsupported_op("float_matmul");
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        unsupported_op("float_cross");
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_recip");
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        unsupported_op("float_swap_dims");
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unsupported_op("float_permute");
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unsupported_op("float_flip");
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unsupported_op("float_reshape");
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_gather");
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_scatter_add");
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_select");
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_select_add");
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        unsupported_op("float_slice");
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_slice_assign");
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        unsupported_op("float_mask_where");
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        unsupported_op("float_mask_fill");
    }

    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_equal");
    }

    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_equal_elem");
    }

    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_greater");
    }

    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_greater_elem");
    }

    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_greater_equal");
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_greater_equal_elem");
    }

    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_lower");
    }

    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_lower_elem");
    }

    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_lower_equal");
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unsupported_op("float_lower_equal_elem");
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_sum");
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_sum_dim");
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_mean_dim");
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_cumsum");
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_cumprod");
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_cummin");
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unsupported_op("float_cummax");
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        unsupported_op("float_cast");
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_exp");
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_log");
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_log1p");
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_powf");
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        unsupported_op("float_powf_scalar_impl");
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_sqrt");
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_abs");
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_cos");
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_sin");
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_tan");
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_cosh");
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_sinh");
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_tanh");
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_acos");
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_acosh");
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_asin");
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_asinh");
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_atan");
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_atanh");
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_atan2");
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_round");
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_floor");
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_ceil");
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_trunc");
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unsupported_op("float_erf");
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        unsupported_op("float_argmax");
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        unsupported_op("float_argmin");
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unsupported_op("float_expand");
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unsupported_op("float_unfold");
    }
}
