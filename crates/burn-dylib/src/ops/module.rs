#![allow(unused_variables)]

use burn_backend::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
    DeformConvOptions, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> ModuleOps<Dylib<E>> for Dylib<E> {
    fn conv2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<2>,
    ) -> FloatTensor<Self> {
        runtime::module_conv2d(x, weight, bias, options).unwrap_or_else(|err| panic!("{err}"))
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        runtime::module_deform_conv2d(x, offset, weight, mask, bias, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        runtime::module_deform_conv2d_backward(x, offset, weight, mask, bias, output_grad, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        runtime::module_conv3d(x, weight, bias, options).unwrap_or_else(|err| panic!("{err}"))
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        runtime::module_conv_transpose2d(x, weight, bias, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        runtime::module_conv_transpose3d(x, weight, bias, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        runtime::module_avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .unwrap_or_else(|err| panic!("{err}"))
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        runtime::module_avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
        .unwrap_or_else(|err| panic!("{err}"))
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        runtime::module_adaptive_avg_pool2d(x, output_size).unwrap_or_else(|err| panic!("{err}"))
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        runtime::module_adaptive_avg_pool2d_backward(x, grad).unwrap_or_else(|err| panic!("{err}"))
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        runtime::module_max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        runtime::module_max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        .unwrap_or_else(|err| panic!("{err}"))
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> MaxPool2dBackward<Self> {
        runtime::module_max_pool2d_with_indices_backward(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            output_grad,
            indices,
        )
        .unwrap_or_else(|err| panic!("{err}"))
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        runtime::module_interpolate(x, output_size, options).unwrap_or_else(|err| panic!("{err}"))
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        runtime::module_interpolate_backward(x, grad, output_size, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionModuleOptions,
    ) -> FloatTensor<Self> {
        runtime::module_attention(query, key, value, mask, attn_bias, options)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn rfft(signal: FloatTensor<Self>, dim: usize) -> (FloatTensor<Self>, FloatTensor<Self>) {
        runtime::module_rfft(signal, dim).unwrap_or_else(|err| panic!("{err}"))
    }
}
