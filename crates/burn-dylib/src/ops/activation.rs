use burn_backend::Scalar;
use burn_backend::ops::ActivationOps;
use burn_backend::tensor::FloatTensor;

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> ActivationOps<Dylib<E>> for Dylib<E> {
    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        runtime::activation_leaky_relu(tensor, negative_slope).unwrap_or_else(|err| panic!("{err}"))
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_relu(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_relu_backward(output, grad).unwrap_or_else(|err| panic!("{err}"))
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_gelu(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_prelu(tensor, alpha).unwrap_or_else(|err| panic!("{err}"))
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_gelu_backward(x, grad).unwrap_or_else(|err| panic!("{err}"))
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_sigmoid(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_sigmoid_backward(output, grad).unwrap_or_else(|err| panic!("{err}"))
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        runtime::activation_hard_sigmoid(tensor, alpha, beta).unwrap_or_else(|err| panic!("{err}"))
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_log_sigmoid(tensor).unwrap_or_else(|err| panic!("{err}"))
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        runtime::activation_log_sigmoid_backward(x, grad).unwrap_or_else(|err| panic!("{err}"))
    }
}
