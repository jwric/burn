use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

/// A small multilayer perceptron mapping a scalar input to a scalar output.
#[derive(Module, Debug)]
pub struct Mlp {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    activation: Relu,
}

impl Mlp {
    pub fn new(hidden: usize, device: &Device) -> Self {
        Self {
            l1: LinearConfig::new(1, hidden).init(device),
            l2: LinearConfig::new(hidden, hidden).init(device),
            l3: LinearConfig::new(hidden, 1).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let x = self.activation.forward(self.l1.forward(input));
        let x = self.activation.forward(self.l2.forward(x));
        self.l3.forward(x)
    }
}
