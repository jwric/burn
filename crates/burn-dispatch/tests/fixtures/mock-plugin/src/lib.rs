use burn_dylib::TensorBinaryOp;
use burn_dylib::adapter::{FloatTensorPlugin, PluginError, PluginMetadata, PluginResult};

#[cfg(not(feature = "variant-b"))]
const BACKEND_NAME_A: &[u8] = b"mock-plugin-a\0";
#[cfg(feature = "variant-b")]
const BACKEND_NAME_B: &[u8] = b"mock-plugin-b\0";
const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const ERR_FAILED: &[u8] = b"operation failed\0";

#[derive(Clone)]
struct MockTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct MockPlugin;

fn invalid_argument() -> PluginError {
    PluginError::invalid_argument(ERR_INVALID_ARGUMENT)
}

fn failed() -> PluginError {
    PluginError::failed(ERR_FAILED)
}

fn checked_numel(shape: &[usize]) -> Result<usize, PluginError> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(failed)
}

fn create_tensor(shape: &[usize], data: &[f32]) -> Result<MockTensor, PluginError> {
    if checked_numel(shape)? != data.len() {
        return Err(invalid_argument());
    }

    Ok(MockTensor {
        shape: shape.to_vec(),
        data: data.to_vec(),
    })
}

fn tensor_add_impl(lhs: &MockTensor, rhs: &MockTensor) -> Result<MockTensor, PluginError> {
    if lhs.shape != rhs.shape {
        return Err(invalid_argument());
    }

    let bias = if cfg!(feature = "variant-b") { 1.0 } else { 0.0 };
    let out_data = lhs
        .data
        .iter()
        .zip(rhs.data.iter())
        .map(|(l, r)| l + r + bias)
        .collect::<Vec<_>>();

    Ok(MockTensor {
        shape: lhs.shape.clone(),
        data: out_data,
    })
}

fn tensor_matmul_impl(lhs: &MockTensor, rhs: &MockTensor) -> Result<MockTensor, PluginError> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err(invalid_argument());
    }

    let m = lhs.shape[0];
    let k = lhs.shape[1];
    let rhs_k = rhs.shape[0];
    let n = rhs.shape[1];

    if k != rhs_k {
        return Err(invalid_argument());
    }

    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                let lhs_idx = row * k + inner;
                let rhs_idx = inner * n + col;
                acc += lhs.data[lhs_idx] * rhs.data[rhs_idx];
            }
            out[row * n + col] = acc;
        }
    }

    Ok(MockTensor {
        shape: vec![m, n],
        data: out,
    })
}

impl PluginMetadata for MockPlugin {
    type Device = ();

    fn backend_name() -> &'static [u8] {
        #[cfg(feature = "variant-b")]
        {
            return BACKEND_NAME_B;
        }

        #[cfg(not(feature = "variant-b"))]
        {
            BACKEND_NAME_A
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }

    fn create_device(_type_id: u16, _ordinal: usize) -> PluginResult<Self::Device> {
        Ok(())
    }
}

impl FloatTensorPlugin for MockPlugin {
    type FloatTensor = MockTensor;

    fn tensor_from_f32_data(
        _device: &Self::Device,
        shape: &[usize],
        data: &[f32],
    ) -> PluginResult<Self::FloatTensor> {
        create_tensor(shape, data)
    }

    fn tensor_into_f32_data(tensor: &Self::FloatTensor) -> PluginResult<Vec<f32>> {
        Ok(tensor.data.clone())
    }

    fn tensor_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>> {
        Ok(tensor.shape.clone())
    }

    fn tensor_binary(
        op: TensorBinaryOp,
        _device: &Self::Device,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        match op {
            TensorBinaryOp::Add => tensor_add_impl(lhs, rhs),
            TensorBinaryOp::Matmul => tensor_matmul_impl(lhs, rhs),
        }
    }
}

burn_dylib::export_plugin_api!(MockPlugin);
