use burn_dylib::adapter::{
    DenseTensorData, FloatTensorPlugin, PluginError, PluginMetadata, PluginResult,
};
use burn_dylib::{DenseTensorBinaryOp, DenseTensorDType};

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
    type IntTensor = MockTensor;
    type BoolTensor = MockTensor;

    fn dense_float_from_data(
        _device: &Self::Device,
        data: DenseTensorData,
    ) -> PluginResult<Self::FloatTensor> {
        if data.dtype != DenseTensorDType::F32 {
            return Err(PluginError::unsupported(b"only f32 dense tensors are supported\0"));
        }
        if data.bytes.len() % core::mem::size_of::<f32>() != 0 {
            return Err(invalid_argument());
        }

        let values = data
            .bytes
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk size should match")))
            .collect::<Vec<_>>();

        create_tensor(&data.shape, &values)
    }

    fn dense_float_into_data(tensor: &Self::FloatTensor) -> PluginResult<DenseTensorData> {
        let bytes = tensor
            .data
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();

        Ok(DenseTensorData {
            dtype: DenseTensorDType::F32,
            shape: tensor.shape.clone(),
            bytes,
        })
    }

    fn float_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>> {
        Ok(tensor.shape.clone())
    }

    fn float_binary(
        op: DenseTensorBinaryOp,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        match op {
            DenseTensorBinaryOp::Add => tensor_add_impl(lhs, rhs),
            DenseTensorBinaryOp::Matmul => tensor_matmul_impl(lhs, rhs),
            _ => Err(PluginError::unsupported(b"float op not implemented\0")),
        }
    }
}

burn_dylib::export_plugin_api!(MockPlugin);
