use crate::{
    BACKEND_PLUGIN_ABI_VERSION, BACKEND_TENSOR_OPS_ABI_VERSION, BackendPluginV1,
    BackendTensorOpsV1, ByteSliceRef, DenseAxesRef, DenseDistribution, DenseScalarValue,
    DenseTensorArgOp, DenseTensorBinaryDimOp, DenseTensorBinaryOp, DenseTensorBoolDType,
    DenseTensorComparisonOp, DenseTensorConvertOp, DenseTensorDType, DenseTensorDataRef,
    DenseTensorHandleListRef, DenseTensorKind, DenseTensorPredicateReduceOp,
    DenseTensorReduceDimOp, DenseTensorReduceOp, DenseTensorScalarOp, DenseTensorScatterOp,
    DenseTensorSelectAssignOp, DenseTensorSlicesRef, DenseTensorTransformArgs,
    DenseTensorTransformOp, DenseTensorUnaryOp, DeviceHandle, F32SliceRef, OwnedByteBuffer,
    OwnedDenseTensorData, OwnedF32Buffer, OwnedUsizeBuffer, PluginStatus, PluginStatusCode,
    TensorBinaryOp, TensorHandle, TensorShapeRef,
};
use core::any::TypeId;
use core::ffi::c_char;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

const ERR_INVALID_ARGUMENT: &[u8] = b"invalid argument\0";
const ERR_PANIC: &[u8] = b"plugin panicked\0";
const ERR_UNSUPPORTED: &[u8] = b"operation not supported\0";

/// Result type used by the plugin adapter traits.
pub type PluginResult<T> = Result<T, PluginError>;

/// Error returned by trait-based plugin implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginError {
    code: PluginStatusCode,
    message: &'static [u8],
}

impl PluginError {
    /// Creates a new plugin error.
    ///
    /// `message` must point to a static, null-terminated string.
    pub const fn new(code: PluginStatusCode, message: &'static [u8]) -> Self {
        Self { code, message }
    }

    /// Creates an `InvalidArgument` error.
    pub const fn invalid_argument(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::InvalidArgument, message)
    }

    /// Creates a `Failed` error.
    pub const fn failed(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::Failed, message)
    }

    /// Creates an `Unsupported` error.
    pub const fn unsupported(message: &'static [u8]) -> Self {
        Self::new(PluginStatusCode::Unsupported, message)
    }

    fn into_status(self) -> PluginStatus {
        PluginStatus::error(self.code, self.message.as_ptr().cast())
    }
}

fn unsupported<T>() -> PluginResult<T> {
    Err(PluginError::unsupported(ERR_UNSUPPORTED))
}

/// Owned dense host tensor payload used by the trait-backed adapter surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTensorData {
    /// Tensor data type.
    pub dtype: DenseTensorDType,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Tensor bytes.
    pub bytes: Vec<u8>,
}

/// Owned transform arguments used by dense tensor transform families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTransformArgs {
    /// Shape payload.
    pub shape: Vec<usize>,
    /// Axes payload.
    pub axes: Vec<usize>,
    /// Primary dimension parameter.
    pub dim: usize,
    /// Secondary dimension parameter.
    pub dim2: usize,
    /// Size parameter used by operations like `unfold`.
    pub size: usize,
    /// Step parameter used by operations like `unfold`.
    pub step: usize,
    /// Repeat count used by operations like `repeat_dim`.
    pub times: usize,
}

/// Owned slice descriptor used by dense slicing families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseSliceSpec {
    /// Slice start index.
    pub start: isize,
    /// Slice end index.
    pub end: Option<isize>,
    /// Slice step.
    pub step: isize,
}

/// Trait for plugin metadata and device management.
pub trait PluginMetadata: Send + Sync + 'static {
    /// Concrete device state stored behind plugin handles.
    type Device: Clone + Send + Sync + 'static;

    /// Returns the backend name.
    ///
    /// The returned bytes must be a static, null-terminated string.
    fn backend_name() -> &'static [u8];

    /// Seeds the plugin backend.
    ///
    /// The adapter provides all currently registered devices so backends that
    /// only expose per-device seed APIs can still implement the global plugin
    /// callback cleanly.
    fn seed(seed: u64, devices: &[Self::Device]) -> PluginResult<()> {
        let _ = (seed, devices);
        Ok(())
    }

    /// Synchronizes the plugin backend.
    ///
    /// The adapter provides all currently registered devices so backends that
    /// only expose per-device synchronization APIs can still implement the
    /// global plugin callback cleanly.
    fn sync(devices: &[Self::Device]) -> PluginResult<()> {
        let _ = devices;
        Ok(())
    }

    /// Returns how many devices are available for `type_id`.
    fn device_count(type_id: u16) -> usize;

    /// Creates a concrete device for `type_id` and `ordinal`.
    fn create_device(type_id: u16, ordinal: usize) -> PluginResult<Self::Device>;
}

/// Trait for the float tensor operations exposed by the current plugin ABI.
pub trait FloatTensorPlugin: PluginMetadata {
    /// Concrete float tensor state stored behind plugin handles.
    type FloatTensor: Clone + Send + Sync + 'static;
    /// Concrete int tensor state stored behind plugin handles.
    type IntTensor: Clone + Send + Sync + 'static;
    /// Concrete bool tensor state stored behind plugin handles.
    type BoolTensor: Clone + Send + Sync + 'static;

    /// Creates a dense float tensor from raw host bytes.
    fn dense_float_from_data(
        device: &Self::Device,
        data: DenseTensorData,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (device, data);
        unsupported()
    }

    /// Materializes a dense float tensor as raw host bytes.
    fn dense_float_into_data(tensor: &Self::FloatTensor) -> PluginResult<DenseTensorData> {
        let _ = tensor;
        unsupported()
    }

    /// Creates a dense int tensor from raw host bytes.
    fn dense_int_from_data(
        device: &Self::Device,
        data: DenseTensorData,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (device, data);
        unsupported()
    }

    /// Materializes a dense int tensor as raw host bytes.
    fn dense_int_into_data(tensor: &Self::IntTensor) -> PluginResult<DenseTensorData> {
        let _ = tensor;
        unsupported()
    }

    /// Creates a dense bool tensor from raw host bytes.
    fn dense_bool_from_data(
        device: &Self::Device,
        data: DenseTensorData,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (device, data);
        unsupported()
    }

    /// Materializes a dense bool tensor as raw host bytes.
    fn dense_bool_into_data(tensor: &Self::BoolTensor) -> PluginResult<DenseTensorData> {
        let _ = tensor;
        unsupported()
    }

    /// Creates a tensor from host f32 data.
    fn tensor_from_f32_data(
        device: &Self::Device,
        shape: &[usize],
        data: &[f32],
    ) -> PluginResult<Self::FloatTensor> {
        let bytes = unsafe {
            slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data))
        }
        .to_vec();

        Self::dense_float_from_data(
            device,
            DenseTensorData {
                dtype: DenseTensorDType::F32,
                shape: shape.to_vec(),
                bytes,
            },
        )
    }

    /// Materializes a tensor into host f32 data.
    fn tensor_into_f32_data(tensor: &Self::FloatTensor) -> PluginResult<Vec<f32>> {
        let data = Self::dense_float_into_data(tensor)?;
        if data.dtype != DenseTensorDType::F32 {
            return unsupported();
        }
        if data.bytes.len() % core::mem::size_of::<f32>() != 0 {
            return Err(PluginError::failed(ERR_INVALID_ARGUMENT));
        }

        Ok(data
            .bytes
            .chunks_exact(core::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk size should match")))
            .collect())
    }

    /// Returns the tensor shape.
    fn tensor_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>> {
        Self::float_shape(tensor)
    }

    /// Returns the float tensor shape.
    fn float_shape(tensor: &Self::FloatTensor) -> PluginResult<Vec<usize>> {
        let _ = tensor;
        unsupported()
    }

    /// Returns the int tensor shape.
    fn int_shape(tensor: &Self::IntTensor) -> PluginResult<Vec<usize>> {
        let _ = tensor;
        unsupported()
    }

    /// Returns the bool tensor shape.
    fn bool_shape(tensor: &Self::BoolTensor) -> PluginResult<Vec<usize>> {
        let _ = tensor;
        unsupported()
    }

    /// Dispatches float binary tensor operations.
    fn tensor_binary(
        op: TensorBinaryOp,
        device: &Self::Device,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = device;
        let dense_op = match op {
            TensorBinaryOp::Add => DenseTensorBinaryOp::Add,
            TensorBinaryOp::Matmul => DenseTensorBinaryOp::Matmul,
        };

        Self::float_binary(dense_op, lhs, rhs)
    }

    /// Creates an empty float tensor.
    fn float_empty(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (device, shape, dtype);
        unsupported()
    }

    /// Creates an empty int tensor.
    fn int_empty(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (device, shape, dtype);
        unsupported()
    }

    /// Creates an empty bool tensor.
    fn bool_empty(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (device, shape, dtype);
        unsupported()
    }

    /// Creates a full float tensor.
    fn float_full(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
        value: DenseScalarValue,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (device, shape, dtype, value);
        unsupported()
    }

    /// Creates a full int tensor.
    fn int_full(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
        value: DenseScalarValue,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (device, shape, dtype, value);
        unsupported()
    }

    /// Creates a full bool tensor.
    fn bool_full(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
        value: DenseScalarValue,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (device, shape, dtype, value);
        unsupported()
    }

    /// Creates a random float tensor.
    fn float_random(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
        distribution: DenseDistribution,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (device, shape, dtype, distribution);
        unsupported()
    }

    /// Creates a random int tensor.
    fn int_random(
        device: &Self::Device,
        shape: &[usize],
        dtype: DenseTensorDType,
        distribution: DenseDistribution,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (device, shape, dtype, distribution);
        unsupported()
    }

    /// Runs a float unary op.
    fn float_unary(
        op: DenseTensorUnaryOp,
        tensor: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor);
        unsupported()
    }

    /// Runs an int unary op.
    fn int_unary(
        op: DenseTensorUnaryOp,
        tensor: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor);
        unsupported()
    }

    /// Runs a bool unary op.
    fn bool_unary(
        op: DenseTensorUnaryOp,
        tensor: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor);
        unsupported()
    }

    /// Runs a float binary op.
    fn float_binary(
        op: DenseTensorBinaryOp,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, lhs, rhs);
        unsupported()
    }

    /// Runs an int binary op.
    fn int_binary(
        op: DenseTensorBinaryOp,
        lhs: &Self::IntTensor,
        rhs: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, lhs, rhs);
        unsupported()
    }

    /// Runs a bool binary op.
    fn bool_binary(
        op: DenseTensorBinaryOp,
        lhs: &Self::BoolTensor,
        rhs: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, lhs, rhs);
        unsupported()
    }

    /// Runs a float scalar op.
    fn float_scalar(
        op: DenseTensorScalarOp,
        tensor: &Self::FloatTensor,
        value: DenseScalarValue,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor, value);
        unsupported()
    }

    /// Runs an int scalar op.
    fn int_scalar(
        op: DenseTensorScalarOp,
        tensor: &Self::IntTensor,
        value: DenseScalarValue,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, value);
        unsupported()
    }

    /// Runs a float comparison op.
    fn float_comparison(
        op: DenseTensorComparisonOp,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, lhs, rhs, out_dtype);
        unsupported()
    }

    /// Runs an int comparison op.
    fn int_comparison(
        op: DenseTensorComparisonOp,
        lhs: &Self::IntTensor,
        rhs: &Self::IntTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, lhs, rhs, out_dtype);
        unsupported()
    }

    /// Runs a bool comparison op.
    fn bool_comparison(
        op: DenseTensorComparisonOp,
        lhs: &Self::BoolTensor,
        rhs: &Self::BoolTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, lhs, rhs, out_dtype);
        unsupported()
    }

    /// Runs a float comparison with scalar rhs.
    fn float_comparison_scalar(
        op: DenseTensorComparisonOp,
        tensor: &Self::FloatTensor,
        value: DenseScalarValue,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, value, out_dtype);
        unsupported()
    }

    /// Runs an int comparison with scalar rhs.
    fn int_comparison_scalar(
        op: DenseTensorComparisonOp,
        tensor: &Self::IntTensor,
        value: DenseScalarValue,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, value, out_dtype);
        unsupported()
    }

    /// Runs a bool comparison with scalar rhs.
    fn bool_comparison_scalar(
        op: DenseTensorComparisonOp,
        tensor: &Self::BoolTensor,
        value: DenseScalarValue,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, value, out_dtype);
        unsupported()
    }

    /// Runs a float reduction.
    fn float_reduce(
        op: DenseTensorReduceOp,
        tensor: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor);
        unsupported()
    }

    /// Runs an int reduction.
    fn int_reduce(
        op: DenseTensorReduceOp,
        tensor: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor);
        unsupported()
    }

    /// Runs a float dim reduction.
    fn float_reduce_dim(
        op: DenseTensorReduceDimOp,
        tensor: &Self::FloatTensor,
        dim: usize,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor, dim);
        unsupported()
    }

    /// Runs an int dim reduction.
    fn int_reduce_dim(
        op: DenseTensorReduceDimOp,
        tensor: &Self::IntTensor,
        dim: usize,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, dim);
        unsupported()
    }

    /// Runs a predicate reduction on a float tensor.
    fn float_predicate_reduce(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::FloatTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, out_dtype);
        unsupported()
    }

    /// Runs a predicate reduction on an int tensor.
    fn int_predicate_reduce(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::IntTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, out_dtype);
        unsupported()
    }

    /// Runs a predicate reduction on a bool tensor.
    fn bool_predicate_reduce(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::BoolTensor,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, out_dtype);
        unsupported()
    }

    /// Runs a dimensional predicate reduction on a float tensor.
    fn float_predicate_reduce_dim(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::FloatTensor,
        dim: usize,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, dim, out_dtype);
        unsupported()
    }

    /// Runs a dimensional predicate reduction on an int tensor.
    fn int_predicate_reduce_dim(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::IntTensor,
        dim: usize,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, dim, out_dtype);
        unsupported()
    }

    /// Runs a dimensional predicate reduction on a bool tensor.
    fn bool_predicate_reduce_dim(
        op: DenseTensorPredicateReduceOp,
        tensor: &Self::BoolTensor,
        dim: usize,
        out_dtype: DenseTensorBoolDType,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, dim, out_dtype);
        unsupported()
    }

    /// Runs an arg reduction on a float tensor.
    fn float_arg(
        op: DenseTensorArgOp,
        tensor: &Self::FloatTensor,
        dim: usize,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, dim, out_dtype);
        unsupported()
    }

    /// Runs an arg reduction on an int tensor.
    fn int_arg(
        op: DenseTensorArgOp,
        tensor: &Self::IntTensor,
        dim: usize,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, dim, out_dtype);
        unsupported()
    }

    /// Runs a float transform op.
    fn float_transform(
        op: DenseTensorTransformOp,
        tensor: &Self::FloatTensor,
        args: &DenseTransformArgs,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor, args);
        unsupported()
    }

    /// Runs an int transform op.
    fn int_transform(
        op: DenseTensorTransformOp,
        tensor: &Self::IntTensor,
        args: &DenseTransformArgs,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, args);
        unsupported()
    }

    /// Runs a bool transform op.
    fn bool_transform(
        op: DenseTensorTransformOp,
        tensor: &Self::BoolTensor,
        args: &DenseTransformArgs,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, args);
        unsupported()
    }

    /// Runs a float slice op.
    fn float_slice(
        tensor: &Self::FloatTensor,
        slices: &[DenseSliceSpec],
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, slices);
        unsupported()
    }

    /// Runs an int slice op.
    fn int_slice(
        tensor: &Self::IntTensor,
        slices: &[DenseSliceSpec],
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, slices);
        unsupported()
    }

    /// Runs a bool slice op.
    fn bool_slice(
        tensor: &Self::BoolTensor,
        slices: &[DenseSliceSpec],
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (tensor, slices);
        unsupported()
    }

    /// Runs a float slice-assign op.
    fn float_slice_assign(
        tensor: &Self::FloatTensor,
        slices: &[DenseSliceSpec],
        value: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, slices, value);
        unsupported()
    }

    /// Runs an int slice-assign op.
    fn int_slice_assign(
        tensor: &Self::IntTensor,
        slices: &[DenseSliceSpec],
        value: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, slices, value);
        unsupported()
    }

    /// Runs a bool slice-assign op.
    fn bool_slice_assign(
        tensor: &Self::BoolTensor,
        slices: &[DenseSliceSpec],
        value: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (tensor, slices, value);
        unsupported()
    }

    /// Runs a float gather op.
    fn float_gather(
        dim: usize,
        tensor: &Self::FloatTensor,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (dim, tensor, indices);
        unsupported()
    }

    /// Runs an int gather op.
    fn int_gather(
        dim: usize,
        tensor: &Self::IntTensor,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (dim, tensor, indices);
        unsupported()
    }

    /// Runs a bool gather op.
    fn bool_gather(
        dim: usize,
        tensor: &Self::BoolTensor,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (dim, tensor, indices);
        unsupported()
    }

    /// Runs a float scatter op.
    fn float_scatter(
        op: DenseTensorScatterOp,
        dim: usize,
        tensor: &Self::FloatTensor,
        indices: &Self::IntTensor,
        value: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, dim, tensor, indices, value);
        unsupported()
    }

    /// Runs an int scatter op.
    fn int_scatter(
        op: DenseTensorScatterOp,
        dim: usize,
        tensor: &Self::IntTensor,
        indices: &Self::IntTensor,
        value: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, dim, tensor, indices, value);
        unsupported()
    }

    /// Runs a bool scatter op.
    fn bool_scatter(
        op: DenseTensorScatterOp,
        dim: usize,
        tensor: &Self::BoolTensor,
        indices: &Self::IntTensor,
        value: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, dim, tensor, indices, value);
        unsupported()
    }

    /// Runs a float select op.
    fn float_select(
        tensor: &Self::FloatTensor,
        dim: usize,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, dim, indices);
        unsupported()
    }

    /// Runs an int select op.
    fn int_select(
        tensor: &Self::IntTensor,
        dim: usize,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, dim, indices);
        unsupported()
    }

    /// Runs a bool select op.
    fn bool_select(
        tensor: &Self::BoolTensor,
        dim: usize,
        indices: &Self::IntTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (tensor, dim, indices);
        unsupported()
    }

    /// Runs a float select-assign op.
    fn float_select_assign(
        op: DenseTensorSelectAssignOp,
        tensor: &Self::FloatTensor,
        dim: usize,
        indices: &Self::IntTensor,
        value: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, tensor, dim, indices, value);
        unsupported()
    }

    /// Runs an int select-assign op.
    fn int_select_assign(
        op: DenseTensorSelectAssignOp,
        tensor: &Self::IntTensor,
        dim: usize,
        indices: &Self::IntTensor,
        value: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (op, tensor, dim, indices, value);
        unsupported()
    }

    /// Runs a bool select-assign op.
    fn bool_select_assign(
        op: DenseTensorSelectAssignOp,
        tensor: &Self::BoolTensor,
        dim: usize,
        indices: &Self::IntTensor,
        value: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (op, tensor, dim, indices, value);
        unsupported()
    }

    /// Runs a float mask-where op.
    fn float_mask_where(
        tensor: &Self::FloatTensor,
        mask: &Self::BoolTensor,
        value: &Self::FloatTensor,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Runs an int mask-where op.
    fn int_mask_where(
        tensor: &Self::IntTensor,
        mask: &Self::BoolTensor,
        value: &Self::IntTensor,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Runs a bool mask-where op.
    fn bool_mask_where(
        tensor: &Self::BoolTensor,
        mask: &Self::BoolTensor,
        value: &Self::BoolTensor,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Runs a float mask-fill op.
    fn float_mask_fill(
        tensor: &Self::FloatTensor,
        mask: &Self::BoolTensor,
        value: DenseScalarValue,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Runs an int mask-fill op.
    fn int_mask_fill(
        tensor: &Self::IntTensor,
        mask: &Self::BoolTensor,
        value: DenseScalarValue,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Runs a bool mask-fill op.
    fn bool_mask_fill(
        tensor: &Self::BoolTensor,
        mask: &Self::BoolTensor,
        value: DenseScalarValue,
    ) -> PluginResult<Self::BoolTensor> {
        let _ = (tensor, mask, value);
        unsupported()
    }

    /// Concatenates float tensors.
    fn float_cat(tensors: &[Self::FloatTensor], dim: usize) -> PluginResult<Self::FloatTensor> {
        let _ = (tensors, dim);
        unsupported()
    }

    /// Concatenates int tensors.
    fn int_cat(tensors: &[Self::IntTensor], dim: usize) -> PluginResult<Self::IntTensor> {
        let _ = (tensors, dim);
        unsupported()
    }

    /// Concatenates bool tensors.
    fn bool_cat(tensors: &[Self::BoolTensor], dim: usize) -> PluginResult<Self::BoolTensor> {
        let _ = (tensors, dim);
        unsupported()
    }

    /// Casts a float tensor.
    fn float_cast(
        tensor: &Self::FloatTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Casts an int tensor.
    fn int_cast(
        tensor: &Self::IntTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Converts a float tensor into an int tensor.
    fn float_into_int(
        tensor: &Self::FloatTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Converts an int tensor into a float tensor.
    fn int_into_float(
        tensor: &Self::IntTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Converts a bool tensor into an int tensor.
    fn bool_into_int(
        tensor: &Self::BoolTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Converts a bool tensor into a float tensor.
    fn bool_into_float(
        tensor: &Self::BoolTensor,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, out_dtype);
        unsupported()
    }

    /// Runs a float binary op with an extra dimension parameter.
    fn float_binary_dim(
        op: DenseTensorBinaryDimOp,
        lhs: &Self::FloatTensor,
        rhs: &Self::FloatTensor,
        dim: usize,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (op, lhs, rhs, dim);
        unsupported()
    }

    /// Sorts a float tensor.
    fn float_sort(
        tensor: &Self::FloatTensor,
        dim: usize,
        descending: bool,
    ) -> PluginResult<Self::FloatTensor> {
        let _ = (tensor, dim, descending);
        unsupported()
    }

    /// Sorts an int tensor.
    fn int_sort(
        tensor: &Self::IntTensor,
        dim: usize,
        descending: bool,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, dim, descending);
        unsupported()
    }

    /// Sorts a float tensor and also returns indices.
    fn float_sort_with_indices(
        tensor: &Self::FloatTensor,
        dim: usize,
        descending: bool,
    ) -> PluginResult<(Self::FloatTensor, Self::IntTensor)> {
        let _ = (tensor, dim, descending);
        unsupported()
    }

    /// Sorts an int tensor and also returns indices.
    fn int_sort_with_indices(
        tensor: &Self::IntTensor,
        dim: usize,
        descending: bool,
    ) -> PluginResult<(Self::IntTensor, Self::IntTensor)> {
        let _ = (tensor, dim, descending);
        unsupported()
    }

    /// Returns argsort indices for a float tensor.
    fn float_argsort(
        tensor: &Self::FloatTensor,
        dim: usize,
        descending: bool,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, dim, descending, out_dtype);
        unsupported()
    }

    /// Returns argsort indices for an int tensor.
    fn int_argsort(
        tensor: &Self::IntTensor,
        dim: usize,
        descending: bool,
        out_dtype: DenseTensorDType,
    ) -> PluginResult<Self::IntTensor> {
        let _ = (tensor, dim, descending, out_dtype);
        unsupported()
    }
}

#[derive(Clone)]
struct TensorState<T> {
    device_handle: u64,
    tensor: T,
}

#[derive(Clone)]
enum AnyTensorState<P: FloatTensorPlugin> {
    Float(TensorState<P::FloatTensor>),
    Int(TensorState<P::IntTensor>),
    Bool(TensorState<P::BoolTensor>),
}

struct AdapterState<P: FloatTensorPlugin> {
    next_device_id: AtomicU64,
    next_tensor_id: AtomicU64,
    devices: Mutex<HashMap<u64, P::Device>>,
    float_tensors: Mutex<HashMap<u64, TensorState<P::FloatTensor>>>,
    int_tensors: Mutex<HashMap<u64, TensorState<P::IntTensor>>>,
    bool_tensors: Mutex<HashMap<u64, TensorState<P::BoolTensor>>>,
}

impl<P: FloatTensorPlugin> AdapterState<P> {
    fn new() -> Self {
        Self {
            next_device_id: AtomicU64::new(1),
            next_tensor_id: AtomicU64::new(1),
            devices: Mutex::new(HashMap::new()),
            float_tensors: Mutex::new(HashMap::new()),
            int_tensors: Mutex::new(HashMap::new()),
            bool_tensors: Mutex::new(HashMap::new()),
        }
    }

    fn clear(&self) {
        self.next_device_id.store(1, Ordering::Relaxed);
        self.next_tensor_id.store(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").clear();
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .clear();
        self.int_tensors.lock().expect("int tensor lock").clear();
        self.bool_tensors.lock().expect("bool tensor lock").clear();
    }

    fn devices_snapshot(&self) -> Vec<P::Device> {
        self.devices
            .lock()
            .expect("device lock")
            .values()
            .cloned()
            .collect()
    }

    fn lookup_device(&self, handle: DeviceHandle) -> Result<P::Device, PluginStatus> {
        self.devices
            .lock()
            .expect("device lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_float_tensor(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::FloatTensor>, PluginStatus> {
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_int_tensor(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::IntTensor>, PluginStatus> {
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_bool_tensor(
        &self,
        handle: TensorHandle,
    ) -> Result<TensorState<P::BoolTensor>, PluginStatus> {
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .get(&handle.0)
            .cloned()
            .ok_or_else(invalid_argument)
    }

    fn lookup_any_tensor(&self, handle: TensorHandle) -> Result<AnyTensorState<P>, PluginStatus> {
        if let Some(state) = self
            .float_tensors
            .lock()
            .expect("float tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(AnyTensorState::Float(state));
        }
        if let Some(state) = self
            .int_tensors
            .lock()
            .expect("int tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(AnyTensorState::Int(state));
        }
        if let Some(state) = self
            .bool_tensors
            .lock()
            .expect("bool tensor lock")
            .get(&handle.0)
            .cloned()
        {
            return Ok(AnyTensorState::Bool(state));
        }

        Err(invalid_argument())
    }

    fn insert_device(&self, device: P::Device) -> DeviceHandle {
        let id = self.next_device_id.fetch_add(1, Ordering::Relaxed);
        self.devices.lock().expect("device lock").insert(id, device);
        DeviceHandle(id)
    }

    fn insert_float_tensor(
        &self,
        device_handle: DeviceHandle,
        tensor: P::FloatTensor,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .insert(
                id,
                TensorState {
                    device_handle: device_handle.0,
                    tensor,
                },
            );
        TensorHandle(id)
    }

    fn insert_int_tensor(&self, device_handle: DeviceHandle, tensor: P::IntTensor) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.int_tensors.lock().expect("int tensor lock").insert(
            id,
            TensorState {
                device_handle: device_handle.0,
                tensor,
            },
        );
        TensorHandle(id)
    }

    fn insert_bool_tensor(
        &self,
        device_handle: DeviceHandle,
        tensor: P::BoolTensor,
    ) -> TensorHandle {
        let id = self.next_tensor_id.fetch_add(1, Ordering::Relaxed);
        self.bool_tensors.lock().expect("bool tensor lock").insert(
            id,
            TensorState {
                device_handle: device_handle.0,
                tensor,
            },
        );
        TensorHandle(id)
    }

    fn release_device(&self, device: DeviceHandle) {
        self.devices.lock().expect("device lock").remove(&device.0);
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .retain(|_, tensor| tensor.device_handle != device.0);
    }

    fn release_tensor(&self, tensor: TensorHandle) {
        self.float_tensors
            .lock()
            .expect("float tensor lock")
            .remove(&tensor.0);
        self.int_tensors
            .lock()
            .expect("int tensor lock")
            .remove(&tensor.0);
        self.bool_tensors
            .lock()
            .expect("bool tensor lock")
            .remove(&tensor.0);
    }
}

static ADAPTER_STATES: LazyLock<Mutex<HashMap<TypeId, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn adapter_state<P: FloatTensorPlugin>() -> &'static AdapterState<P> {
    let mut states = ADAPTER_STATES.lock().expect("adapter state lock");
    let ptr = states
        .entry(TypeId::of::<P>())
        .or_insert_with(|| Box::into_raw(Box::new(AdapterState::<P>::new())) as usize);

    unsafe { &*(*ptr as *const AdapterState<P>) }
}

fn ok() -> PluginStatus {
    PluginStatus::ok()
}

fn invalid_argument() -> PluginStatus {
    PluginError::invalid_argument(ERR_INVALID_ARGUMENT).into_status()
}

fn with_boundary(f: impl FnOnce() -> PluginStatus) -> PluginStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(status) => status,
        Err(_) => PluginError::failed(ERR_PANIC).into_status(),
    }
}

fn status_from_result(result: PluginResult<()>) -> PluginStatus {
    match result {
        Ok(()) => ok(),
        Err(error) => error.into_status(),
    }
}

fn try_shape(shape: TensorShapeRef) -> Result<Vec<usize>, PluginStatus> {
    if shape.rank == 0 {
        return Ok(Vec::new());
    }
    if shape.dims.is_null() {
        return Err(invalid_argument());
    }

    let dims = unsafe { slice::from_raw_parts(shape.dims, shape.rank) };
    Ok(dims.to_vec())
}

fn try_f32_data(data: F32SliceRef) -> Result<Vec<f32>, PluginStatus> {
    if data.len == 0 {
        return Ok(Vec::new());
    }
    if data.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(data.ptr, data.len) };
    Ok(values.to_vec())
}

fn try_bytes(data: ByteSliceRef) -> Result<Vec<u8>, PluginStatus> {
    if data.len == 0 {
        return Ok(Vec::new());
    }
    if data.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(data.ptr, data.len) };
    Ok(values.to_vec())
}

fn try_dense_data(data: DenseTensorDataRef) -> Result<DenseTensorData, PluginStatus> {
    Ok(DenseTensorData {
        dtype: data.dtype,
        shape: try_shape(data.shape)?,
        bytes: try_bytes(data.bytes)?,
    })
}

fn try_axes(axes: DenseAxesRef) -> Result<Vec<usize>, PluginStatus> {
    if axes.len == 0 {
        return Ok(Vec::new());
    }
    if axes.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(axes.ptr, axes.len) };
    Ok(values.to_vec())
}

fn try_slices(slices: DenseTensorSlicesRef) -> Result<Vec<DenseSliceSpec>, PluginStatus> {
    if slices.len == 0 {
        return Ok(Vec::new());
    }
    if slices.ptr.is_null() {
        return Err(invalid_argument());
    }

    let values = unsafe { slice::from_raw_parts(slices.ptr, slices.len) };
    Ok(values
        .iter()
        .map(|slice| DenseSliceSpec {
            start: slice.start,
            end: if slice.has_end == 0 {
                None
            } else {
                Some(slice.end)
            },
            step: slice.step,
        })
        .collect())
}

fn try_transform_args(args: DenseTensorTransformArgs) -> Result<DenseTransformArgs, PluginStatus> {
    Ok(DenseTransformArgs {
        shape: try_shape(args.shape)?,
        axes: try_axes(args.axes)?,
        dim: args.dim,
        dim2: args.dim2,
        size: args.size,
        step: args.step,
        times: args.times,
    })
}

fn try_tensor_handle_list(
    list: DenseTensorHandleListRef,
) -> Result<Vec<TensorHandle>, PluginStatus> {
    if list.len == 0 {
        return Ok(Vec::new());
    }
    if list.ptr.is_null() {
        return Err(invalid_argument());
    }

    let handles = unsafe { slice::from_raw_parts(list.ptr, list.len) };
    Ok(handles.to_vec())
}

fn write_tensor(out_tensor: *mut TensorHandle, handle: TensorHandle) {
    unsafe {
        *out_tensor = handle;
    }
}

fn write_dense_tensor_data(out_data: *mut OwnedDenseTensorData, data: DenseTensorData) {
    let mut shape = data.shape;
    let mut bytes = data.bytes;
    let owned = OwnedDenseTensorData {
        dtype: data.dtype,
        shape: OwnedUsizeBuffer {
            ptr: shape.as_mut_ptr(),
            len: shape.len(),
        },
        bytes: OwnedByteBuffer {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
        },
    };

    std::mem::forget(shape);
    std::mem::forget(bytes);

    unsafe {
        *out_data = owned;
    }
}

unsafe extern "C" fn backend_name<P: FloatTensorPlugin>() -> *const c_char {
    P::backend_name().as_ptr().cast()
}

unsafe extern "C" fn seed<P: FloatTensorPlugin>(seed: u64) -> PluginStatus {
    with_boundary(|| {
        let devices = adapter_state::<P>().devices_snapshot();
        status_from_result(P::seed(seed, &devices))
    })
}

unsafe extern "C" fn sync<P: FloatTensorPlugin>() -> PluginStatus {
    with_boundary(|| {
        let devices = adapter_state::<P>().devices_snapshot();
        status_from_result(P::sync(&devices))
    })
}

unsafe extern "C" fn device_count<P: FloatTensorPlugin>(type_id: u16) -> usize {
    catch_unwind(AssertUnwindSafe(|| P::device_count(type_id))).unwrap_or(0)
}

unsafe extern "C" fn create_device<P: FloatTensorPlugin>(
    type_id: u16,
    ordinal: usize,
    out_device: *mut DeviceHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_device.is_null() {
            return invalid_argument();
        }

        let available = P::device_count(type_id);
        if available != 0 && ordinal >= available {
            return invalid_argument();
        }

        let device = match P::create_device(type_id, ordinal) {
            Ok(device) => device,
            Err(error) => return error.into_status(),
        };
        let handle = adapter_state::<P>().insert_device(device);

        unsafe {
            *out_device = handle;
        }
        ok()
    })
}

unsafe extern "C" fn release_device<P: FloatTensorPlugin>(device: DeviceHandle) -> PluginStatus {
    with_boundary(|| {
        adapter_state::<P>().release_device(device);
        ok()
    })
}

unsafe extern "C" fn tensor_from_f32_data<P: FloatTensorPlugin>(
    device: DeviceHandle,
    shape: TensorShapeRef,
    data: F32SliceRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };
        let values = match try_f32_data(data) {
            Ok(values) => values,
            Err(status) => return status,
        };

        let tensor = match P::tensor_from_f32_data(&device_state, &shape, &values) {
            Ok(tensor) => tensor,
            Err(error) => return error.into_status(),
        };
        let handle = adapter_state::<P>().insert_float_tensor(device, tensor);

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn tensor_into_f32_data<P: FloatTensorPlugin>(
    tensor: TensorHandle,
    out_data: *mut OwnedF32Buffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let tensor_state = match adapter_state::<P>().lookup_float_tensor(tensor) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let mut values = match P::tensor_into_f32_data(&tensor_state.tensor) {
            Ok(values) => values,
            Err(error) => return error.into_status(),
        };
        let buffer = OwnedF32Buffer {
            ptr: values.as_mut_ptr(),
            len: values.len(),
        };
        std::mem::forget(values);

        unsafe {
            *out_data = buffer;
        }
        ok()
    })
}

unsafe extern "C" fn tensor_shape<P: FloatTensorPlugin>(
    tensor: TensorHandle,
    out_shape: *mut OwnedUsizeBuffer,
) -> PluginStatus {
    with_boundary(|| {
        if out_shape.is_null() {
            return invalid_argument();
        }

        let mut dims = match adapter_state::<P>().lookup_any_tensor(tensor) {
            Ok(AnyTensorState::Float(state)) => match P::tensor_shape(&state.tensor) {
                Ok(dims) => dims,
                Err(error) => return error.into_status(),
            },
            Ok(AnyTensorState::Int(state)) => match P::int_shape(&state.tensor) {
                Ok(dims) => dims,
                Err(error) => return error.into_status(),
            },
            Ok(AnyTensorState::Bool(state)) => match P::bool_shape(&state.tensor) {
                Ok(dims) => dims,
                Err(error) => return error.into_status(),
            },
            Err(status) => return status,
        };
        let buffer = OwnedUsizeBuffer {
            ptr: dims.as_mut_ptr(),
            len: dims.len(),
        };
        std::mem::forget(dims);

        unsafe {
            *out_shape = buffer;
        }
        ok()
    })
}

unsafe extern "C" fn tensor_binary<P: FloatTensorPlugin>(
    op: TensorBinaryOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let lhs_state = match adapter_state::<P>().lookup_float_tensor(lhs) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let rhs_state = match adapter_state::<P>().lookup_float_tensor(rhs) {
            Ok(state) => state,
            Err(status) => return status,
        };

        if lhs_state.device_handle != rhs_state.device_handle {
            return invalid_argument();
        }

        let device = match adapter_state::<P>().lookup_device(DeviceHandle(lhs_state.device_handle))
        {
            Ok(device) => device,
            Err(status) => return status,
        };

        let out = match P::tensor_binary(op, &device, &lhs_state.tensor, &rhs_state.tensor) {
            Ok(out) => out,
            Err(error) => return error.into_status(),
        };
        let handle =
            adapter_state::<P>().insert_float_tensor(DeviceHandle(lhs_state.device_handle), out);

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn release_tensor<P: FloatTensorPlugin>(tensor: TensorHandle) -> PluginStatus {
    with_boundary(|| {
        adapter_state::<P>().release_tensor(tensor);
        ok()
    })
}

unsafe extern "C" fn release_f32_buffer(buffer: OwnedF32Buffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn release_usize_buffer(buffer: OwnedUsizeBuffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn release_byte_buffer(buffer: OwnedByteBuffer) -> PluginStatus {
    with_boundary(|| {
        if !buffer.ptr.is_null() {
            unsafe {
                let _ = Vec::from_raw_parts(buffer.ptr, buffer.len, buffer.len);
            }
        }
        ok()
    })
}

unsafe extern "C" fn dense_tensor_from_data<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    device: DeviceHandle,
    data: DenseTensorDataRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let data = match try_dense_data(data) {
            Ok(data) => data,
            Err(status) => return status,
        };

        let handle = match kind {
            DenseTensorKind::Float => match P::dense_float_from_data(&device_state, data) {
                Ok(tensor) => adapter_state::<P>().insert_float_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Int => match P::dense_int_from_data(&device_state, data) {
                Ok(tensor) => adapter_state::<P>().insert_int_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Bool => match P::dense_bool_from_data(&device_state, data) {
                Ok(tensor) => adapter_state::<P>().insert_bool_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_into_data<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    out_data: *mut OwnedDenseTensorData,
) -> PluginStatus {
    with_boundary(|| {
        if out_data.is_null() {
            return invalid_argument();
        }

        let data = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::dense_float_into_data(&state.tensor) {
                    Ok(data) => data,
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::dense_int_into_data(&state.tensor) {
                    Ok(data) => data,
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::dense_bool_into_data(&state.tensor) {
                    Ok(data) => data,
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };

        write_dense_tensor_data(out_data, data);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_empty<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let handle = match kind {
            DenseTensorKind::Float => match P::float_empty(&device_state, &shape, dtype) {
                Ok(tensor) => adapter_state::<P>().insert_float_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Int => match P::int_empty(&device_state, &shape, dtype) {
                Ok(tensor) => adapter_state::<P>().insert_int_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Bool => match P::bool_empty(&device_state, &shape, dtype) {
                Ok(tensor) => adapter_state::<P>().insert_bool_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_full<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    value: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let handle = match kind {
            DenseTensorKind::Float => match P::float_full(&device_state, &shape, dtype, value) {
                Ok(tensor) => adapter_state::<P>().insert_float_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Int => match P::int_full(&device_state, &shape, dtype, value) {
                Ok(tensor) => adapter_state::<P>().insert_int_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Bool => match P::bool_full(&device_state, &shape, dtype, value) {
                Ok(tensor) => adapter_state::<P>().insert_bool_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_random<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    distribution: DenseDistribution,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let device_state = match adapter_state::<P>().lookup_device(device) {
            Ok(device_state) => device_state,
            Err(status) => return status,
        };
        let shape = match try_shape(shape) {
            Ok(shape) => shape,
            Err(status) => return status,
        };

        let handle = match kind {
            DenseTensorKind::Float => {
                match P::float_random(&device_state, &shape, dtype, distribution) {
                    Ok(tensor) => adapter_state::<P>().insert_float_tensor(device, tensor),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => match P::int_random(&device_state, &shape, dtype, distribution)
            {
                Ok(tensor) => adapter_state::<P>().insert_int_tensor(device, tensor),
                Err(error) => return error.into_status(),
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_unary<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorUnaryOp,
    tensor: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_unary(op, &state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_unary(op, &state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_unary(op, &state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_binary<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorBinaryOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => {
                let lhs = match adapter_state::<P>().lookup_float_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_float_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::float_binary(op, &lhs.tensor, &rhs.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let lhs = match adapter_state::<P>().lookup_int_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_int_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::int_binary(op, &lhs.tensor, &rhs.tensor) {
                    Ok(out) => {
                        adapter_state::<P>().insert_int_tensor(DeviceHandle(lhs.device_handle), out)
                    }
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let lhs = match adapter_state::<P>().lookup_bool_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_bool_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::bool_binary(op, &lhs.tensor, &rhs.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_scalar<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorScalarOp,
    tensor: TensorHandle,
    scalar: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_scalar(op, &state.tensor, scalar) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_scalar(op, &state.tensor, scalar) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_comparison<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorComparisonOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => {
                let lhs = match adapter_state::<P>().lookup_float_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_float_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::float_comparison(op, &lhs.tensor, &rhs.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let lhs = match adapter_state::<P>().lookup_int_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_int_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::int_comparison(op, &lhs.tensor, &rhs.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let lhs = match adapter_state::<P>().lookup_bool_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_bool_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::bool_comparison(op, &lhs.tensor, &rhs.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_comparison_scalar<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorComparisonOp,
    tensor: TensorHandle,
    scalar: DenseScalarValue,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_comparison_scalar(op, &state.tensor, scalar, out_dtype)
                {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_comparison_scalar(op, &state.tensor, scalar, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => {
                    match P::bool_comparison_scalar(op, &state.tensor, scalar, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    }
                }
                Err(status) => return status,
            },
        };

        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_reduce<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorReduceOp,
    tensor: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_reduce(op, &state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_reduce(op, &state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_reduce_dim<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorReduceDimOp,
    tensor: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_reduce_dim(op, &state.tensor, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_reduce_dim(op, &state.tensor, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_predicate_reduce<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorPredicateReduceOp,
    tensor: TensorHandle,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_predicate_reduce(op, &state.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_predicate_reduce(op, &state.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_predicate_reduce(op, &state.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_predicate_reduce_dim<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorPredicateReduceOp,
    tensor: TensorHandle,
    dim: usize,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_predicate_reduce_dim(op, &state.tensor, dim, out_dtype)
                {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_predicate_reduce_dim(op, &state.tensor, dim, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => {
                    match P::bool_predicate_reduce_dim(op, &state.tensor, dim, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    }
                }
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_arg<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorArgOp,
    tensor: TensorHandle,
    dim: usize,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_arg(op, &state.tensor, dim, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_arg(op, &state.tensor, dim, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_transform<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorTransformOp,
    tensor: TensorHandle,
    args: DenseTensorTransformArgs,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let args = match try_transform_args(args) {
            Ok(args) => args,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_transform(op, &state.tensor, &args) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_transform(op, &state.tensor, &args) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_transform(op, &state.tensor, &args) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_slice<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    slices: DenseTensorSlicesRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_slice(&state.tensor, &slices) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_slice(&state.tensor, &slices) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_slice(&state.tensor, &slices) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_slice_assign<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    slices: DenseTensorSlicesRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let slices = match try_slices(slices) {
            Ok(slices) => slices,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => {
                let tensor_state = match adapter_state::<P>().lookup_float_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_float_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::float_slice_assign(&tensor_state.tensor, &slices, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let tensor_state = match adapter_state::<P>().lookup_int_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_int_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::int_slice_assign(&tensor_state.tensor, &slices, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let tensor_state = match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_bool_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::bool_slice_assign(&tensor_state.tensor, &slices, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_gather<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let indices = match adapter_state::<P>().lookup_int_tensor(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_gather(dim, &state.tensor, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_gather(dim, &state.tensor, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_gather(dim, &state.tensor, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_scatter<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorScatterOp,
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let indices = match adapter_state::<P>().lookup_int_tensor(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => {
                let tensor_state = match adapter_state::<P>().lookup_float_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_float_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::float_scatter(
                    op,
                    dim,
                    &tensor_state.tensor,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let tensor_state = match adapter_state::<P>().lookup_int_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_int_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::int_scatter(
                    op,
                    dim,
                    &tensor_state.tensor,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let tensor_state = match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_bool_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::bool_scatter(
                    op,
                    dim,
                    &tensor_state.tensor,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_select<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let indices = match adapter_state::<P>().lookup_int_tensor(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_select(&state.tensor, dim, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_select(&state.tensor, dim, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_select(&state.tensor, dim, &indices.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_select_assign<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorSelectAssignOp,
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let indices = match adapter_state::<P>().lookup_int_tensor(indices) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => {
                let tensor_state = match adapter_state::<P>().lookup_float_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_float_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::float_select_assign(
                    op,
                    &tensor_state.tensor,
                    dim,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let tensor_state = match adapter_state::<P>().lookup_int_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_int_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::int_select_assign(
                    op,
                    &tensor_state.tensor,
                    dim,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let tensor_state = match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_bool_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::bool_select_assign(
                    op,
                    &tensor_state.tensor,
                    dim,
                    &indices.tensor,
                    &value_state.tensor,
                ) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_mask_where<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let mask = match adapter_state::<P>().lookup_bool_tensor(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => {
                let tensor_state = match adapter_state::<P>().lookup_float_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_float_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::float_mask_where(&tensor_state.tensor, &mask.tensor, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let tensor_state = match adapter_state::<P>().lookup_int_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_int_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::int_mask_where(&tensor_state.tensor, &mask.tensor, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let tensor_state = match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let value_state = match adapter_state::<P>().lookup_bool_tensor(value) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                match P::bool_mask_where(&tensor_state.tensor, &mask.tensor, &value_state.tensor) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(tensor_state.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_mask_fill<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    mask: TensorHandle,
    value: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let mask = match adapter_state::<P>().lookup_bool_tensor(mask) {
            Ok(state) => state,
            Err(status) => return status,
        };
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_mask_fill(&state.tensor, &mask.tensor, value) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_mask_fill(&state.tensor, &mask.tensor, value) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => match adapter_state::<P>().lookup_bool_tensor(tensor) {
                Ok(state) => match P::bool_mask_fill(&state.tensor, &mask.tensor, value) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_cat<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensors: DenseTensorHandleListRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handles = match try_tensor_handle_list(tensors) {
            Ok(handles) => handles,
            Err(status) => return status,
        };
        if handles.is_empty() {
            return invalid_argument();
        }

        let handle = match kind {
            DenseTensorKind::Float => {
                let mut tensors = Vec::with_capacity(handles.len());
                let mut device_handle = None;
                for handle in handles {
                    let state = match adapter_state::<P>().lookup_float_tensor(handle) {
                        Ok(state) => state,
                        Err(status) => return status,
                    };
                    device_handle.get_or_insert(state.device_handle);
                    if device_handle != Some(state.device_handle) {
                        return invalid_argument();
                    }
                    tensors.push(state.tensor);
                }
                match P::float_cat(&tensors, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(device_handle.unwrap()), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int => {
                let mut tensors = Vec::with_capacity(handles.len());
                let mut device_handle = None;
                for handle in handles {
                    let state = match adapter_state::<P>().lookup_int_tensor(handle) {
                        Ok(state) => state,
                        Err(status) => return status,
                    };
                    device_handle.get_or_insert(state.device_handle);
                    if device_handle != Some(state.device_handle) {
                        return invalid_argument();
                    }
                    tensors.push(state.tensor);
                }
                match P::int_cat(&tensors, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(device_handle.unwrap()), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Bool => {
                let mut tensors = Vec::with_capacity(handles.len());
                let mut device_handle = None;
                for handle in handles {
                    let state = match adapter_state::<P>().lookup_bool_tensor(handle) {
                        Ok(state) => state,
                        Err(status) => return status,
                    };
                    device_handle.get_or_insert(state.device_handle);
                    if device_handle != Some(state.device_handle) {
                        return invalid_argument();
                    }
                    tensors.push(state.tensor);
                }
                match P::bool_cat(&tensors, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_bool_tensor(DeviceHandle(device_handle.unwrap()), out),
                    Err(error) => return error.into_status(),
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_cast<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_cast(&state.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_cast(&state.tensor, out_dtype) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_convert<P: FloatTensorPlugin>(
    op: DenseTensorConvertOp,
    tensor: TensorHandle,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match op {
            DenseTensorConvertOp::FloatIntoInt => {
                match adapter_state::<P>().lookup_float_tensor(tensor) {
                    Ok(state) => match P::float_into_int(&state.tensor, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_int_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    },
                    Err(status) => return status,
                }
            }
            DenseTensorConvertOp::IntIntoFloat => {
                match adapter_state::<P>().lookup_int_tensor(tensor) {
                    Ok(state) => match P::int_into_float(&state.tensor, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_float_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    },
                    Err(status) => return status,
                }
            }
            DenseTensorConvertOp::BoolIntoInt => {
                match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => match P::bool_into_int(&state.tensor, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_int_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    },
                    Err(status) => return status,
                }
            }
            DenseTensorConvertOp::BoolIntoFloat => {
                match adapter_state::<P>().lookup_bool_tensor(tensor) {
                    Ok(state) => match P::bool_into_float(&state.tensor, out_dtype) {
                        Ok(out) => adapter_state::<P>()
                            .insert_float_tensor(DeviceHandle(state.device_handle), out),
                        Err(error) => return error.into_status(),
                    },
                    Err(status) => return status,
                }
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_binary_dim<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    op: DenseTensorBinaryDimOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => {
                let lhs = match adapter_state::<P>().lookup_float_tensor(lhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                let rhs = match adapter_state::<P>().lookup_float_tensor(rhs) {
                    Ok(state) => state,
                    Err(status) => return status,
                };
                if lhs.device_handle != rhs.device_handle {
                    return invalid_argument();
                }
                match P::float_binary_dim(op, &lhs.tensor, &rhs.tensor, dim) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(lhs.device_handle), out),
                    Err(error) => return error.into_status(),
                }
            }
            DenseTensorKind::Int | DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_sort<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_tensor: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_tensor.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_sort(&state.tensor, dim, descending) {
                    Ok(out) => adapter_state::<P>()
                        .insert_float_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_sort(&state.tensor, dim, descending) {
                    Ok(out) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), out),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_tensor, handle);
        ok()
    })
}

unsafe extern "C" fn dense_tensor_sort_with_indices<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_values: *mut TensorHandle,
    out_indices: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_values.is_null() || out_indices.is_null() {
            return invalid_argument();
        }
        match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_sort_with_indices(&state.tensor, dim, descending) {
                    Ok((values, indices)) => {
                        write_tensor(
                            out_values,
                            adapter_state::<P>()
                                .insert_float_tensor(DeviceHandle(state.device_handle), values),
                        );
                        write_tensor(
                            out_indices,
                            adapter_state::<P>()
                                .insert_int_tensor(DeviceHandle(state.device_handle), indices),
                        );
                        ok()
                    }
                    Err(error) => error.into_status(),
                },
                Err(status) => status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_sort_with_indices(&state.tensor, dim, descending) {
                    Ok((values, indices)) => {
                        write_tensor(
                            out_values,
                            adapter_state::<P>()
                                .insert_int_tensor(DeviceHandle(state.device_handle), values),
                        );
                        write_tensor(
                            out_indices,
                            adapter_state::<P>()
                                .insert_int_tensor(DeviceHandle(state.device_handle), indices),
                        );
                        ok()
                    }
                    Err(error) => error.into_status(),
                },
                Err(status) => status,
            },
            DenseTensorKind::Bool => PluginError::unsupported(ERR_UNSUPPORTED).into_status(),
        }
    })
}

unsafe extern "C" fn dense_tensor_argsort<P: FloatTensorPlugin>(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_dtype: DenseTensorDType,
    out_indices: *mut TensorHandle,
) -> PluginStatus {
    with_boundary(|| {
        if out_indices.is_null() {
            return invalid_argument();
        }
        let handle = match kind {
            DenseTensorKind::Float => match adapter_state::<P>().lookup_float_tensor(tensor) {
                Ok(state) => match P::float_argsort(&state.tensor, dim, descending, out_dtype) {
                    Ok(indices) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), indices),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Int => match adapter_state::<P>().lookup_int_tensor(tensor) {
                Ok(state) => match P::int_argsort(&state.tensor, dim, descending, out_dtype) {
                    Ok(indices) => adapter_state::<P>()
                        .insert_int_tensor(DeviceHandle(state.device_handle), indices),
                    Err(error) => return error.into_status(),
                },
                Err(status) => return status,
            },
            DenseTensorKind::Bool => {
                return PluginError::unsupported(ERR_UNSUPPORTED).into_status();
            }
        };
        write_tensor(out_indices, handle);
        ok()
    })
}

/// Builds the metadata table for a trait-backed plugin implementation.
pub const fn backend_plugin_v1<P: FloatTensorPlugin>() -> BackendPluginV1 {
    BackendPluginV1 {
        abi_version: BACKEND_PLUGIN_ABI_VERSION,
        backend_name: backend_name::<P>,
        seed: seed::<P>,
        sync: sync::<P>,
        device_count: device_count::<P>,
    }
}

/// Builds the tensor operation table for a trait-backed plugin implementation.
pub const fn backend_tensor_ops_v1<P: FloatTensorPlugin>() -> BackendTensorOpsV1 {
    BackendTensorOpsV1 {
        abi_version: BACKEND_TENSOR_OPS_ABI_VERSION,
        create_device: create_device::<P>,
        release_device: release_device::<P>,
        tensor_from_f32_data: tensor_from_f32_data::<P>,
        tensor_into_f32_data: tensor_into_f32_data::<P>,
        tensor_shape: tensor_shape::<P>,
        tensor_binary: tensor_binary::<P>,
        release_tensor: release_tensor::<P>,
        release_f32_buffer,
        release_usize_buffer,
        release_byte_buffer,
        dense_tensor_from_data: dense_tensor_from_data::<P>,
        dense_tensor_into_data: dense_tensor_into_data::<P>,
        dense_tensor_empty: dense_tensor_empty::<P>,
        dense_tensor_full: dense_tensor_full::<P>,
        dense_tensor_random: dense_tensor_random::<P>,
        dense_tensor_unary: dense_tensor_unary::<P>,
        dense_tensor_binary: dense_tensor_binary::<P>,
        dense_tensor_scalar: dense_tensor_scalar::<P>,
        dense_tensor_comparison: dense_tensor_comparison::<P>,
        dense_tensor_comparison_scalar: dense_tensor_comparison_scalar::<P>,
        dense_tensor_reduce: dense_tensor_reduce::<P>,
        dense_tensor_reduce_dim: dense_tensor_reduce_dim::<P>,
        dense_tensor_predicate_reduce: dense_tensor_predicate_reduce::<P>,
        dense_tensor_predicate_reduce_dim: dense_tensor_predicate_reduce_dim::<P>,
        dense_tensor_arg: dense_tensor_arg::<P>,
        dense_tensor_transform: dense_tensor_transform::<P>,
        dense_tensor_slice: dense_tensor_slice::<P>,
        dense_tensor_slice_assign: dense_tensor_slice_assign::<P>,
        dense_tensor_gather: dense_tensor_gather::<P>,
        dense_tensor_scatter: dense_tensor_scatter::<P>,
        dense_tensor_select: dense_tensor_select::<P>,
        dense_tensor_select_assign: dense_tensor_select_assign::<P>,
        dense_tensor_mask_where: dense_tensor_mask_where::<P>,
        dense_tensor_mask_fill: dense_tensor_mask_fill::<P>,
        dense_tensor_cat: dense_tensor_cat::<P>,
        dense_tensor_cast: dense_tensor_cast::<P>,
        dense_tensor_convert: dense_tensor_convert::<P>,
        dense_tensor_binary_dim: dense_tensor_binary_dim::<P>,
        dense_tensor_sort: dense_tensor_sort::<P>,
        dense_tensor_sort_with_indices: dense_tensor_sort_with_indices::<P>,
        dense_tensor_argsort: dense_tensor_argsort::<P>,
    }
}

/// Clears the adapter state for a plugin implementation.
///
/// This is primarily intended for tests.
#[doc(hidden)]
pub fn reset_state<P: FloatTensorPlugin>() {
    adapter_state::<P>().clear();
}
