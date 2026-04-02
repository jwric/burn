use crate::{DeviceHandle, OwnedUsizeBuffer, PluginStatus, TensorHandle, TensorShapeRef};

/// Borrowed byte slice passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSliceRef {
    /// Pointer to contiguous bytes.
    pub ptr: *const u8,
    /// Number of bytes.
    pub len: usize,
}

/// Owned byte buffer returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedByteBuffer {
    /// Pointer to contiguous bytes allocated by the plugin.
    pub ptr: *mut u8,
    /// Number of bytes.
    pub len: usize,
}

impl OwnedByteBuffer {
    /// Creates an empty buffer.
    pub const fn empty() -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
        }
    }
}

/// Borrowed dense tensor payload passed from host to plugin.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorDataRef {
    /// Tensor data type.
    pub dtype: DenseTensorDType,
    /// Tensor shape.
    pub shape: TensorShapeRef,
    /// Tensor bytes.
    pub bytes: ByteSliceRef,
}

/// Owned dense tensor payload returned by plugin to host.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OwnedDenseTensorData {
    /// Tensor data type.
    pub dtype: DenseTensorDType,
    /// Tensor shape.
    pub shape: OwnedUsizeBuffer,
    /// Tensor bytes.
    pub bytes: OwnedByteBuffer,
}

impl OwnedDenseTensorData {
    /// Creates an empty dense tensor payload.
    pub const fn empty(dtype: DenseTensorDType) -> Self {
        Self {
            dtype,
            shape: OwnedUsizeBuffer::empty(),
            bytes: OwnedByteBuffer::empty(),
        }
    }
}

/// Dense tensor kind selector used by the generic dense-tensor ABI families.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorKind {
    /// Floating-point tensors.
    Float = 0,
    /// Integer tensors.
    Int = 1,
    /// Boolean tensors.
    Bool = 2,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorDType {
    F64 = 0,
    F32 = 1,
    Flex32 = 2,
    F16 = 3,
    BF16 = 4,
    I64 = 5,
    I32 = 6,
    I16 = 7,
    I8 = 8,
    U64 = 9,
    U32 = 10,
    U16 = 11,
    U8 = 12,
    BoolNative = 13,
    BoolU8 = 14,
    BoolU32 = 15,
}

/// Dense bool output dtype used by comparison and predicate-reduction callbacks.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorBoolDType {
    /// Native boolean storage.
    Native = 0,
    /// `u8` boolean storage.
    U8 = 1,
    /// `u32` boolean storage.
    U32 = 2,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseScalarKind {
    Float = 0,
    Int = 1,
    UInt = 2,
    Bool = 3,
}

/// Scalar payload passed to scalar dense tensor operations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseScalarValue {
    /// Active scalar variant.
    pub kind: DenseScalarKind,
    /// Floating-point payload.
    pub float: f64,
    /// Signed integer payload.
    pub int: i64,
    /// Unsigned integer payload.
    pub uint: u64,
    /// Boolean payload encoded as `0` or `1`.
    pub boolean: u8,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseDistributionKind {
    Default = 0,
    Bernoulli = 1,
    Uniform = 2,
    Normal = 3,
}

/// Random distribution descriptor for dense tensor creation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseDistribution {
    /// Distribution kind.
    pub kind: DenseDistributionKind,
    /// First numeric parameter, such as `low`, `mean`, or probability.
    pub a: f64,
    /// Second numeric parameter, such as `high` or standard deviation.
    pub b: f64,
}

/// Borrowed axis list passed to dense tensor transform callbacks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseAxesRef {
    /// Pointer to axes.
    pub ptr: *const usize,
    /// Number of axes.
    pub len: usize,
}

/// Single slice spec used by dense slicing callbacks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorSlice {
    /// Inclusive start index.
    pub start: isize,
    /// Exclusive end index.
    pub end: isize,
    /// Whether `end` is present.
    pub has_end: u8,
    /// Slice step.
    pub step: isize,
}

/// Borrowed slice spec list passed to dense slicing callbacks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorSlicesRef {
    /// Pointer to slices.
    pub ptr: *const DenseTensorSlice,
    /// Number of slices.
    pub len: usize,
}

/// Borrowed tensor handle list passed to dense concatenation callbacks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorHandleListRef {
    /// Pointer to tensor handles.
    pub ptr: *const TensorHandle,
    /// Number of tensor handles.
    pub len: usize,
}

/// Parameter bundle for dense tensor transform callbacks.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DenseTensorTransformArgs {
    /// Optional shape payload.
    pub shape: TensorShapeRef,
    /// Optional axes payload.
    pub axes: DenseAxesRef,
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

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorUnaryOp {
    Neg = 0,
    Recip = 1,
    Exp = 2,
    Log = 3,
    Log1p = 4,
    Sqrt = 5,
    Abs = 6,
    Cos = 7,
    Sin = 8,
    Tan = 9,
    Cosh = 10,
    Sinh = 11,
    Tanh = 12,
    Acos = 13,
    Acosh = 14,
    Asin = 15,
    Asinh = 16,
    Atan = 17,
    Atanh = 18,
    Round = 19,
    Floor = 20,
    Ceil = 21,
    Trunc = 22,
    Erf = 23,
    Not = 24,
    BitwiseNot = 25,
    Sign = 26,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorBinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Remainder = 4,
    Matmul = 5,
    Powf = 6,
    Atan2 = 7,
    And = 8,
    Or = 9,
    Xor = 10,
    BitwiseAnd = 11,
    BitwiseOr = 12,
    BitwiseXor = 13,
    LeftShift = 14,
    RightShift = 15,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorScalarOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Remainder = 4,
    Powf = 5,
    BitwiseAnd = 6,
    BitwiseOr = 7,
    BitwiseXor = 8,
    LeftShift = 9,
    RightShift = 10,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorComparisonOp {
    Equal = 0,
    NotEqual = 1,
    Greater = 2,
    GreaterEqual = 3,
    Lower = 4,
    LowerEqual = 5,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorReduceOp {
    Sum = 0,
    Prod = 1,
    Mean = 2,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorReduceDimOp {
    Sum = 0,
    Prod = 1,
    Mean = 2,
    Cumsum = 3,
    Cumprod = 4,
    Cummin = 5,
    Cummax = 6,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorPredicateReduceOp {
    Any = 0,
    All = 1,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorArgOp {
    Argmax = 0,
    Argmin = 1,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorTransformOp {
    Reshape = 0,
    SwapDims = 1,
    Permute = 2,
    Flip = 3,
    Expand = 4,
    Unfold = 5,
    RepeatDim = 6,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorScatterOp {
    Add = 0,
    Or = 1,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorSelectAssignOp {
    Add = 0,
    Or = 1,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorConvertOp {
    FloatIntoInt = 0,
    IntIntoFloat = 1,
    BoolIntoInt = 2,
    BoolIntoFloat = 3,
}

#[allow(missing_docs)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseTensorBinaryDimOp {
    Cross = 0,
}

/// Creates a dense tensor from raw host data bytes.
pub type DenseTensorFromDataFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    device: DeviceHandle,
    data: DenseTensorDataRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Materializes a dense tensor as raw host bytes plus shape and dtype metadata.
pub type DenseTensorIntoDataFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    out_data: *mut OwnedDenseTensorData,
) -> PluginStatus;

/// Creates an empty dense tensor.
pub type DenseTensorEmptyFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Creates a full dense tensor.
pub type DenseTensorFullFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    value: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Creates a random dense tensor.
pub type DenseTensorRandomFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    device: DeviceHandle,
    shape: TensorShapeRef,
    dtype: DenseTensorDType,
    distribution: DenseDistribution,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a unary dense tensor operation.
pub type DenseTensorUnaryFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorUnaryOp,
    tensor: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a same-kind binary dense tensor operation.
pub type DenseTensorBinaryFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorBinaryOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense tensor operation with a scalar rhs.
pub type DenseTensorScalarFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorScalarOp,
    tensor: TensorHandle,
    scalar: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs an element-wise dense tensor comparison producing a bool tensor.
pub type DenseTensorComparisonFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorComparisonOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a scalar dense tensor comparison producing a bool tensor.
pub type DenseTensorComparisonScalarFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorComparisonOp,
    tensor: TensorHandle,
    scalar: DenseScalarValue,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense tensor reduction producing a tensor of the same kind.
pub type DenseTensorReduceFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorReduceOp,
    tensor: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dimensional dense tensor reduction producing a tensor of the same kind.
pub type DenseTensorReduceDimFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorReduceDimOp,
    tensor: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a predicate reduction producing a bool tensor.
pub type DenseTensorPredicateReduceFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorPredicateReduceOp,
    tensor: TensorHandle,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dimensional predicate reduction producing a bool tensor.
pub type DenseTensorPredicateReduceDimFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorPredicateReduceOp,
    tensor: TensorHandle,
    dim: usize,
    out_dtype: DenseTensorBoolDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs an arg reduction producing an int tensor.
pub type DenseTensorArgFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorArgOp,
    tensor: TensorHandle,
    dim: usize,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense tensor transform such as reshape or permute.
pub type DenseTensorTransformFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorTransformOp,
    tensor: TensorHandle,
    args: DenseTensorTransformArgs,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense slice operation.
pub type DenseTensorSliceFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    slices: DenseTensorSlicesRef,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense slice assign operation.
pub type DenseTensorSliceAssignFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    slices: DenseTensorSlicesRef,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense gather operation.
pub type DenseTensorGatherFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense scatter operation.
pub type DenseTensorScatterFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorScatterOp,
    dim: usize,
    tensor: TensorHandle,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense select operation.
pub type DenseTensorSelectFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense select-assign operation.
pub type DenseTensorSelectAssignFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorSelectAssignOp,
    tensor: TensorHandle,
    dim: usize,
    indices: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense mask-where operation.
pub type DenseTensorMaskWhereFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    mask: TensorHandle,
    value: TensorHandle,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a dense mask-fill operation.
pub type DenseTensorMaskFillFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    mask: TensorHandle,
    value: DenseScalarValue,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Concatenates dense tensors along a dimension.
pub type DenseTensorCatFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensors: DenseTensorHandleListRef,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Casts a dense tensor within its tensor kind.
pub type DenseTensorCastFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Converts a dense tensor to another tensor kind.
pub type DenseTensorConvertFn = unsafe extern "C" fn(
    op: DenseTensorConvertOp,
    tensor: TensorHandle,
    out_dtype: DenseTensorDType,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Runs a binary dense tensor operation with an extra dimension parameter.
pub type DenseTensorBinaryDimFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    op: DenseTensorBinaryDimOp,
    lhs: TensorHandle,
    rhs: TensorHandle,
    dim: usize,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Sorts a dense tensor along the given dimension.
pub type DenseTensorSortFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_tensor: *mut TensorHandle,
) -> PluginStatus;

/// Sorts a dense tensor and returns both values and indices.
pub type DenseTensorSortWithIndicesFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_values: *mut TensorHandle,
    out_indices: *mut TensorHandle,
) -> PluginStatus;

/// Returns the argsort indices for a dense tensor.
pub type DenseTensorArgsortFn = unsafe extern "C" fn(
    kind: DenseTensorKind,
    tensor: TensorHandle,
    dim: usize,
    descending: bool,
    out_dtype: DenseTensorDType,
    out_indices: *mut TensorHandle,
) -> PluginStatus;

/// Releases a plugin-allocated byte buffer.
pub type ReleaseByteBufferFn = unsafe extern "C" fn(buffer: OwnedByteBuffer) -> PluginStatus;
