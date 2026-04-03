mod stubs;
mod tensor;

fn unsupported_op(name: &str) -> ! {
    panic!(
        "Dylib operation '{}' is not implemented yet. Supported today: from_data, into_data, to_device, add, matmul.",
        name
    )
}
