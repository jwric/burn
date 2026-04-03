mod bool_tensor;
mod int_tensor;
mod module;
mod stubs;
mod tensor;

fn unsupported_op(name: &str) -> ! {
    panic!(
        "Dylib operation '{}' is not implemented for this tensor family.",
        name
    )
}
