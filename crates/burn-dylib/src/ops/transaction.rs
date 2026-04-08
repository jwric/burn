use burn_backend::{
    ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};

use super::super::backend::Dylib;
use super::super::runtime;

impl<E: Send + Sync + 'static> TransactionOps<Dylib<E>> for Dylib<E> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Dylib<E>>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
        runtime::transaction_execute(
            transaction.read_floats,
            transaction.read_qfloats,
            transaction.read_ints,
            transaction.read_bools,
        )
        .map_err(runtime::to_execution_error)
    }
}
