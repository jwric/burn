use burn_backend::ops::TransactionOps;

use super::super::backend::Dylib;

impl<E: Send + Sync + 'static> TransactionOps<Dylib<E>> for Dylib<E> {}
