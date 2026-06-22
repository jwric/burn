# Recorded graph replay (coarse remote inference)

Goal: run a model on a remote peer without re-streaming its whole op graph each forward pass. Record
the op stream once, then send only inputs and get back outputs.

## Idea

A model forward pass is a sequence of `OperationIr` referencing tensor ids: resident weights
(registered once, kept alive), per-call inputs, and internally produced intermediates/outputs. The
client already emits this stream, including `Drop` ops for intermediates as they go out of scope.

So a *graph* = the recorded op stream for one forward pass, minus the input registrations (those are
the per-call variable), plus the output `TensorIr`s to read back. Weights are resident and referenced
by id.

Replaying the stream verbatim with the same ids is self-cleaning: the recorded `Drop`s free the
per-call intermediates, weights are `ReadOnly` so they persist, and inputs are re-registered each
call. No id remapping and no interpreter changes are needed. Cost: replay is sequential per graph
(template ids are reused), which is exactly the single-client inference case.

## Protocol (`shared/task.rs`)

- `GraphId` — newtype, like `SessionId`.
- `Task::RegisterGraph { stream_id, graph_id, ops, outputs }` — fire-and-forget; store the graph.
- `Task::RunGraph { request_id, stream_id, graph_id, inputs }` — bind inputs, replay ops, read
  outputs, respond.
- `TaskResponseContent::RunGraph(Result<Vec<TensorData>, ExecutionError>)`.

## Server

`SessionHandler` keeps a `graphs: HashMap<GraphId, Graph>`. `RunGraph`: register each input under its
id, replay each op, read each output (with a consuming status so outputs don't linger), respond.

## Client (later)

A recording API that tees the op stream for a closure into a buffer, plus a `RemoteGraph` handle with
record-once / run-many. Not in the first cut.

## Limits

Sequential replay per graph; static graphs only (data-dependent control flow isn't captured).
