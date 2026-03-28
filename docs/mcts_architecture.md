# MCTS Runtime Architecture

This directory implements self-play and match-evaluation runtimes around one
idea: split the system by ownership boundaries, not by algorithm details.

If you are new here, read:

1. `docs/mcts_architecture.md`
2. `docs/mcts_contracts.md`
3. `docs/mcts_searcher_notes.md` when touching search
4. `docs/mcts_storage.md` when touching output or replay

## Mental Model

The self-play path has five long-lived pieces:

- `SearchCoordinator`
  - Assembles the system, owns the queues, starts and stops threads, and seeds initial tasks.
- `WorkerRuntime`
  - Runs search coroutines until they either need model evaluation or finish a root search.
- `InferenceRuntime`
  - Batches evaluation items, encodes features, calls the backend, and returns replies to waiting tasks.
- `GameRunner`
  - Turns a completed root search into either the next ply of the same game or a completed game payload.
- `ChunkWriterRuntime`
  - Optionally persists completed games as raw chunk files.

The hot path is direct queue handoff between these components. The coordinator owns the plumbing but is not on the data path once the run starts.

The match-eval path reuses the same worker/inference split with different
output ownership:

- `MatchCoordinator`
  - Owns lifecycle and bounded match assembly for head-to-head evaluation.
- `MatchGameRunner`
  - Turns a completed root search into either the next ply of the same match
    game or a PGN-ready terminal payload.
- `InferenceRuntime`
  - Can serve one backend or multiple backends, depending on what the task type
    asks it to route.

## End-to-End Flow

At a high level:

```text
GameTaskFactory
  -> ready_queue
  -> workers
  -> inference
  -> workers
  -> game runner
  -> workers again or raw storage
```

For match evaluation, the tail changes but the ownership model stays the same:

```text
GameTaskFactory
  -> ready_queue
  -> workers
  -> inference
  -> workers
  -> match game runner
  -> workers again or PGN output
```

What matters is not the exact loop structure, but the ownership model:

- the same `GameTask` survives across the whole game
- the searcher owns the current game position
- the runtime moves envelopes, not chess state

## Design Intent

These boundaries are the main reason the code is shaped the way it is:

- Workers own coroutine execution.
- Inference owns batching and backend calls.
- Game runner owns post-search game progression.
- Writer owns persistence only.
- Coordinator owns lifecycle and wiring only.

Runtime-specific task metadata may change, but those ownership boundaries should
not.

When reading or changing the code, preserve those boundaries first.

## Current Limits

The current implementation intentionally stays simple:

- `Stop()` is not a graceful full drain.
- Completed self-play games are not automatically replaced with new ones.
- Self-play raw output remains the only training-data persistence path.
- Match evaluation is currently a bounded two-player run with PGN output; any
  rating aggregation lives above this layer.

Those are current system properties, not accidental omissions in one file.
