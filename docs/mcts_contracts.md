# MCTS Runtime Contracts

This document keeps only the non-obvious contracts that matter across files.

## Searcher Boundary

The runtime only relies on two searcher operations:

- `Run()`
  - starts one root search as a request/reply coroutine
- `CommitMove(move)`
  - advances the searcher after a non-terminal decision was accepted

The important invariant is that the searcher, not `GameTask`, owns the authoritative current position. The runtime never progresses board state directly.

## Search Result Semantics

`SearchResult` has two modes:

- Non-terminal
  - `game_result == UNDECIDED`
  - root snapshot, selected move, legal moves, and improved policy are meaningful
- Terminal
  - `game_result != UNDECIDED`
  - the rest of the payload should be treated as invalid

That split is central to the worker/game-runner boundary.

## GameTask Semantics

`GameTask` is a transport envelope with unique ownership at each step.
Its runtime state is about coroutine lifecycle, not game semantics:

- `kNew`
  - a fresh root search has not started
- `kWaitingEval`
  - the coroutine yielded a request and is waiting for a reply
- `kReady`
  - inference attached the reply and the coroutine can resume

The task exists to carry a searcher, a suspended coroutine, and accumulated per-game payload across threads.

## Task Routing Boundary

`GameTask` may also provide per-item backend routing for inference.

The runtime contract is:

- default single-backend tasks route everything to slot `0`,
- match tasks may route white-to-move and black-to-move plies to different
  backend slots,
- every yielded `EvalRequestItem` must still be routable by the active runtime.

## Inference Boundary

Inference batches `EvalRequestItem`s, not whole tasks.

That has two consequences that are easy to miss:

- one large request may be split across multiple backend runs
- leftover items from one task may share a later batch with items from another task
- when multiple backends exist, items from one task may be routed to different
  backend slots before the task becomes ready again

The backend is intentionally narrower than the runtime. It only sees dense model inputs and outputs, never `GameTask` or coroutine state.

## Completion Boundary

Self-play and match evaluation diverge only after a root search completes.

The important split is:

- self-play completion preserves replayable training payload,
- match completion preserves player assignment, move history, and terminal
  result for PGN output,
- both still advance the searcher only through accepted selected moves.

## Training Payload Boundary

`TrainingSampleDraft` and `CompletedGame` preserve game-level meaning, not model-ready tensors.

The design intent is:

- runtime emits replayable search decisions
- offline readers reconstruct features and dense policy targets later

This keeps storage decoupled from the current encoder and learner shape.

## Coordinator Assumptions

`GameTaskFactory` must create tasks that are already valid runtime inputs:

- task is non-null
- `task->searcher` is non-null
- `task->state == kNew`

Shutdown ordering also matters at the contract level: workers stop before inference so new eval requests cannot appear after inference shutdown begins.
