# Learner Contracts

This document keeps only the non-obvious learner contracts that matter across
files.

## Raw Game Boundary

The stored game object is still semantic storage, not a tensor batch.

Important rules:

- `game_result` must be decided.
- A game with no plies is allowed.
  - It encodes to metadata-only empty buffers.
- A game with plies must have `initial_fen`.
  - Encoding needs a replayable root state.

## Move Semantics

Stored move strings are absolute UCI strings like `e2e4` or `a7a8q`.

The learner contract is:

- parse each stored move against the current replayed root position,
- rely on the chess layer to reinterpret black-to-move positions correctly,
- validate that the stored selected move is legal at that root.

This means the learner does not trust the chunk blindly. It replays and
re-validates legality.

## Policy Target Semantics

Stored policy is sparse over legal moves.

The learner enforces:

- every stored policy move must be legal at the root,
- policy entries must not collide on the dense policy-head index,
- the selected move must appear in the stored policy.

The learner does not reinterpret policy values beyond scattering them into the
current dense policy head.

## WDL Target Semantics

WDL targets are not stored per ply. They are reconstructed from the final game
result and the root side to move.

The key invariant is:

- targets are always from the root position's side-to-move perspective.

So the same final game result may map to different one-hot WDL targets on
different plies of the same game.

## History Reconstruction Boundary

The authoritative replay path during encoding is:

1. start from `initial_fen`,
2. encode the current replayed history,
3. append the stored selected move,
4. continue to the next ply.

Policy moves do not advance history. Only the accepted selected move does.

## Model Shape Boundary

The Python model spec must declare the same input and policy shape as the
native learner side.

If those shapes diverge, runner construction fails before training starts.

## Replay Buffer Boundary

Replay owns learner-ready arrays, not per-game objects.

Important behavior:

- chunk ingestion is atomic across the provided chunk-path list,
- newly ingested samples are appended after existing samples,
- capacity overflow trims the oldest samples,
- minibatch sampling uses a private numpy RNG,
- that RNG state is checkpointed and later restored.

## Checkpoint Boundary

The primary checkpoint is the full training restore point. Snapshot payloads are
lighter-weight model snapshots rather than full optimizer restores.

A full restore also requires trainer-config compatibility.

## Runner vs Orchestrator Boundary

There are two distinct ownership modes:

- Standalone learner mode
  - learner code may own W&B lifecycle itself.
- Orchestrated mode
  - orchestrator owns W&B lifecycle and learner checkpoints only carry the
    identity needed to resume that run.

Keep that boundary stable. The orchestrator depends on it.
