# Learner Architecture

The learner stack starts after raw self-play chunks already exist.

Its job is not to run search. Its job is to:

- read raw chunk files,
- replay each stored root through chess state,
- encode model inputs and dense targets,
- build replay minibatches,
- run SGD,
- persist checkpoints and snapshots.

## Mental Model

The learner has five layers:

- Raw storage facade
  - parses chunk bytes into learner-owned semantic game objects.
- Raw-game encoder
  - replays one stored game into encoded inputs and targets.
- Python API boundary
  - exposes the native learner and self-play entrypoints to Python.
- Training core
  - replay, SGD, and checkpoint handling.
- Iteration driver
  - builds replay for one training iteration and delegates optimization and
    checkpointing to the training core.

## End-to-End Flow

For one training iteration, the learner path is:

```text
chunk paths
  -> chunk parser
  -> semantic game objects
  -> sample encoder
  -> replay buffer
  -> SGD/update step
  -> latest.pt + snapshot
```

What matters most is the semantic boundary:

- raw chunk files preserve game meaning,
- sample encoding reconstructs tensor meaning,
- replay owns learner-ready sample arrays,
- training owns model and optimizer state,
- runner owns per-iteration replay construction only.

## Ownership Boundaries

- Learner raw types
  - learner-owned semantic mirror of raw chunk contents.
  - These types are storage-facing, not model-facing.
- Storage facade
  - owns file I/O and binary parsing handoff from raw chunk code.
- Sample encoder
  - owns replaying chess state from the stored root plus selected moves.
  - owns conversion from sparse legal-move policy to the model-facing dense
    policy target.
- Replay buffer
  - owns accumulated sample arrays and minibatch sampling RNG.
- Training core
  - owns model state, optimizer state, global learner step, and checkpoint
    writes.
- Iteration runner
  - owns shape validation, checkpoint restore, replay reconstruction, and
    optional learner-side W&B logging.

## What Learner Does Not Own

The learner deliberately does not own:

- self-play search state,
- raw chunk binary layout,
- artifact export,
- iteration numbering,
- self-play output directories,
- orchestrator-level resume policy.

Those belong to `actor/mcts/`, `artifacts.py`, and `orchestrator.py`.

The learner-facing raw game representation is the stable semantic handoff
between raw storage and the rest of the learner stack.
