# Project Architecture

This repository is a chess training stack, not just a search runtime and not
just a learner.

The codebase has one main training loop plus one optional evaluation side path:

1. export a model checkpoint into an inference artifact,
2. run native self-play with that artifact,
3. store completed games as raw semantic chunks,
4. rebuild replay from chunk files,
5. train a new checkpoint,
6. export the next artifact,
7. optionally run snapshot-based eval matches and rebuild ratings,
8. repeat.

If you are new here, read:

1. `docs/project_architecture.md`
2. `docs/mcts_architecture.md`
3. `docs/mcts_contracts.md`
4. `docs/learner_architecture.md`
5. `docs/learner_contracts.md`

## Mental Model

The system is easiest to understand as five stacked layers:

- `src/lc0/`
  - Chess rules, move generation, repetition tracking, and policy-head move
    indexing.
- `src/engine/`
  - Feature encoding and the narrow inference-backend interface.
- `src/actor/mcts/`
  - Native self-play and head-to-head match runtime around `MCTSSearcher`,
    inference batching, game progression, raw chunk writing, and match PGN
    output.
- `src/learner/` plus `python/codfish/learner/`
  - Raw chunk reading, raw-game encoding, replay, SGD training, and checkpoint
    handling.
- `python/codfish/artifacts.py` and `python/codfish/orchestrator.py`
  - Checkpoint-to-artifact export, the full self-play/update lifecycle, and
    optional snapshot-based evaluation.

The core design choice is that self-play storage stays game-semantic. The actor
runtime writes replayable games; the learner later reconstructs tensors.

Snapshot-based evaluation is a side path on top of that loop. It compares
selected exported artifacts against earlier ones and writes match and rating
outputs under the run root, but it does not feed replay directly.

## End-to-End Flow

At a high level:

```text
checkpoint
  -> inference artifact
  -> native self-play
  -> raw chunk files
  -> learner replay reconstruction
  -> SGD update
  -> next checkpoint
  -> next artifact
  -> optional eval matches + ratings
```

The important ownership split is:

- outer-loop lifecycle belongs to the orchestrator and artifact code,
- snapshot selection, eval scheduling, and rating rebuild also belong to the
  orchestrator,
- self-play runtime belongs to `actor/mcts`,
- replay reconstruction and SGD belong to the learner,
- chess semantics and input encoding belong to the lower layers.

## Current Limits

The current implementation intentionally stays simple in a few places:

- Native orchestrated self-play requires CUDA because artifact export and AOTI
  inference are CUDA-oriented.
- Replay is rebuilt from chunk files each iteration; there is no long-lived
  serialized replay database.
- Runtime shutdown is not yet a fully graceful end-to-end drain.
- Orchestrated training creates a fresh `LearnerRunner` each iteration and
  restores state through checkpoints, not through long-lived Python objects.
- Raw chunk storage is the only persisted self-play format; learner-ready
  tensors are reconstructed downstream every time.
- Evaluation is a snapshot-window side path, not a general tournament service.
- Rating outputs are rebuilt from accumulated match PGNs under the run root and
  currently depend on an external `ordo` executable.
