# MCTS Raw Storage

This runtime stores raw game trajectories, not training tensors.

That is the main thing to know before reading the storage code.

## Why Raw Storage Exists

The writer preserves enough information for a later reader to:

- replay the game,
- recover each sampled root,
- re-encode features with the current encoder,
- rebuild policy targets for the current learner.

The storage format is therefore tied to game semantics, not to one version of the training pipeline.

## What Gets Preserved

At the semantic level, storage keeps:

- the played trajectory,
- the search policy over legal moves at each sampled root,
  keyed by standard absolute UCI move strings,
- the final game result,
- the initial position when available.

This is why the runtime writes `TrainingSampleDraft` and `CompletedGame` style payloads instead of dense feature tensors.

## What Storage Does Not Do

The writer does not:

- encode training features
- materialize dense policy heads
- interpret storage as learner-ready minibatches

Those responsibilities belong downstream.

## Practical Consequence

Because storage stays raw:

- encoder changes do not require rewriting old chunks
- storage remains useful across training-pipeline revisions
- readers must still know how to replay chess state and scatter sparse policy entries into the current head layout

The exact binary layout lives in `raw_chunk_format.{h,cc}` and should be treated as code, not as something this document mirrors field-by-field.
