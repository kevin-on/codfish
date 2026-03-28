# Learner Encoding And Replay

The learner does not consume raw chunks directly as minibatches.

It first reconstructs each stored root into current chess state and only then
builds dense arrays.

## Raw To Encoded Samples

For one stored game, the encoder does this:

1. load the stored root state,
2. for each stored ply:
   - take the current root position,
   - generate legal moves,
   - parse and validate the stored selected move,
   - parse and validate every stored policy move,
   - encode the current history with the current input encoder,
   - scatter sparse policy entries into the current dense policy target,
   - derive one-hot WDL target from final game result and root side to move,
   - append the selected move to history.

This is why raw storage can stay stable across encoder revisions. The learner
recomputes tensors late.

## Why Absolute UCI Still Works

The runtime writes absolute UCI, including black moves such as `e7e5`.

The learner replays positions through the chess layer, whose internal board is
side-to-move oriented. That parser handles the orientation conversion, so the
learner can parse absolute UCI strings against both white-to-move and
black-to-move roots without extra caller-side transforms.

## Feature Encoding Boundary

Input encoding is delegated to the engine-side encoder.

That means the learner sample encoder does not define feature layout itself.
Changing the encoder changes learner tensors, but not raw chunk format.

## Dense Policy Construction

Stored policy is keyed by legal move UCI strings.

The learner turns that into dense policy targets by:

- parsing each move back into a legal chess move,
- mapping that move into the fixed policy head,
- writing the stored probability into that slot.

Everything not present in the sparse policy stays zero.

## Replay Buffer Behavior

Replay stores three concatenated arrays:

- model inputs
- policy targets
- WDL targets

It does not keep per-game semantic objects after ingestion.

Practical behavior:

- chunk ingestion stages all encoded samples first,
- any malformed chunk aborts the whole ingest without partial mutation,
- capacity trimming is FIFO at the sample level,
- minibatch sampling uses replacement,
- update count is driven by replay ratio and newly added sample count.

## LearnerRunner Replay Policy

The learner runner rebuilds replay from chunk paths on every call.

It intentionally does not hold a persistent replay buffer across iterations.
Instead it:

- counts samples in the new chunk set,
- reserves capacity for those new samples,
- selects the newest historical chunk tail that fits the remaining sample
  budget,
- ingests history first, then current-iteration chunks.

That is the current replay policy contract used by orchestrated training.
