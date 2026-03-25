# MCTS Searcher Notes

This codebase currently has one concrete searcher: `GumbelMCTS`.

Before reading the implementation, the important points are:

## Runtime Role

`GumbelMCTS` is the concrete `MCTSSearcher` used by the runtime. It owns:

- the current root history,
- the current search tree,
- the request/reply pattern for model evaluation.

The runtime does not know those internals. It only drives the coroutine and later calls `CommitMove()`.

## Ownership Model

Game progression happens through the searcher, not around it.

In practice this means:

- the searcher is the source of truth for the current position
- `GameRunner` advances the game only by calling `CommitMove()`
- subtree reuse is a local searcher concern, not a runtime concern

## Algorithm Shape

The implementation is best understood as "Gumbel at the root, simpler search below the root."

That is close enough to the paper to explain the design, but not a claim of paper-faithful reproduction.

## Deliberate Simplifications

The main implementation simplifications worth knowing up front are:

- non-root selection remains AlphaZero-style PUCT
- sequential halving budget accounting is simplified
- runtime contract fidelity takes priority over exact paper fidelity

If you want to make the implementation closer to the paper, start from those choices.

## What Must Stay Stable

When changing the searcher, preserve the runtime-facing semantics:

- yielded requests and expected replies must stay matched
- terminal vs non-terminal `SearchResult` meaning must stay intact
- `CommitMove()` must keep the internal root consistent with the accepted move

Everything else is an implementation detail.
