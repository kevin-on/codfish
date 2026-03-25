## Docs Index

Read these before diving into `src/actor/mcts/`:

- `docs/mcts_architecture.md`
  - High-level runtime shape, ownership boundaries, and current system limits.
- `docs/mcts_contracts.md`
  - Non-obvious cross-component contracts and runtime invariants.
- `docs/mcts_searcher_notes.md`
  - Searcher-specific notes, especially for `GumbelMCTS`.
- `docs/mcts_storage.md`
  - Why self-play output is stored as raw game data and what downstream readers are expected to do.

Do not use `docs/chess_engine_dev_spec.md` as an MCTS runtime reference.
