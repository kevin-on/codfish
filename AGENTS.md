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

Read these before diving into `src/learner/` or `python/codfish/learner/`:

- `docs/learner_architecture.md`
  - Learner subsystem shape, boundaries, and ownership split.
- `docs/learner_contracts.md`
  - Non-obvious learner invariants across storage, encoding, replay, and checkpoints.
- `docs/learner_encoding_and_replay.md`
  - How raw chunk data becomes dense learner tensors and how replay is rebuilt.
- `docs/learner_runloop_notes.md`
  - Orchestrated learner lifecycle, bootstrap/resume rules, and failure handling.

Read this before touching `python/codfish/orchestrator.py`, `python/codfish/artifacts.py`,
or other cross-system flow:

- `docs/project_architecture.md`
  - End-to-end project loop from checkpoint to artifact to self-play to learner and back.

## Test Execution

- Use `ctest --test-dir build -LE 'slow|smoke' --output-on-failure` as the default local verification loop.
- Run `ctest --test-dir build -L smoke --output-on-failure` when touching search coordinator or other cross-runtime wiring.
- Run `ctest --test-dir build -L slow --output-on-failure` only when touching chess move generation, legality, or FEN parsing code.
