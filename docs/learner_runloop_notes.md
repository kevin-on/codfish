# Learner Runloop Notes

This codebase currently has one orchestrated learner loop:
`python/codfish/orchestrator.py`.

Before changing it, the important points are:

## Runtime Role

The orchestrator is the owner of iteration lifecycle.

It owns:

- bootstrap checkpoint creation,
- bootstrap artifact creation,
- artifact selection for self-play,
- self-play output directory lifecycle,
- learner runner creation per iteration,
- artifact export after training,
- resume and recovery policy.

The learner runner is only one iteration of training work inside that larger
loop.

## Iteration Numbering

The artifact and self-play numbering contract is:

- artifact iteration `N`
  - is the model used to generate self-play iteration `N + 1`
- self-play iteration `N`
  - feeds learner iteration `N`

So a fresh run starts from:

- a bootstrap artifact for iteration zero,
- self-play output for iteration one,
- learner iteration `1`

## Bootstrap Path

Fresh runs need both:

- an initial learner checkpoint,
- a bootstrap artifact.

If neither exists, the orchestrator creates both from a newly initialized model.
If one exists, the other must be reconstructed or validated to match the same
model spec and iteration `0`.

Iteration `1` is a special case:

- the runner is created with `resume=False`,
- then the orchestrator restores bootstrap weights into that fresh runner
  before training.

## Resume Path

Resume starts from `learner/latest.pt`.

The contract is:

- next iteration is `latest.iteration + 1`,
- the previous artifact is expected for the previous learner iteration,
- if that artifact is missing, it may be regenerated from `latest.pt`,
- if it exists, it must match the current model spec and expected iteration.

Historical self-play directories for earlier iterations must also exist. Missing
history is treated as an error, not silently skipped.

## Failure Handling

Self-play output uses a `.partial` directory and rename-on-success pattern.

That gives two important behaviors:

- if self-play fails, the partial directory is deleted,
- if learner training fails after self-play already succeeded, the completed
  self-play iteration directory remains and is reused on retry.

The orchestrator applies the same idea to artifact export through the helper in
`artifacts.py`.

## W&B Ownership

Orchestrated W&B is not the same as standalone learner W&B.

The orchestrator:

- creates one W&B run for the whole outer loop,
- logs once per completed learner iteration,
- restores the persisted run id on resume,
- persists the run identity into learner checkpoints so resume can reconnect to
  the same run.

That is why orchestrated learner runs must not let the learner own W&B
lifecycle directly.

## Current Limits

The current loop intentionally stays simple:

- it rebuilds replay from chunk files every iteration,
- it creates a fresh learner runner every iteration,
- it requires CUDA for orchestrated AOTI self-play,
- it has no separate artifact registry beyond the run-root directory layout.
