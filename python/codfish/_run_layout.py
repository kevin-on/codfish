"""Canonical run-root layout.

<run_root>/
  learner/
    latest.pt
    previous.pt
    initial.pt
    snapshots/
  artifacts/
    iter_000000/
    iter_000001/
  selfplay/
    iter_000001/
    iter_000002/
  eval/
    matches/
      iter_000010_step_000000281__iter_000020_step_000000579.pgn
    ordo/
      all_games.pgn
      ratings.txt
      ratings.csv
"""

from __future__ import annotations

import os
from pathlib import Path


def learner_dir(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / "learner"


def artifacts_dir(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / "artifacts"


def selfplay_dir(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / "selfplay"


def eval_dir(run_root: str | os.PathLike[str]) -> Path:
    return Path(run_root) / "eval"


def eval_matches_dir(run_root: str | os.PathLike[str]) -> Path:
    return eval_dir(run_root) / "matches"


def eval_ordo_dir(run_root: str | os.PathLike[str]) -> Path:
    return eval_dir(run_root) / "ordo"


def latest_checkpoint_path(learner_dir_path: str | os.PathLike[str]) -> Path:
    return Path(learner_dir_path) / "latest.pt"


def previous_checkpoint_path(learner_dir_path: str | os.PathLike[str]) -> Path:
    return Path(learner_dir_path) / "previous.pt"


def initial_checkpoint_path(learner_dir_path: str | os.PathLike[str]) -> Path:
    return Path(learner_dir_path) / "initial.pt"


def snapshot_dir(learner_dir_path: str | os.PathLike[str]) -> Path:
    return Path(learner_dir_path) / "snapshots"


def snapshot_path(
    learner_dir_path: str | os.PathLike[str],
    *,
    iteration: int,
    global_step: int,
) -> Path:
    return snapshot_dir(learner_dir_path) / (
        f"iter_{iteration:06d}_step_{global_step:09d}.pt"
    )


def artifact_iteration_dir(
    artifacts_dir_path: str | os.PathLike[str], iteration: int
) -> Path:
    return Path(artifacts_dir_path) / f"iter_{iteration:06d}"


def selfplay_iteration_dir(
    selfplay_dir_path: str | os.PathLike[str], iteration: int
) -> Path:
    return Path(selfplay_dir_path) / f"iter_{iteration:06d}"


def eval_match_path(
    run_root: str | os.PathLike[str],
    snapshot_stem_a: str,
    snapshot_stem_b: str,
) -> Path:
    lower, upper = sorted((snapshot_stem_a, snapshot_stem_b))
    return eval_matches_dir(run_root) / f"{lower}__{upper}.pgn"


def eval_all_games_pgn_path(run_root: str | os.PathLike[str]) -> Path:
    return eval_ordo_dir(run_root) / "all_games.pgn"


def eval_ratings_text_path(run_root: str | os.PathLike[str]) -> Path:
    return eval_ordo_dir(run_root) / "ratings.txt"


def eval_ratings_csv_path(run_root: str | os.PathLike[str]) -> Path:
    return eval_ordo_dir(run_root) / "ratings.csv"


def partial_path(final_path: str | os.PathLike[str]) -> Path:
    path = Path(final_path)
    return path.with_name(f"{path.name}.partial")
