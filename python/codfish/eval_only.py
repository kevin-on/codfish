from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .orchestrator import EvalConfig, run_eval_only


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codfish eval-only replay")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--eval-snapshot-interval", required=True, type=int)
    parser.add_argument("--eval-match-games", default=200, type=int)
    parser.add_argument("--eval-num-workers", default=1, type=int)
    parser.add_argument("--eval-num-action", default=8, type=int)
    parser.add_argument("--eval-num-simulation", default=16, type=int)
    parser.add_argument("--eval-c-puct", default=1.0, type=float)
    parser.add_argument("--eval-c-visit", default=1.0, type=float)
    parser.add_argument("--eval-c-scale", default=1.0, type=float)
    parser.add_argument("--eval-window-offset", nargs="+", type=int, default=[1, 3, 10])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_config = EvalConfig(
        snapshot_interval=args.eval_snapshot_interval,
        match_games=args.eval_match_games,
        num_workers=args.eval_num_workers,
        num_action=args.eval_num_action,
        num_simulation=args.eval_num_simulation,
        c_puct=args.eval_c_puct,
        c_visit=args.eval_c_visit,
        c_scale=args.eval_c_scale,
        window_offsets=tuple(args.eval_window_offset),
    )

    try:
        result = run_eval_only(run_root=Path(args.run_root), eval_config=eval_config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if result is not None:
        print(result.ratings_csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
