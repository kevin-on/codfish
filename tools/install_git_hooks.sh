#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config --local core.hooksPath .githooks
chmod +x .githooks/pre-commit tools/clang_format.sh tools/install_git_hooks.sh

echo "Configured git hooks for this clone: .githooks"
