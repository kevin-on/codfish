#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: tools/clang_format.sh <check|fix> [files...]

Without file arguments, the script operates on all tracked C/C++ source files.
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

mode="$1"
shift

case "$mode" in
  check|fix)
    ;;
  *)
    usage
    exit 1
    ;;
esac

clang_format_bin="${CLANG_FORMAT_BIN:-clang-format}"

if ! command -v "$clang_format_bin" >/dev/null 2>&1; then
  echo "clang-format binary not found: $clang_format_bin" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

declare -a cpp_globs=(
  "*.c"
  "*.cc"
  "*.cpp"
  "*.cxx"
  "*.h"
  "*.hh"
  "*.hpp"
)

is_cpp_file() {
  case "$1" in
    *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

declare -a files=()

if [ "$#" -gt 0 ]; then
  for path in "$@"; do
    if [ -f "$path" ] && is_cpp_file "$path"; then
      files+=("$path")
    fi
  done
else
  while IFS= read -r -d '' path; do
    files+=("$path")
  done < <(git ls-files -z -- "${cpp_globs[@]}")
fi

if [ "${#files[@]}" -eq 0 ]; then
  exit 0
fi

if [ "$mode" = "check" ]; then
  "$clang_format_bin" --dry-run --Werror -style=file "${files[@]}"
else
  "$clang_format_bin" -i -style=file "${files[@]}"
fi
