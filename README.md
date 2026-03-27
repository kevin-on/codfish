# codfish

My hobby project to train a chess engine that can reach **2500+ Elo** on a single GPU.

## Build

Requirements:

- CMake 3.23+
- A C++23 compiler
- `clang-format` for C/C++ formatting checks
- GoogleTest available to CMake
- `uv` for the managed Python environment when building learner bindings
- Python 3 development headers only when `CODFISH_BUILD_LEARNER_PYTHON=ON`

On macOS with Homebrew, GoogleTest can be installed with:

```bash
brew install googletest
```

## Python Environment

The learner Python bindings use a project-local virtual environment in
`.venv`, managed by `uv`.

Create the default environment:

```bash
uv sync
```

This installs the Python dependencies used by the learner and orchestrator,
including `numpy`, `torch`, and `wandb`.

## Configure

Use explicit build directories so the `PEXT` mode is always clear from the path.

- Fallback move generator (`NO_PEXT`), no learner Python bindings:

```bash
cmake -S . -B build/no-pext -DCODFISH_ENABLE_PEXT=OFF
```

- Fallback move generator (`NO_PEXT`) with learner Python bindings:

```bash
cmake -S . -B build/no-pext-python \
  -DCODFISH_ENABLE_PEXT=OFF \
  -DCODFISH_BUILD_LEARNER_PYTHON=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

- BMI2/PEXT runtime on x86_64 with no learner Python bindings:

```bash
cmake -S . -B build/pext -DCODFISH_ENABLE_PEXT=ON
```

- BMI2/PEXT runtime on x86_64 with learner Python bindings:

```bash
cmake -S . -B build/pext-python \
  -DCODFISH_ENABLE_PEXT=ON \
  -DCODFISH_BUILD_LEARNER_PYTHON=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

Build after configuring:

```bash
cmake --build build/no-pext
```

Run the test suite:

```bash
ctest --test-dir build/no-pext --output-on-failure
```

For faster local iteration, skip the smoke test:

```bash
ctest --test-dir build/no-pext -LE smoke --output-on-failure
```

If you built learner Python bindings, run the same commands against the matching
`*-python` build directory instead, for example:

```bash
cmake --build build/no-pext-python
ctest --test-dir build/no-pext-python --output-on-failure
```

The configure step writes `compile_commands.json` into the selected build
directory for clangd and other editor tooling.

## Formatting

C/C++ formatting uses `.clang-format`.

Check all tracked C/C++ files:

```bash
tools/clang_format.sh check
```

Rewrite all tracked C/C++ files:

```bash
tools/clang_format.sh fix
```

Python linting and formatting uses `ruff`, including import sorting.

Check all Python files:

```bash
uvx ruff check python
uvx ruff format --check python
```

Rewrite Python files and apply safe lint fixes:

```bash
uvx ruff check --fix python
uvx ruff format python
```

Enable the tracked git hooks for this clone:

```bash
tools/install_git_hooks.sh
```

The `pre-commit` hook auto-formats staged C/C++ files, runs `ruff` fixes on
staged Python files, and re-stages them. It refuses to run if one of those
files still has unstaged changes, so partial commits do not silently widen.
GitHub Actions runs the same C/C++ formatting and Python lint/format checks on
every push and pull request.
