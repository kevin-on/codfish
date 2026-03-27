# codfish

My hobby project to train a chess engine that can reach **2500+ Elo** on a single GPU.

## Build

Requirements:

- CMake 3.23+
- A C++23 compiler
- `clang-format` for C/C++ formatting checks
- GoogleTest available to CMake
- `uv` for the managed Python environment
- Python 3 development headers
- `torch` installed in the Python environment used for the build

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

This installs the Python dependencies used by the learner, orchestrator, and
AOTI export path, including `numpy`, `torch`, and `wandb`.

## Configure

Use `build/` as the canonical local development build directory. Re-run CMake
into the same directory when switching variants so `build/compile_commands.json`
stays aligned with clangd and Cursor.

Point CMake at the same Python environment you use for `torch` and derive
`CMAKE_PREFIX_PATH` from that interpreter.

- Fallback move generator (`NO_PEXT`):

```bash
PYTHON_BIN="$PWD/.venv/bin/python"
TORCH_CMAKE_DIR="$("$PYTHON_BIN" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "share/cmake")')"

cmake -S . -B build \
  -DCODFISH_ENABLE_PEXT=OFF \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_DIR"
```

- BMI2/PEXT runtime on x86_64:

```bash
PYTHON_BIN="$PWD/.venv/bin/python"
TORCH_CMAKE_DIR="$("$PYTHON_BIN" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "share/cmake")')"

cmake -S . -B build \
  -DCODFISH_ENABLE_PEXT=ON \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_DIR"
```

Build after configuring:

```bash
cmake --build build
```

Run the test suite:

```bash
ctest --test-dir build --output-on-failure
```

For faster local iteration, skip the smoke test:

```bash
ctest --test-dir build -LE smoke --output-on-failure
```

The configure step writes `compile_commands.json` into `build/` for clangd and
other editor tooling.

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
