# codfish

My hobby project to train a chess engine that can reach **2500+ Elo** on a single GPU.

## Build

Requirements:

- CMake 3.23+
- A C++23 compiler
- GoogleTest available to CMake
- Python 3, Python development headers, and NumPy only when
  `CODFISH_BUILD_LEARNER_PYTHON=ON`

On macOS with Homebrew, GoogleTest can be installed with:

```bash
brew install googletest
```

Configure for the target you want:

- Linux x86_64/BMI2 runtime with PEXT enabled:

```bash
cmake -S . -B build
```

- Local Apple Silicon development with the fallback move generator:

```bash
cmake -S . -B build -DCODFISH_ENABLE_PEXT=OFF
```

- Build the learner Python bindings as well:

```bash
cmake -S . -B build -DCODFISH_BUILD_LEARNER_PYTHON=ON
```

- Local Apple Silicon development with learner Python bindings:

```bash
cmake -S . -B build -DCODFISH_ENABLE_PEXT=OFF -DCODFISH_BUILD_LEARNER_PYTHON=ON
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

The configure step also writes `build/compile_commands.json` for clangd and
other editor tooling.
