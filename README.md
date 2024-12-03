# tt-torch

The tt-torch project requires environment setup from tt-mlir project. https://github.com/tenstorrent/tt-mlir/
See documentation of tt-mlir and follow Environment setup. https://docs.tenstorrent.com/tt-mlir/build.html

## Steps:
```
source env/activate
cmake -G Ninja -B build
cmake --build build
cmake --install build
```

## Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)
