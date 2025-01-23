
## Pre-Commit
Pre-Commit applies a Git hook to the local repository, ensuring linting is checked and applied on every git commit action. Install it from the root of the repository using:

```bash
source env/activate
pre-commit install
```

If you have already made commits before installing the pre-commit hooks, you can run the following to “catch up”:

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)
