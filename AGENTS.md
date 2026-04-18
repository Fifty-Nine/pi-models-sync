# Agent Guide

This document provides instructions and standards for agents working on this project.

## Makefile Usage

The project uses a `Makefile` to manage development tasks. Use these commands to ensure consistency and maintain project standards.

### Setup
- `make setup`: Initializes the development environment. It syncs dependencies using `uv` and installs `pre-commit` hooks. Use this when starting in a new environment or when dependencies change.

### Verification & Testing
- `make test`: Runs the test suite using `pytest`. This should be run after any logic changes.
- `make testverbose`: Runs tests with verbose output. Useful for debugging specific failures.
- `make check`: Runs all static analysis tools (linting and type checking). This is a composite command for `lint`, `lint-sg`, and `typecheck`.
- `make all-checks`: Runs `pre-commit` hooks across all files. This is the most comprehensive verification step.

### Static Analysis
- `make lint`: Runs `ruff` to check for code style and formatting issues.
- `make lint-sg`: Runs `sg scan` for project-specific structural or semantic linting.
- `make typecheck`: Runs `mypy` to verify type safety.

### Formatting
- `make format`: Automatically formats the code and applies safe fixes using `ruff`. Run this before committing changes if you have made structural or stylistic edits.

### Workflow Recommendations
1. **After making changes:** Run `make format` followed by `make check`.
2. **Before finishing a task:** Always run `make test` and `make all-checks` to ensure no regressions and that all standards are met.

## Style Guidelines

### Line Length
- **Constraint:** Lines must not exceed **80 characters**.
- **Enforcement:** This is strictly enforced by the linter (`ruff`). The CI and `make check` commands will fail if this limit is exceeded.
- **Remediation:** If the linter reports a line length violation, use `make format` or manually refactor the code to stay within the 80-character limit.

### Nesting
- **Constraint:** Avoid deep nesting of logical blocks.
- **Enforcement:** Logical control flow (if, for, while, try, with, match) and nested functions must not exceed **3 levels** of depth. This is enforced by `ast-grep` via `make lint-sg`.
- **Remediation:** Use early returns (guard clauses), extract complex logic into separate functions, or flatten the control flow to maintain readability and stay within the depth limit.
