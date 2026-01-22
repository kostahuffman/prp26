# Contributing

Thanks for contributing to this repository.
This project enforces **automatic code quality checks** to keep the codebase clean, consistent, and reviewable.

Please read this before making your first commit.

---

## Development Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 2. Create and activate a Python environment

Use your preferred tool (`venv`, `uv`, `poetry`, etc.), then install dependencies as per the project README.

---

## Pre-commit (Required)

This repository uses **pre-commit** to run linting and formatting checks automatically.

### One-time setup

Install pre-commit:

```bash
pip install pre-commit
```

Install the git hooks:

```bash
pre-commit install
```

Once installed, hooks run automatically on every commit.

---

## What happens on commit

On each `git commit`, pre-commit will:

* Lint code using **Ruff**
* Auto-fix safe issues (e.g. import sorting, unused imports)
* Format code using **Ruff formatter**

If pre-commit modifies files:

1. The commit will fail
2. Review the changes
3. Re-stage the files
4. Commit again

This is expected behavior.

---

## Running checks manually

Run all hooks on all files (recommended after first setup):

```bash
pre-commit run --all-files
```

Run hooks only on staged files:

```bash
pre-commit run
```

Run a specific hook:

```bash
pre-commit run ruff
```

---

## If a commit fails

Most failures are auto-fixable.

Typical workflow:

```bash
git commit
# pre-commit applies fixes
git add .
git commit
```

If an error cannot be auto-fixed, correct it manually and re-commit.

---

## Skipping hooks (not recommended)

If you must bypass pre-commit locally:

```bash
git commit --no-verify
```

⚠️ CI will still enforce checks — use this sparingly.

---

## Updating hooks (maintainers only)

To update hook versions:

```bash
pre-commit autoupdate
```

---

## Code Style & Expectations

* Follow existing code patterns and structure
* Do not disable Ruff rules without discussion
* Keep changes focused and minimal
* Add tests where applicable

---

## Troubleshooting

### Hooks not running?

```bash
pre-commit install
```

### Cache or environment issues

```bash
pre-commit clean
pre-commit run --all-files
```

### Python version issues

Ensure you are using the Python version specified by the project.
