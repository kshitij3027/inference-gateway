# AGENTS.md — Agent Behavior Guidelines

This file defines behavioral expectations for any AI coding agent (Claude Code, Codex, etc.) operating in this repository.

---

## Identity & Context

- This repository is a **backend/infrastructure project** with multiple interconnected services and shared state.
- Prioritize correctness, clarity, and robustness over cleverness.
- When in doubt about anything — scope, module boundaries, tech choices — **ask the user**.

---

## Hard Rules (Never Violate)

| # | Rule |
|---|------|
| 1 | **Understand the codebase structure before making changes.** Read relevant files and understand module boundaries first. |
| 2 | **Don't modify infrastructure config (Docker Compose, Makefile, CI) without confirming intent.** These affect the entire system. |
| 3 | **If the scope of a change is ambiguous, ask for confirmation before proceeding.** |
| 4 | **Commit after each small, testable, independent unit of work.** Push to the remote. |
| 5 | **Never commit secrets, `.env` files, or credentials.** Use `.env.example` for templates. |
| 6 | **Don't refactor unrelated modules while working on a feature.** Stay focused on the task at hand. |

---

## Git Workflow

See `.claude/skills/commit-workflow/SKILL.md` for the full commit procedure.

Key points:
- One logical change per commit.
- Run available tests/linters before committing when possible.
- Push to remote after each commit or after a small batch.
- Stage specific files or directories relevant to the change — avoid accidentally staging generated files or secrets.

---

## Interaction Style

- **Be explicit about what you're about to do** before doing it. ("I'll add the circuit breaker module in `src/circuit_breaker.py` and its tests.")
- **Summarize what was done** after completing a unit of work. ("Added the rate limiter with sliding window algorithm. Unit tests pass. Committed and pushed.")
- If a task is large, **break it into steps** and confirm the plan before executing.
- When you encounter an error or test failure, explain what went wrong and how you plan to fix it before making changes.

---

## Testing & Validation

- If the project includes tests, run them after making changes and before committing.
- If Docker is set up, verify the container builds and services start successfully after relevant changes.
- For API projects, do a quick smoke test (curl, httpie, or similar) when feasible.
- For multi-service projects, verify inter-service connectivity (can the app reach the database, can the metrics scraper reach the app, etc.).
- If you cannot run or test something (e.g., missing external service, API key), note it in the commit message or tell the user.

---

## File & Folder Hygiene

- `.gitignore`: IDE/OS patterns (`.idea/`, `.vscode/`, `.DS_Store`, `*.swp`, `*.log`) plus language-specific patterns (`__pycache__/`, `*.pyc`, `.env`, `node_modules/`, `dist/`, `build/`, `.pytest_cache/`, `.mypy_cache/`).
- Keep generated/build artifacts out of version control.
- Include dependency lock files in version control.
- Configuration templates (`.env.example`, sample YAML) are committed; actual config with secrets is not.

---

## Things to Avoid

- Don't install global system packages unless absolutely necessary and confirmed by the user.
- Don't refactor or "improve" unrelated modules while working on a specific task.
- Don't introduce unnecessary dependencies. Prefer standard library solutions for simple tasks.
- Don't create deeply nested abstractions for straightforward logic. Keep it simple.
- Don't assume a tech stack or tooling preference. If the user hasn't specified, ask.
- Don't modify shared infrastructure (Docker Compose, Makefile, config files) as a side effect of a feature change without calling it out.
