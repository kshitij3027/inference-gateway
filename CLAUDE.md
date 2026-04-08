# CLAUDE.md — Project Instructions

## Repository Overview

This is a **single backend/infrastructure project**. All source code, configuration, tests, and infrastructure definitions live in one repository with a well-defined directory structure.

---

## Project Structure

```
project-root/
├── CLAUDE.md
├── AGENTS.md
├── .gitignore
├── README.md
├── DESIGN.md                  # Architecture decisions & tradeoffs
├── Makefile                   # Project commands (up, down, test, seed, etc.)
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml / requirements.txt
├── .env.example               # Template for environment variables
├── config/                    # Declarative configuration files (YAML, JSON)
├── src/ or <package_name>/    # Application source code
│   ├── __init__.py
│   ├── main.py
│   └── ...
├── tests/                     # Unit and integration tests
│   ├── unit/
│   └── integration/
├── scripts/                   # Bootstrap, seed, and utility scripts
├── grafana/ / prometheus/     # Observability config (dashboards, scrape configs)
└── .claude/
    ├── skills/                # Agent workflow skills
    └── rules/                 # Language-specific and tooling conventions
```

---

## Core Rules

### 1. Understand the Structure Before Changing It

- Read relevant source files and configuration before making changes.
- Understand how modules connect — especially service boundaries, shared state, and config flow.
- If unsure which module or file a change belongs to, **ask before editing**.

### 2. Respect Service Boundaries

- Keep distinct concerns in separate modules (e.g., routing logic, caching logic, rate limiting).
- Don't leak implementation details across module boundaries — interact through well-defined interfaces.
- Infrastructure config (Docker Compose, Makefile, Prometheus config, Grafana dashboards) lives at the project root or in dedicated config directories, not mixed into application source.

### 3. Configuration is Declarative, Code is Not

- Service topology, backend definitions, feature flags, and tunables belong in configuration files (YAML, env vars), not hardcoded in source.
- Never hardcode service URLs, ports, credentials, or environment-specific values. Use environment variables or config files.
- Include `.env.example` with placeholder values for every required env var.

### 4. Commit Early, Commit Often

- After each **small, independent, testable** piece of work is completed and tested, commit and push.
- Commit message format: `[scope] short imperative description`
  - Scope examples: `router`, `cache`, `rate-limiter`, `docker`, `tests`, `docs`, `config`
- See `.claude/skills/commit-workflow/SKILL.md` for the full procedure.

### 5. Maintain .gitignore

- Keep `.gitignore` updated with common patterns (IDE files, OS files, build artifacts).
- Include language/framework-specific ignores (`__pycache__/`, `*.pyc`, `.env`, `dist/`, `build/`, `node_modules/`, `.pytest_cache/`, `.mypy_cache/`).
- Never commit secrets, `.env` files, or credentials. If a project requires env vars, include a `.env.example` with placeholder values.

### 6. Plans Go in .claude/plans/

- Whenever you create an implementation plan, write it to a file in the `.claude/plans/` folder (e.g., `.claude/plans/feature-name.md`).
- Add `.claude/plans/` to `.gitignore` — plans are working documents, not committed artifacts.

### 7. Test in Docker After Every Commit

- After each commit, run **unit tests inside Docker** (e.g., `make test`) — never run tests only on the host machine.
- **E2E / data-flow testing is mandatory, not optional.** After every commit (where services can be run), run a real end-to-end test that exercises the full data flow through the running services. This is the most important verification step.
- Unit tests alone are not sufficient. The primary question to answer after each commit is: **"Does the actual data/user flow work end-to-end?"** If you can spin up containers and push real data through the system, do it.
- Do **not** proceed to the next commit until both unit tests and E2E tests pass in Docker.
- If a Dockerfile or docker-compose file was changed, verify it builds before committing.

### 8. Context Management & Delegation

The main agent thread is an **orchestrator, not an implementer**. Protect its context window aggressively.

- **Hard limit**: Main thread context must stay below **60%** at all times. If approaching this threshold, compact immediately.
- **Delegation threshold**: Any implementation task that involves creating or modifying **3 or more files** MUST be delegated to a subagent via the Task tool.
- **Main thread responsibilities** (only these):
  1. Read the plan / understand requirements
  2. Create and manage the task list
  3. Spin up subagents with clear, self-contained prompts
  4. Verify subagent output (read key files, run tests, check Docker)
  5. Commit and push (using the commit-workflow skill)
- **Subagent responsibilities** (everything else):
  - Reading existing code for context
  - Writing new files and editing existing ones
  - Drafting tests
- **Parallelize when independent**: If two tasks don't share files or interfaces, run their subagents concurrently.
- **Subagent prompts must be self-contained**: Include the project path, which files to read first, what to create/modify, expected interfaces, and testing expectations. Never assume a subagent has prior context.
- See `.claude/skills/orchestration/SKILL.md` for the detailed delegation pattern.

### 9. DESIGN.md Is a Living Document

DESIGN.md is the project's architecture journal. It is updated **once per phase** — not per commit, not retroactively after all phases.

- The phased breakdown file (`InferenceGateway_Breakdown.md`) specifies a **section name** and **content guidance** for each phase. Writing that section is **mandatory output**, equal in importance to code and tests. A phase is not complete until its DESIGN.md section is written.
- If DESIGN.md doesn't exist, create it with a project title (`# InferenceGateway — Design Document`) and the first section.
- Write the section **during the phase** as you learn from implementing — don't defer it to the end.
- The final commit of a phase must include the DESIGN.md update: `[docs] DESIGN.md — <section name>`.
- **Never rewrite or restructure previous phases' sections** — only append new sections.
- Content covers architecture decisions, tradeoffs, and "why" — not just implementation details.

**Use this exact section template:**

```markdown
## <Feature Name>

### Why this exists
<Problem being solved — what breaks or is missing without this component>

### How it works
<Step-by-step request/data flow through this component>

### Implementation
<Key modules, classes, functions, storage mechanisms, algorithms used>

### Key design decisions
<Why this specific approach was chosen>

### Alternatives considered
<Option A vs Option B — what was rejected and why>

### Failure modes and edge cases
<Timeouts, partial failures, race conditions, stale state, split-brain, etc.>

### Observability
<Logs, metrics, admin/debug endpoints related to this component>

### Testing
<Unit tests, integration tests, load tests — what is covered and how>

### Production gaps
<What is simplified for local dev vs what a real deployment would need>

### Interview talking points
- <Point 1>
- <Point 2>
- <Point 3>

### Likely interview questions
**Q:** ...
**A:** ...
```

---

## Development Conventions

### README Standards

The project README should include:
1. **Title & one-line description** — what this project is.
2. **Architecture diagram** — how services connect (Mermaid or ASCII).
3. **Tech stack** — languages, frameworks, databases, tools used.
4. **How to run** — step-by-step instructions (prefer Docker-based: `make up`, `make seed`, etc.).
5. **How to test** — how to run unit tests, integration tests, E2E tests.
6. **API docs or usage** — endpoints, CLI commands, example requests/responses.
7. **Design decisions** — link to DESIGN.md for architecture tradeoffs.

### Code Quality

- Write clean, readable code with reasonable comments where intent isn't obvious.
- Follow the idiomatic style of whatever language/framework the project uses.
- Include basic error handling — don't leave happy-path-only code.
- Place tests in a conventional location (`tests/`, `tests/unit/`, `tests/integration/`).
- Check `.claude/rules/` for language-specific and tooling conventions (linting, formatting, type checking, etc.).

### Docker

- Every runnable service should have a `Dockerfile`.
- Use multi-stage builds where appropriate to keep images lean.
- If the project needs multiple services, use `docker-compose.yaml`.
- Pin base image versions (e.g., `python:3.12-slim`, not `python:latest`).
- Use a `Makefile` to wrap common Docker operations (`make up`, `make down`, `make test`, `make logs`).

### Dependencies & Environment

- Always include the dependency manifest (`pyproject.toml`, `requirements.txt`, `package.json`, `go.mod`, etc.).
- Include lock files in version control where applicable.
- Never hardcode secrets, ports, or absolute paths — use env vars and config files.

---

## Branching (Optional but Recommended)

- `main` branch should always be in a working state.
- For non-trivial features, work on a feature branch: `<scope>/<feature>` (e.g., `cache/semantic-lookup`, `infra/add-grafana`).
- Merge to `main` when the feature is complete and tested.

---

## Skills & Rules

- **Skills** (`.claude/skills/`) cover workflows: commit flow, Docker testing, orchestration, UI testing.
- **Rules** (`.claude/rules/`) cover coding conventions: language-specific patterns, linting/formatting preferences, tooling instructions. Check these before writing code.

## Plugins

Two plugins are installed locally (see `.claude/settings.local.json`). Always use their skills and agents during implementation.

- **python-development** — Python 3.12+ conventions, async patterns, FastAPI, testing, tooling (ruff, mypy, uv, pytest)
- **backend-development** — API design, architecture patterns, resilience, observability, microservices, test strategy

See `.claude/rules/plugins.md` for usage directives and `.claude/rules/project.md` for project-specific context.
