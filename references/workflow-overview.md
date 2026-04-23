# Workflow Overview

This repo defines a shared, subtree-friendly workflow for project-local use under `.agents/`.

## Skills

1. `/discover`
2. `/plan`
3. `/refine`
4. `/implement`
5. `/test`
6. `/review`
7. `/debug`
8. `/validate`

## Default Flow

`/discover -> /plan -> /refine -> /implement -> /test + /review -> /validate`

Use `/debug` when failures need diagnosis before more changes are made.

## Agent Roles

- `developer`: see `agents/developer.md`
- `platform-researcher`: see `agents/platform-researcher.md`
- `researcher`: see `agents/researcher.md`
- `reviewer`: see `agents/reviewer.md`
- `tester`: see `agents/tester.md`

## Durable Contracts

Artifacts stored in project repos under `docs/<feature>/`:

- `requirements.yaml`
- `plan.yaml`
- `implementation-report.yaml`
- `validation-report.yaml`

Supporting contract:

- `task-handoff.yaml`

## Workflow Principles

- Keep requirements separate from design.
- Keep design separate from execution.
- Use narrow context for subagents.
- Prefer durable phase-boundary artifacts over long chat history.
- Keep the process small and reusable across repos.
