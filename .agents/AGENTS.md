# AGENTS.md

This repo is the shared source of truth for subtree-friendly workflow assets that can be imported into project repos under `.agents/`.

## Purpose

Use these assets to run a structured workflow with explicit phase boundaries, narrow subagent context, and durable handoff artifacts stored in project repos under `docs/<feature>/`.

## Workflow

The default workflow is:

1. `/discover`
2. `/plan`
3. `/refine`
4. `/implement`
5. `/test`
6. `/review`
7. `/debug` when failures need diagnosis
8. `/validate`

Typical path:

`/discover -> /plan -> /refine -> /implement -> /test + /review -> /validate`

Use `/debug` as a read-only side loop when a failure needs investigation before more implementation.

## Durable Artifacts

Project-local outputs should live in `docs/<feature>/`:

- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`
- `docs/<feature>/validation-report.yaml`

Task-level developer coordination should use:

- `templates/task-handoff.yaml`

## Shared Templates

- `templates/requirements.yaml`
- `templates/plan.yaml`
- `templates/task-handoff.yaml`
- `templates/implementation-report.yaml`
- `templates/validation-report.yaml`

## Shared References

- `references/workflow-overview.md`
- `references/test-ladder.md`
- `references/review-checklist.md`
- `references/validation-checklist.md`

## Agent Roles

- [agents/developer.md](/Users/wentrekin/Documents/agent-skills/agents/developer.md)
- [agents/platform-researcher.md](/Users/wentrekin/Documents/agent-skills/agents/platform-researcher.md)
- [agents/researcher.md](/Users/wentrekin/Documents/agent-skills/agents/researcher.md)
- [agents/reviewer.md](/Users/wentrekin/Documents/agent-skills/agents/reviewer.md)
- [agents/tester.md](/Users/wentrekin/Documents/agent-skills/agents/tester.md)

## Principles

- Keep requirements separate from design.
- Keep design separate from execution.
- Use subagents with minimal necessary context.
- Prefer durable artifacts over long chat history.
- Keep the workflow small, explicit, and reusable across repos.
