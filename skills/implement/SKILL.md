---
name: implement
description: Execute one approved bounded task or one bounded batch of tasks from plan.yaml by delegating to developer agents and producing an implementation report.
---

# Purpose

Use this skill to carry out approved planned work.

This skill owns:
- selecting the next task or task batch from `plan.yaml`
- preparing developer handoffs
- orchestrating bounded development execution
- collecting implementation outputs
- writing `docs/<feature>/implementation-report.yaml`

This skill does not own final code review or final validation.

# When to Use

Use when:
- `docs/<feature>/plan.yaml` is approved
- one or more tasks are ready for implementation

Do not use when:
- planning is still incomplete
- the work is blocked on unanswered requirements questions

# Inputs

- approved `docs/<feature>/plan.yaml`
- relevant task slice from the plan
- repo conventions and source-of-truth docs

# References

- agent definition: `agents/developer.md`
- task handoff template: `templates/task-handoff.yaml`
- report template: `templates/implementation-report.yaml`
- upstream contract: `templates/plan.yaml`

# Outputs

Required output:
- `docs/<feature>/implementation-report.yaml`

Supporting output:
- task handoff contract for each developer agent

Write developer handoffs using `templates/task-handoff.yaml`.

Write the implementation report using `templates/implementation-report.yaml` and store it at `docs/<feature>/implementation-report.yaml`.

# Agent Model

`/implement` uses one or more `developer` agents.

Developers must receive:
- the task handoff
- only the relevant files and docs
- explicit owned scope
- explicit forbidden scope
- validation expectations
- escalation rules

Run developers in parallel only when owned scopes do not materially overlap.

# Responsibilities

`/implement` must:
1. choose the next bounded task set
2. define ordering and parallelism from the plan
3. issue task handoffs
4. collect code-change summaries and local validation results
5. record what was implemented, what remains, and any blockers
6. write the implementation report

# Implementation Report Standard

The report must capture:
- task ids attempted
- files/modules changed
- local validation performed
- known issues
- unresolved blockers
- recommended next step: `review`, `test`, `debug`, or `implement`

# Escalation Rules

Escalate when:
- the approved plan is insufficient for safe execution
- implementation requires a new architecture or product decision
- tasks overlap too heavily for safe delegation
- scope expansion would be required

# Style Guidance

- Keep execution tightly aligned to the approved plan.
- Favor small, reviewable diffs.
- Do not silently expand task boundaries.
