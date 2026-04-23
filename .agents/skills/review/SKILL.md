---
name: review
description: Review implemented changes against requirements, plan, scope, and engineering quality, then approve, request changes, or escalate.
---

# Purpose

Use this skill for code review after implementation.

This skill owns:
- code review against requirements and plan
- scope enforcement
- regression and maintainability review
- decision on whether changes are ready to proceed

This skill does not rewrite the implementation unless explicitly instructed.

# When to Use

Use when:
- implemented changes are available
- `requirements.yaml`, `plan.yaml`, and `implementation-report.yaml` exist

Do not use when:
- there is no concrete implementation to inspect
- the task is still in active development

# Inputs

- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`
- changed files/diff
- relevant repo docs

# References

- agent definition: `agents/reviewer.md`
- checklist: `references/review-checklist.md`
- upstream contracts: `templates/requirements.yaml`, `templates/plan.yaml`, `templates/implementation-report.yaml`

# Outputs

Primary output:
- review decision and actionable findings

# Agent Model

`/review` uses the `reviewer` agent.

The reviewer should use the shared code-review checklist.

# Decision Outcomes

Return exactly one:
- `APPROVED`
- `APPROVED_WITH_FOLLOWUP`
- `CHANGES_REQUIRED`
- `ESCALATE`

# Review Standard

Check:
- requirements compliance
- adherence to plan and task boundaries
- correctness and edge cases
- integration risk
- documentation impact
- adequacy of local validation already performed

# Escalation Rules

Escalate when:
- the plan conflicts with the implementation target
- review requires product decisions not present in the artifacts
- the changed behavior cannot be judged responsibly from the available context

# Style Guidance

- Findings first.
- Be concrete and file-aware.
- Tie feedback to requirements, plan, or correctness.
