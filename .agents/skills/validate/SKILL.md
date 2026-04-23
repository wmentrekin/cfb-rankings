---
name: validate
description: Verify end-to-end readiness against requirements and plan, run final completion checks including pre-commit validation, and produce docs/<feature>/validation-report.yaml.
---

# Purpose

Use this skill at the end of the workflow to determine whether the work is actually ready to ship, merge, or hand off.

This skill owns:
- end-to-end verification against acceptance criteria
- cross-checking requirements, plan, implementation, review, and test status
- final completion checks
- pre-commit validation
- writing `docs/<feature>/validation-report.yaml`

# When to Use

Use when:
- implementation has completed
- review and test outcomes are available
- the user wants a final readiness decision

Do not use when:
- core implementation work is still incomplete
- there are known blocking failures that still require debugging or fixes

# Inputs

- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`
- review outcome
- test outcome
- repo-specific pre-commit or final-check instructions

# References

- checklist: `references/validation-checklist.md`
- report template: `templates/validation-report.yaml`
- upstream contracts: `templates/requirements.yaml`, `templates/plan.yaml`, `templates/implementation-report.yaml`

# Outputs

Required output:
- `docs/<feature>/validation-report.yaml`

Write it using `templates/validation-report.yaml` and store it at `docs/<feature>/validation-report.yaml`.

# Validation Standard

Validate:
- acceptance criteria coverage
- alignment between implemented scope and approved plan
- presence of unresolved blockers
- final technical readiness
- pre-commit checks or equivalent local completion checks
- any remaining release or merge caveats

# Status Outcomes

Return exactly one:
- `READY`
- `READY_WITH_CAVEATS`
- `NOT_READY`
- `ESCALATE`

# Escalation Rules

Escalate when:
- final readiness depends on missing evidence
- environment-specific checks are too risky or ambiguous
- requirements and final behavior still disagree

# Style Guidance

- Think in terms of completion, not just passing commands.
- Make caveats explicit.
- Keep the final report durable and decision-oriented.
