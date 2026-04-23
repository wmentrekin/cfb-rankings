---
name: debug
description: Investigate failures in a read-only manner, identify likely causes and next actions, and hand off diagnosis without making fixes.
---

# Purpose

Use this skill to diagnose failures without modifying code.

This skill owns:
- failure investigation
- evidence gathering
- narrowing likely causes
- recommending the next owner

This skill is read-only.

# When to Use

Use when:
- `/test`, `/review`, or `/validate` surfaces a failure or ambiguity
- diagnosis is needed before more code changes

Do not use when:
- the root cause is already obvious and implementation changes can proceed directly
- the task is actually missing requirements rather than exhibiting a failure

# Inputs

- failing command or observed behavior
- relevant artifacts (`requirements.yaml`, `plan.yaml`, `implementation-report.yaml`)
- logs, traces, test output, or code paths tied to the failure

# References

- shared overview: `references/workflow-overview.md`
- upstream contracts: `templates/requirements.yaml`, `templates/plan.yaml`, `templates/implementation-report.yaml`

# Outputs

Primary output:
- diagnosis report in chat

The diagnosis report must include:
- observed failure
- evidence examined
- likely cause
- confidence level
- suggested next owner: `implement`, `test`, `review`, `discover`, or `validate`

# Read-Only Rule

Do not:
- edit files
- apply fixes
- rewrite configs

If a fix seems obvious, still report it as a recommendation rather than making the change.

# Escalation Rules

Escalate when:
- the failure depends on inaccessible systems or missing data
- the environment is too incomplete for diagnosis
- the real issue appears to be requirements ambiguity

# Style Guidance

- Be evidence-driven.
- Separate confirmed facts from hypotheses.
- Recommend the smallest responsible next step.
