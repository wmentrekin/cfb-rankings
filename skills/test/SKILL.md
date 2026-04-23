---
name: test
description: Run graduated technical validation for implemented scope using the shared test ladder and report readiness, failures, and gaps.
---

# Purpose

Use this skill to validate changed code paths after implementation.

This skill owns:
- selecting the smallest sufficient checks
- running the graduated test ladder
- reporting failures and gaps

This skill does not change code unless explicitly instructed elsewhere.

# When to Use

Use when:
- `implementation-report.yaml` exists for the changed scope
- the work is ready for technical validation

Do not use when:
- the implementation is obviously incomplete
- there is no clear changed scope to validate

# Inputs

- approved `plan.yaml`
- `implementation-report.yaml`
- relevant changed files
- repo testing instructions

# References

- agent definition: `agents/tester.md`
- ladder: `references/test-ladder.md`
- upstream contracts: `templates/plan.yaml`, `templates/implementation-report.yaml`

# Outputs

Primary output:
- test results in chat or a structured handoff to `/debug`, `/implement`, or `/validate`

# Agent Model

`/test` uses the `tester` agent.

# Test Ladder

Use the shared ladder in order unless the task clearly justifies otherwise:
1. lint
2. types
3. unit
4. integration
5. smoke

Expand only as needed for confidence.

# Status Outcomes

Return exactly one:
- `PASSED`
- `PASSED_WITH_GAPS`
- `FAILED`
- `ESCALATE`

# Failure Handling

When checks fail:
- identify the failing check
- identify likely impacted task(s), file(s), or area(s)
- recommend whether the next owner is `/debug` or `/implement`

# Escalation Rules

Escalate when:
- the environment is unsafe or ambiguous
- destructive or live validation would be required
- the intended behavior cannot be inferred from the plan

# Style Guidance

- Distinguish confirmed failures from untested risk.
- Keep the test sequence disciplined and explicit.
