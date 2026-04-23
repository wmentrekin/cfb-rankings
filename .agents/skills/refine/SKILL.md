---
name: refine
description: Critique and improve docs/<feature>/plan.yaml against requirements, drive the plan review loop, and approve or escalate within a maximum of three cycles.
---

# Purpose

Use this skill to review and tighten a plan before implementation starts.

This skill owns:
- plan review
- revision control for the plan loop
- approval or escalation

This skill does not gather new requirements unless the plan must be sent back to `/discover`.

# When to Use

Use when:
- a draft `docs/<feature>/plan.yaml` exists
- `requirements.yaml` exists
- the workflow needs a design review before implementation

Do not use when:
- no concrete plan exists
- requirements are still unstable

# Inputs

- `docs/<feature>/requirements.yaml`
- draft `docs/<feature>/plan.yaml`
- relevant repo context if needed

# References

- agent definition: `agents/reviewer.md`
- checklist: `references/review-checklist.md`
- template under review: `templates/plan.yaml`

# Outputs

Required output:
- approved or revised `docs/<feature>/plan.yaml`

The reviewed artifact must remain at `docs/<feature>/plan.yaml` and conform to `templates/plan.yaml`.

Optional outputs:
- explicit escalation to `/discover` or the user

# Agent Model

`/refine` uses the `reviewer` agent to critique the plan.

The reviewer must evaluate:
- requirements alignment
- scope control
- feasibility
- task decomposition quality
- validation completeness
- risks and open questions

# Loop Rules

Maximum loop count:
- 3 review cycles

After 3 cycles, do one of:
- approve if good enough
- escalate to the user
- send back to `/discover` if requirements are the real problem

Do not loop indefinitely.

# Decision Outcomes

Return exactly one:
- `APPROVED`
- `APPROVED_WITH_CHANGES`
- `CHANGES_REQUIRED`
- `ESCALATE`

# Approval Standard

The plan is ready only if:
- it aligns with requirements
- task boundaries are clear
- dependencies are explicit
- validation expectations exist
- major risks are known
- `/implement` should not need to guess

# Escalation Rules

Escalate when:
- requirements and design conflict
- architecture choices cannot be resolved responsibly
- the plan still depends on unknown facts
- scope is no longer bounded

# Style Guidance

- Be specific and actionable.
- Prefer targeted changes over broad rewrites.
- Protect implementation from ambiguity.
