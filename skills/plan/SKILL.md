---
name: plan
description: Turn frozen requirements into an implementation design and task decomposition, then draft docs/<feature>/plan.yaml for refinement.
---

# Purpose

Use this skill to design the solution once requirements are stable.

This skill owns:
- implementation-oriented design
- task decomposition
- dependency mapping
- sequencing and parallelism decisions
- drafting `docs/<feature>/plan.yaml`

This skill does not own plan approval.

# When to Use

Use when:
- `docs/<feature>/requirements.yaml` exists and is effectively frozen
- the user wants an implementation-ready design

Do not use when:
- requirements are incomplete or unstable
- the task is still in discovery

# Inputs

- `docs/<feature>/requirements.yaml`
- relevant repo docs, code, and architecture references
- constraints from source-of-truth project docs

# References

- template: `templates/plan.yaml`
- upstream contract: `templates/requirements.yaml`
- shared overview: `references/workflow-overview.md`

# Outputs

Required output:
- draft `docs/<feature>/plan.yaml`

Write it using `templates/plan.yaml` and store it at `docs/<feature>/plan.yaml`.

# Responsibilities

The plan must define:
- design overview
- major technical decisions
- interfaces, data contracts, or schema implications as needed
- bounded tasks
- task ownership boundaries
- dependencies and ordering
- parallelizable work
- validation expectations per task
- delivery risks

# Task Decomposition Rules

Each task must include:
- id
- title
- objective
- owned scope
- dependencies
- parallelizable status
- expected validation

Tasks must be:
- bounded
- implementable
- reviewable
- testable

Do not produce vague task groups like "refactor backend" or "clean up UI".

# Prohibited Behavior

Do not:
- invent new requirements
- absorb unresolved product questions into the design
- defer major ambiguity to `/implement`

# Handoff Standard

The output must be concrete enough for `/refine` to critique and for `/implement` to execute after approval.

# Style Guidance

- Prefer implementable clarity over elegant abstraction.
- Make scope boundaries explicit.
- Call out assumptions instead of hiding them.
