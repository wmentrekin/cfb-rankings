---
name: discover
description: Identify the task, gather and freeze requirements, coordinate repo and optional web research, and produce docs/<feature>/requirements.yaml for downstream planning.
---

# Purpose

Use this skill at the start of work to clarify what should be built, why it matters, and what constraints define success.

This skill owns:
- task identification
- requirements gathering
- repo-local research coordination
- optional web/platform research coordination
- requirements freeze recommendation
- creation of `docs/<feature>/requirements.yaml`

This skill does not own solution design or code changes.

# When to Use

Use when:
- the user has a new feature, fix, refactor, or investigation request
- the task is not yet clearly defined
- repo context or external platform facts need to be gathered before planning

Do not use when:
- an approved `plan.yaml` already exists and the task is ready for execution
- the user only wants implementation on a clearly bounded existing plan

# Inputs

- user request
- repo docs and source-of-truth files
- prior work artifacts in `docs/<feature>/` if they exist
- project constraints from `AGENTS.md` or equivalent

# References

- agent definition: `agents/researcher.md`
- agent definition: `agents/platform-researcher.md`
- template: `templates/requirements.yaml`
- shared overview: `references/workflow-overview.md`

# Outputs

Required output:
- `docs/<feature>/requirements.yaml`

Optional supporting outputs:
- brief research notes in chat
- escalations to the user for unresolved product or scope questions

# Agent Model

`/discover` may use:
- `researcher` for repo-local investigation
- `platform-researcher` only when current external/platform/web facts are required

Both researchers must receive:
- a narrow question
- only the relevant files or source links
- a concrete expected output

# Workflow

## 1. Identify the Task

Normalize the user request into a bounded task definition:
- problem statement
- motivation
- target outcome
- obvious boundaries

If the task is too broad, narrow it before moving on.

## 2. Gather Requirements

Work with the user to capture:
- goals
- non-goals
- constraints
- assumptions
- acceptance criteria
- open questions

Do not move into design.

## 3. Run Targeted Research

Use repo research when local context is missing.

Use platform/web research only when:
- current APIs, libraries, vendor behavior, pricing, standards, or docs may have changed
- the task depends on latest information

Do not research speculatively.

## 4. Freeze Candidate Requirements

Requirements are ready for handoff when:
- the problem is clear
- goals and non-goals are explicit
- constraints are known
- acceptance criteria are testable
- open questions are resolved or explicitly marked blocking

If they are not ready, keep clarifying or escalate.

## 5. Write `requirements.yaml`

Write the durable requirements contract using the shared template.

# Escalation Rules

Escalate when:
- the request contains conflicting goals
- a major product decision is unresolved
- repo context is insufficient to define the task responsibly
- external research is needed but the question is still vague
- new scope appears that changes the task materially

Do not guess at product intent.

# Handoff Standard

`requirements.yaml` must be strong enough that `/plan` can produce a solution without reopening basic task-definition work.

Write it using `templates/requirements.yaml` and store it at `docs/<feature>/requirements.yaml`.

# Style Guidance

- Be collaborative and explicit.
- Ask targeted questions.
- Separate facts from assumptions.
- Keep the artifact concise but complete.
