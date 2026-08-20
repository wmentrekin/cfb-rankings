---
name: work
description: >
  Primary engineering workflow for building, fixing, investigating, designing,
  planning, implementing, debugging, refactoring, modifying, reviewing, or
  testing code in this repository. Use for any substantive development task:
  new features, bug fixes, pipeline or model changes, infrastructure changes,
  architecture decisions, or code review. Coordinates requirements
  clarification, repository and platform research, scoped subagents, planning
  with an explicit user checkpoint, bounded execution, review, and testing.
  Do not use for trivial one-line edits or pure Q&A with no code change.
---

# Workflow Position

`$work` is the only user-facing workflow entrypoint.

It owns the conversation with the user and coordinates all internal workflow stages.

# Must Read First

1. `.agents/AGENTS.md`
2. `.agents/skills/work/SKILL.md`
3. `.agents/agents/orchestrator/agent.md`
4. `.agents/references/workflow-architecture.md`
5. `.agents/references/chat-and-board-format.md`
6. `.agents/references/branch-and-pr-workflow.md`
7. the mode-relevant templates in `.agents/templates/`

# Must Spawn

`$work` is orchestration-first.

Default behavior:
- use subagents for research, review, implementation, and testing
- keep the main session focused on coordination, synthesis, status updates, and user discussion
- do not do substantive editing or deep analysis in the main session unless the work slice is truly tiny and spawning would add more overhead than value

## Spawning a Role

Every agent under `.agents/agents/<role>/agent.md` is a portable role brief, not
a tool-specific persona registration. To act as (or spawn) a role:

1. read `.agents/agents/<role>/agent.md` in full
2. hand its content, plus the task-specific handoff, to whatever generic
   delegation primitive the current tool exposes (a general-purpose subagent
   call, an agent-spawning tool, etc.)
3. never assume the role name itself is a recognized subagent type — the
   brief's content is what defines the role, not a registration step

This works identically regardless of whether the current tool has a native
named-subagent format. See `.agents/references/provider-notes.md` for the
verified mechanics per tool.

# Inputs

- the user request
- repo docs and source-of-truth files
- existing project-local artifacts in `docs/<feature>/` if they exist
- project constraints from `.agents/AGENTS.md`

# Outputs

Always maintain:
- `docs/<feature>/status.yaml` using `.agents/templates/status.yaml`

Mode-dependent durable artifacts:
- `docs/<feature>/requirements.yaml` using `.agents/templates/requirements.yaml`
- `docs/<feature>/plan.yaml` using `.agents/templates/plan.yaml`
- `docs/<feature>/implementation-report.yaml` using `.agents/templates/implementation-report.yaml`

Execution agent handoffs use:
- `.agents/templates/task-handoff.yaml`

# User Updates

Keep the conversation active while background tasks run.

At minimum, update on:
- stage changes
- agent spawn
- agent completion
- blockers or doubts
- loop-cap stops
- pre-execution checkpoint

Each update should include:
- current objective
- current stage
- active agent board or a concise delta from it
- the latest important finding, blocker, or decision
- the next likely action

Use `$work` in recommendations, not old phase commands.

# Agent Board

Mirror the live board rules in `.agents/references/chat-and-board-format.md`.

The board should make it obvious:
- which agents exist
- what each one is doing
- which are blocked
- what finished recently

# Subagent Routing

Use these agents by default:

- `.agents/agents/repo-researcher/agent.md`
  Use for repo-local investigation in discovery, planning, or verification.

- `.agents/agents/platform-researcher/agent.md`
  Use only when current external facts are needed.

- `.agents/agents/reviewer/agent.md`
  Use for plan critique and code review.

- `.agents/agents/tester/agent.md`
  Use for verification, test running, and read-only failure isolation during verification.

## Execution Domain Routing

During execution, route each task to the domain agent named in its `assigned_domain` field in
`docs/<feature>/plan.yaml`, not to a single generic developer:

- `.agents/agents/data-engineer/agent.md` — ingestion, pipelines, schema, data quality
- `.agents/agents/analytics-engineer/agent.md` — transformation/semantic layer (dbt-style)
- `.agents/agents/data-scientist/agent.md` — analysis, modeling, experimentation
- `.agents/agents/mlops-engineer/agent.md` — training pipelines, model registry/deployment, monitoring
- `.agents/agents/platform-engineer/agent.md` — cloud infra (IaC, provisioning) and CI/CD delivery
- `.agents/agents/frontend-engineer/agent.md` — UI/UX implementation
- `.agents/agents/generalist-developer/agent.md` — fallback when no domain fits

When planning, assign each task's `assigned_domain` based on which files/components it touches.
For tasks with no clear domain fit, assign `generalist-developer`.

# Mode Selection

Classify the work up front using `.agents/references/workflow-architecture.md`.

Supported modes:
- `quick-fix`
- `investigation`
- `bounded-feature`
- `large-initiative`

Promote the mode if complexity grows.

# Internal Stage Model

`$work` uses four internal stages:

1. discovery
2. planning
3. execution
4. verification

Treat these as internal states, not separate user-facing commands.

# Process

1. classify the work mode
2. create or update `docs/<feature>/status.yaml`
3. run discovery only to the depth needed by the selected mode
4. create `requirements.yaml` if the mode requires it
5. create `plan.yaml` if the mode requires it
6. stop for the required pre-execution checkpoint
7. create the feature branch (and worktrees, if the plan calls for parallel domains) per
   `.agents/references/branch-and-pr-workflow.md`
8. spawn the routed domain (or generalist) agents for bounded execution
9. update `implementation-report.yaml`
10. open the PR per `.agents/references/branch-and-pr-workflow.md`
11. run verification with reviewer and tester agents, posting findings to the PR
12. either complete, loop back once or twice, or stop and ask the user for direction

# Loop Cap

No correction loop may exceed 2 cycles.

When a loop hits 2 cycles:
- stop autonomous looping
- summarize what happened
- explain what is still blocked
- offer options
- recommend one next step

# User Checkpoint

Before execution begins, require explicit user direction:
- proceed
- revise plan
- narrow scope
- stop

Do not silently move from planning into implementation.

# Escalation

Escalate when:
- product intent is unclear
- scope changes materially
- external facts remain ambiguous
- the loop cap is reached
- execution would require a new architecture or product decision
- verification results conflict and the next step is not obvious

# Completion

`$work` is complete only if:
- the current mode is still correct
- the live status board is current
- durable artifacts are sufficient for handoff
- for any mode that reached execution, a PR exists and is reported ready for the user's review
- the user has clear visibility into what happened, what remains, and what the next step is

# Next Recommended Command

- continue with `$work`
