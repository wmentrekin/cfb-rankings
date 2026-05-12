---
name: work
description: Run the full workflow as an outward-facing orchestrator that classifies the work, keeps the user conversation active, spawns subagents, updates a live status board, and moves through discovery, planning, execution, and verification.
---

# Workflow Position

`$work` is the only user-facing workflow entrypoint.

It owns the conversation with the user and coordinates all internal workflow stages.

# Must Read First

1. `.agents/AGENTS.md`
2. `.agents/skills/work/SKILL.md`
3. `.agents/agents/orchestrator.md`
4. `.agents/references/workflow-architecture.md`
5. `.agents/references/chat-and-board-format.md`
6. the mode-relevant templates in `.agents/templates/`

# Must Spawn

`$work` is orchestration-first.

Default behavior:
- use subagents for research, review, implementation, and testing
- keep the main session focused on coordination, synthesis, status updates, and user discussion
- do not do substantive editing or deep analysis in the main session unless the work slice is truly tiny and spawning would add more overhead than value

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

Developer handoffs use:
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

- `.agents/agents/repo-researcher.md`
  Use for repo-local investigation in discovery, planning, or verification.

- `.agents/agents/platform-researcher.md`
  Use only when current external facts are needed.

- `.agents/agents/reviewer.md`
  Use for plan critique and code review.

- `.agents/agents/developer.md`
  Use for bounded execution work only.

- `.agents/agents/tester.md`
  Use for verification, test running, and read-only failure isolation during verification.

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
7. spawn developer agents for bounded execution
8. update `implementation-report.yaml`
9. run verification with reviewer and tester agents
10. either complete, loop back once or twice, or stop and ask the user for direction

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
- the user has clear visibility into what happened, what remains, and what the next step is

# Next Recommended Command

- continue with `$work`
