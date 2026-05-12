# AGENTS.md

Use this file as the entry point when these workflow assets are imported into a project repo under `.agents/`.

## Command Convention

In Codex, invoke the workflow with `$work`.

Do not route the user across multiple top-level phase commands.

## First Read Order

When starting in a project repo, read in this order:

1. `.agents/AGENTS.md`
2. `.agents/skills/work/SKILL.md`
3. any referenced agent doc in `.agents/agents/`
4. any referenced template in `.agents/templates/`
5. any referenced checklist or guide in `.agents/references/`

Do not rely on memory of the workflow. Re-anchor to these files explicitly.

## Workflow Model

This workflow is orchestration-first.

The outward-facing agent is the orchestrator. It owns the user conversation and should do most real work through subagents, not in the main session.

The orchestrator should:

- orient itself
- classify the work mode
- decide the current internal stage
- spawn the correct subagent(s)
- monitor them while they run
- integrate outputs
- keep the user updated in chat
- maintain a live status board
- stop and ask for direction when it reaches uncertainty or loop limits

Direct main-session work should be the exception, not the default.

## Internal Stages

`$work` uses four internal stages:

- discovery
- planning
- execution
- verification

These are internal states, not separate user-facing commands.

## Skills

- `.agents/skills/work/SKILL.md`

## Agents

- `.agents/agents/orchestrator.md`
- `.agents/agents/developer.md`
- `.agents/agents/platform-researcher.md`
- `.agents/agents/repo-researcher.md`
- `.agents/agents/reviewer.md`
- `.agents/agents/tester.md`

## Templates

Project-local artifacts should be created under `docs/<feature>/` using these templates:

- `.agents/templates/status.yaml`
- `.agents/templates/requirements.yaml`
- `.agents/templates/plan.yaml`
- `.agents/templates/task-handoff.yaml`
- `.agents/templates/implementation-report.yaml`

Expected project-local outputs:

- `docs/<feature>/status.yaml`
- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`

## References

- `.agents/references/workflow-architecture.md`
- `.agents/references/chat-and-board-format.md`
- `.agents/references/review-checklist.md`
- `.agents/references/test-ladder.md`
- `.agents/references/verification-checklist.md`

## Core Rules

- Keep the orchestrator active in the user conversation.
- Keep most substantive work in subagents.
- Keep requirements, planning, execution, and verification as internal states.
- Use subagents with minimal necessary context.
- Use project-local paths explicitly when citing workflow files.
- Prefer durable artifact handoffs over long conversational context.
- Keep the docs sufficient for competent handoff to new agents.
- Maintain `docs/<feature>/status.yaml` as the live coordination artifact.
- Hard-stop any self-correction loop after 2 cycles and ask the user for direction.
- Require an explicit user checkpoint before execution starts.

## Chat Contract

The user should always see either:

- an active conversation update
- or a current high-level work snapshot with the live agent board

Every meaningful update should include:

- current objective
- current internal stage
- active or recently completed agents
- blocker, doubt, or decision if one exists
- next likely action

## Loop Policy

Any correction loop may run at most 2 cycles before the orchestrator must stop and ask for direction.

Applies to:

- planning revisions
- execution or review revisions
- execution or test revisions

When the cap is hit, return:

- what happened
- what is still blocked
- options
- one recommended next step

## User Checkpoint

Before starting execution, the orchestrator must explicitly ask the user to:

- proceed
- revise plan
- narrow scope
- stop

## Standard Handoff Contract

Every subagent handoff should be explicit about:

- objective
- relevant files
- allowed scope
- forbidden scope
- required output format
- escalation conditions

## Model Guidance

Recommended defaults:

- orchestrator: highest reasoning
- developer: medium to high reasoning based on task size
- repo-researcher: medium reasoning
- platform-researcher: medium or high reasoning based on ambiguity
- reviewer: medium reasoning
- tester: medium reasoning
