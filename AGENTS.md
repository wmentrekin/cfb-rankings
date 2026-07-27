# AGENTS.md

Use this file as the entry point when these workflow assets are imported into a project repo under `.agents/`.

## Command Convention

This workflow is invoked as `/work` in all three supported tools:

- Claude Code: `/work`, discovered via `.claude/skills/work`
- Codex: `/work` (or `$work` mention), discovered via `.codex/skills/work`
- Antigravity: `/work`, discovered natively from `.agents/workflows/work.md`

See "Tool Compatibility" below for how each of these paths is populated.

Do not route the user across multiple top-level phase commands.

## Tool Compatibility

These assets are written to work natively across Claude Code, Codex, and Antigravity:

| Concept | Claude Code | Codex | Antigravity |
|---|---|---|---|
| Invokable skill | `.claude/skills/work/SKILL.md` | `.codex/skills/work/SKILL.md` | `.agents/workflows/work.md` |
| Persona/subagent | n/a (role-played via prompt) | n/a (role-played via prompt) | `.agents/agents/<name>/agent.md` |

`.agents/skills/work/SKILL.md` is the single canonical source. Antigravity reads `.agents/`
directly, so `.agents/workflows/work.md` and `.agents/agents/*/agent.md` work with no setup.
Claude Code and Codex look outside `.agents/`, so run `.agents/scripts/link-tools.sh` once
after import (and again after `git subtree pull` if skills are added or removed) to symlink
`.claude/skills/work` and `.codex/skills/work` back to the canonical source.

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

## Workflows

- `.agents/workflows/work.md` (Antigravity-native entry point; points back at the skill above)

## Agents

- `.agents/agents/orchestrator/agent.md`
- `.agents/agents/platform-researcher/agent.md`
- `.agents/agents/repo-researcher/agent.md`
- `.agents/agents/reviewer/agent.md`
- `.agents/agents/tester/agent.md`

Execution roles, routed per-task by domain (see `.agents/skills/work/SKILL.md`'s "Execution
Domain Routing"):

- `.agents/agents/data-engineer/agent.md`
- `.agents/agents/analytics-engineer/agent.md`
- `.agents/agents/data-scientist/agent.md`
- `.agents/agents/mlops-engineer/agent.md`
- `.agents/agents/platform-engineer/agent.md`
- `.agents/agents/frontend-engineer/agent.md`
- `.agents/agents/generalist-developer/agent.md` (fallback when no domain fits)

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
- `.agents/references/engineering-standards.md`
- `.agents/references/branch-and-pr-workflow.md`

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
- execution roles (data-engineer, analytics-engineer, data-scientist, mlops-engineer,
  platform-engineer, frontend-engineer, generalist-developer): medium to high reasoning based on
  task size
- repo-researcher: medium reasoning
- platform-researcher: medium or high reasoning based on ambiguity
- reviewer: medium reasoning
- tester: medium reasoning
