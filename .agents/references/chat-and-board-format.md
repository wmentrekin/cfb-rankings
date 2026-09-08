# Chat And Board Format

The orchestrator should keep the user informed while background agents run.

## Update Rule

The user should always see either:
- an active conversational update
- or a current work snapshot with the agent board

Update on:
- stage changes
- agent spawn
- agent completion
- blockers
- doubts
- loop-cap stops
- pre-execution checkpoint

## Standard Update Shape

Include:
- current objective
- current stage
- the most important new information
- blockers or open questions if any
- the next likely action

## Agent Board

Show the current board in a consistent form.

Suggested shape:

```text
Agent Board
- orchestrator | active | stage: planning | task: preparing draft plan
- repo-researcher-1 | running | stage: discovery | task: inspect API routes
- reviewer-1 | queued | stage: planning | task: critique plan draft
```

Each row should make clear:
- agent id
- status
- stage
- current task
- requested model tier and reasoning level when relevant
- actual provider model when known

If an agent is blocked, say why.

Suggested routed row:

```text
- developer-1 | running | execution | balanced/medium -> Sonnet 5 | task: implement parser
```

## Status Board Sync

Mirror the same state into `docs/<feature>/status.yaml` on state transitions.

## Conversation Rule

Do not let the workflow disappear into documents or silent loops.

Docs hold durable handoff detail.
Chat holds summaries, decisions, blockers, and next steps.
