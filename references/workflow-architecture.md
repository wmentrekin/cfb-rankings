# Workflow Architecture

This workflow is designed for project-local use under `.agents/` and is driven only through `$work`.

## User-Facing Model

The user invokes `$work`.

`$work` is the outward-facing orchestrator. It stays in the main chat, spawns bounded agents, keeps the user updated, and maintains the live status board.

## Internal Stages

`$work` uses four internal stages:

1. discovery
2. planning
3. execution
4. verification

These are internal workflow states, not user-facing commands.

## Modes

Classify each request into one of these modes:

### quick-fix

Use when:
- the change is small and localized
- broad planning would be overhead

Required artifacts:
- `docs/<feature>/status.yaml`
- `docs/<feature>/implementation-report.yaml` once code changes start

Optional artifacts:
- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`

### investigation

Use when:
- the task is primarily understanding, diagnosis, or discovery
- implementation may or may not follow

Required artifacts:
- `docs/<feature>/status.yaml`

Optional artifacts:
- `docs/<feature>/requirements.yaml` if the investigation is likely to turn into build work

### bounded-feature

Use when:
- the work needs clear requirements, a plan, execution, and verification

Required artifacts:
- `docs/<feature>/status.yaml`
- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`

### large-initiative

Use when:
- the work spans multiple task batches or substantial coordination

Required artifacts:
- `docs/<feature>/status.yaml`
- `docs/<feature>/requirements.yaml`
- `docs/<feature>/plan.yaml`
- `docs/<feature>/implementation-report.yaml`

Additional expectation:
- execution should be broken into explicit batches

## Stage Guidance

### Discovery

Owns:
- understanding the task
- identifying relevant repo or external facts
- deciding whether a durable requirements artifact is needed

Default agents:
- repo-researcher
- platform-researcher when current external facts matter

### Planning

Owns:
- defining the implementation approach
- decomposing execution into bounded work
- preparing for the user checkpoint

Default agents:
- reviewer for plan critique
- repo-researcher if more repo facts are needed

### Execution

Owns:
- bounded implementation work
- keeping implementation handoff context current

Default agents:
- developer

### Verification

Owns:
- code review
- test execution
- read-only failure isolation
- deciding whether work can complete or must loop back

Default agents:
- reviewer
- tester
- repo-researcher if repo-local failure evidence is needed

## Loop Policy

Any correction loop may run at most 2 cycles before the orchestrator must stop and ask the user for direction.

## User Checkpoint

Before execution starts, the orchestrator must explicitly ask the user to:
- proceed
- revise plan
- narrow scope
- stop
