# Model Routing

Route models per task instance, not permanently by agent role. The same developer may need a fast
model for a deterministic rename and a frontier model for a security-sensitive migration.

The orchestrator selects the cheapest tier and reasoning level likely to produce an accepted
result. Optimize for cost or quota consumed per accepted task, not price per token or a single
benchmark score.

## Capability Tiers

### frontier

Use for ambiguous requirements, architecture, cross-repository integration, novel failure modes,
high-risk migrations, security-sensitive work, final adjudication, or tasks where an incorrect
decision is expensive.

Default reasoning: `high`. Use the provider's maximum reasoning only when task evidence justifies
it; maximum effort is not the general default.

### balanced

Use for most implementation, plan critique, code review, test design, platform research, and
bounded debugging.

Default reasoning: `medium`.

### fast

Use for repository inventory, path and asset discovery, deterministic transformations, lint/type/
test execution, result extraction, and routine handoff documentation.

Default reasoning: `low`. Use `medium` when synthesis is required.

## Routing Factors

Classify each handoff using:

- `task_profile`: orchestration, research, implementation, review, test_execution, test_design,
  debugging, or documentation
- `uncertainty`: low, medium, or high
- `consequence`: low, medium, or high cost of an incorrect result
- `scope_size`: narrow, bounded, or cross_cutting
- `determinism`: deterministic, interpretive, or open_ended

Use `frontier` when uncertainty and consequence are both high, or when the orchestrator must make
a cross-cutting decision. Use `fast` only when the task is narrow and deterministic. Default to
`balanced` otherwise.

## Default Task Routing

| Task | Tier | Reasoning |
|---|---|---|
| Orchestration, requirements, architecture | frontier | high |
| Routine planning and plan review | balanced | medium |
| Repository and asset discovery | fast | low |
| Complex repository or platform research | balanced | medium |
| Ordinary implementation | balanced | medium |
| Mechanical bounded implementation | fast | low or medium |
| Lint, types, unit, integration, smoke execution | fast | low |
| Test design and integration analysis | balanced | medium |
| Routine code review | balanced | medium |
| High-risk review or ambiguous debugging | frontier | high |
| Reports and handoff documentation | fast | low or medium |

## Escalation Policy

- A worker may request one tier increase when evidence shows the selected tier is insufficient.
- Return evidence and the unanswered question before escalating; do not silently retry.
- Do not retry repeatedly at the same model and reasoning configuration.
- Model escalation counts as part of the existing correction loop and does not bypass the 2-cycle
  workflow cap.
- The orchestrator may downgrade a task after decomposition makes it narrower or more
  deterministic.

## Provider Resolution

Resolve capability tiers to currently available provider models using
`.agents/references/provider-model-map.md`. If the requested model is unavailable, choose the
closest model in the same tier, record the substitution, and preserve the routing intent.

Model choice is advisory where a tool cannot enforce per-subagent selection. Record what was
requested and what the runtime actually used; never claim enforcement that the provider does not
support.
