# Test Ladder

Default order during verification:

1. lint
2. types
3. unit
4. integration
5. smoke

Guidelines:
- Move in order unless the task clearly justifies a narrower subset.
- Stop early on blocking failures.
- Report both executed checks and meaningful gaps.
- Escalate before risky environment-affecting validation.
- Use `fast/low` for deterministic command execution; use `balanced/medium` when designing tests
  or interpreting non-obvious failures, per `.agents/references/model-routing.md`.
