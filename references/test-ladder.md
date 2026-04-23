# Test Ladder

Default order for `/test`:

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
