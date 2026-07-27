# work

Entry point for the shared `/work` orchestrator, shared across Claude Code, Codex, and
Antigravity.

Read `.agents/AGENTS.md`, then `.agents/skills/work/SKILL.md`, and follow those instructions
exactly. This file only exists to give Antigravity a native `/work` trigger — all workflow
logic, stage model, agent routing, and artifact contracts live in the files above; do not
duplicate or reinterpret them here.

Use `.agents/agents/<role>/agent.md` for persona definitions when acting as a subagent role
within this workflow.
