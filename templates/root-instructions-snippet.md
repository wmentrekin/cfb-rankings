# Root Instructions Snippet

Optional, one-time, manual step. This repo is imported under `.agents/` via
`git subtree`, so it cannot write files outside that prefix — this snippet is
not installed by any script.

Codex CLI and Antigravity both read a project's root `AGENTS.md` as always-loaded
context (unlike a skill's progressively-disclosed description), so a short
pointer there measurably improves activation reliability. Claude Code discovers
`/work` natively via the skill symlink and doesn't need this, but it's harmless
to include in `CLAUDE.md` too.

Add this section once to your project's own root `AGENTS.md` (and/or
`CLAUDE.md`) — not this repo's `AGENTS.md`:

```markdown
## Development Workflow

For substantive repository development, use the shared workflow defined at:

`.agents/skills/work/SKILL.md`

Use it for investigation, feature development, planning, implementation,
debugging, refactoring, review, and testing. Do not bypass it for substantive
engineering work.
```

Keep your project's own architecture, constraints, and conventions in your root
file as usual — this snippet only points at the shared workflow; it doesn't
replace or duplicate your project-specific instructions.
