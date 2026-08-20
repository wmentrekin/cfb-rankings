# agent-skills

Shared workflow assets for importing into project repos under `.agents/`. After import, start with `.agents/AGENTS.md` and use `/work`.

Works natively with Claude Code, Codex CLI, and Google Antigravity — see "Tool Compatibility" in
`.agents/AGENTS.md` and `.agents/references/provider-notes.md` for how each tool discovers it.

```bash
git remote add agent-skills https://github.com/wmentrekin/agent-skills.git
git subtree add --prefix=.agents agent-skills main --squash
bash .agents/scripts/link-claude-skills.sh
```

```bash
git subtree pull --prefix=.agents agent-skills main --squash
bash .agents/scripts/link-claude-skills.sh
```

`link-claude-skills.sh` symlinks `.claude/skills/work` back to the canonical
`.agents/skills/work`, giving Claude Code a native `/work` skill. Codex CLI and Antigravity need
no extra step — both read `.agents/skills/` directly.

Optionally, add a short pointer to the shared workflow in your project's own root
`AGENTS.md`/`CLAUDE.md` — see `.agents/templates/root-instructions-snippet.md`. This isn't
required for discovery but improves activation reliability on tools that always load their root
instruction file as context.
