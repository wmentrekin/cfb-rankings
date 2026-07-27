# agent-skills

Shared workflow assets for importing into project repos under `.agents/`. After import, start with `.agents/AGENTS.md` and use `/work`.

Works natively with Claude Code, Codex, and Google Antigravity — see "Tool Compatibility" in
`.agents/AGENTS.md` for how each tool discovers it.

```bash
git remote add agent-skills https://github.com/wmentrekin/agent-skills.git
git subtree add --prefix=.agents agent-skills main --squash
bash .agents/scripts/link-tools.sh
```

```bash
git subtree pull --prefix=.agents agent-skills main --squash
bash .agents/scripts/link-tools.sh
```

`link-tools.sh` symlinks `.claude/skills/work` and `.codex/skills/work` back to the canonical
`.agents/skills/work`, giving Claude Code and Codex a native `/work` skill. Antigravity needs no
extra step — it reads `.agents/workflows/` and `.agents/agents/` directly.
