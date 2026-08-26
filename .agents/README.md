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
no extra step to discover skills — both read `.agents/skills/` directly.

Antigravity users only: this repo also ships `.agents/rules/work-invariants.md`, an Always-On
reminder of `$work`'s core invariants. Antigravity rule files carry no in-file field for
activation mode — after import, open Antigravity's Customizations → Rules panel and set this
rule's Activation Mode to **Always On** once. Without that manual step the file is imported but
inert. Claude Code and Codex CLI ignore `.agents/rules/` entirely; no action needed there.

Optionally, add a short pointer to the shared workflow in your project's own root
`AGENTS.md`/`CLAUDE.md` — see `.agents/templates/root-instructions-snippet.md`. This isn't
required for discovery but improves activation reliability on tools that always load their root
instruction file as context.
