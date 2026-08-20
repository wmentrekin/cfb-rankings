# Provider Notes

Verified-current mechanics per tool for how this framework is discovered and run.
Written to avoid re-deriving this from scratch after every tool release; update
this file, not tribal knowledge, when something changes.

Anthropic published Agent Skills as an open standard (agentskills.io) on
2025-12-18. Codex CLI and Google's Antigravity adopted the same `SKILL.md`
convention within days. As a result, `.agents/skills/` is now natively readable
by three of the four surfaces this framework targets — only Claude Code still
needs an adapter.

## Skill discovery

| Tool | Native scan of `.agents/skills/`? | Notes |
|---|---|---|
| Claude Code | No | Scans `.claude/skills/` (project) and `~/.claude/skills/` (user). A skill entry may be a symlink to an arbitrary target; Claude Code follows it and reads `SKILL.md` from there. `.agents/skills/*` must be symlinked into `.claude/skills/` — see `scripts/link-claude-skills.sh`. |
| Codex CLI | Yes | Scans `.agents/skills` walking cwd → parent dirs → repo root, plus `$HOME/.agents/skills`, `/etc/codex/skills`. No `.codex/skills` needed. |
| Antigravity (IDE + `agy` CLI) | Yes | Defaults to `.agents/skills/` (legacy `.agent/skills/` also supported), plus global fallback paths. No adapter needed. |
| Gemini CLI | Sunset | Stopped serving individual/free-tier requests 2026-06-18. Replaced by Antigravity CLI (`agy`), which shares Antigravity's harness. Treat as historical. |

Frontmatter: the open spec requires only `name` and `description`. Keep
canonical `SKILL.md` files to exactly those two fields for maximum portability
— other tools ignore unrecognized keys per spec, but there's no upside to
adding tool-specific extensions to the shared file.

Activation on every tool above is driven by matching the request text against
the skill's `description` — this is the single highest-leverage lever for
reliable activation. Write descriptions with concrete trigger verbs/nouns and
an explicit negative boundary, not conceptual/architectural language.

## Persistent root instructions

| Tool | File | Scope/precedence |
|---|---|---|
| Claude Code | `CLAUDE.md` | Project root + `~/.claude/CLAUDE.md`; always loaded (not progressive). |
| Codex CLI | `AGENTS.md` | Walked from repo git root down to cwd, one file per level, concatenated (closer to cwd overrides). `AGENTS.override.md` beats `AGENTS.md` at the same level. 32 KiB cap (`project_doc_max_bytes`) — content beyond that is silently dropped. |
| Antigravity | `AGENTS.md` and `GEMINI.md` | Reads both as always-loaded baseline context, in addition to its own `.agents/rules/` files (12,000-char limit each, with Manual/Always-On/Model-Decision/Glob activation modes). |

Because this repo is imported under `.agents/` via subtree, it cannot ship a
project-root `AGENTS.md`/`CLAUDE.md` itself — those files belong to the
consuming project. See `templates/root-instructions-snippet.md` for the
optional one-time addition a user can make to their own root file(s).

## Subagents / role briefs

| Tool | Native persona format | How this repo's `agents/<role>/agent.md` maps in |
|---|---|---|
| Claude Code | `.claude/agents/*.md` (YAML+Markdown) | Not required. Claude Code reliably role-plays a role by reading `agent.md` and spawning a general-purpose subagent with that content as the prompt — this is the pattern already in use and it works. |
| Codex CLI | `.codex/agents/*.toml` only — no Markdown format | Cannot be registered natively. Read `agent.md`, pass its content as the prompt to Codex's generic `spawn_agent` primitive. |
| Antigravity | `.agents/agents/<name>.md` or `.agents/agents/<name>/agent.md` (YAML+Markdown) | Exact match for this repo's existing layout — reads natively, no adapter needed. |

Across all tools, the canonical pattern is: **`agent.md` is a portable role
brief, not a tool-specific registration.** The orchestrator reads it and hands
its content to whatever generic delegation primitive the current tool exposes.
This is intentional, not a gap — see `skills/work/SKILL.md`'s "Spawning a
Role" section.

## Known unknowns (flag, don't guess)

- Codex CLI's exact `$skill-name` explicit-invocation UX in interactive vs.
  `codex exec` (non-interactive/scripted) mode — not independently verified.
- Whether Gemini CLI's (legacy) `@agent-name` forced-delegation syntax has a
  direct Antigravity equivalent — not confirmed, and increasingly moot given
  the sunset.
- Antigravity's global skill-path precedence ordering across the IDE vs. `agy`
  CLI when the same skill name exists at multiple global fallback paths —
  sources partially conflict; workspace `.agents/skills/` reliably wins
  regardless, which is the only case this framework depends on.

Last verified: 2026-08-20, against official docs for each tool at that date.
Re-verify before trusting this table if it's more than a few months old —
this space moves fast.
