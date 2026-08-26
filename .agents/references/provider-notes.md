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
| Antigravity | `AGENTS.md` and `GEMINI.md` | Reads both as always-loaded baseline context, in addition to its own `.agents/rules/` files (12,000-char limit each, with Manual/Always-On/Model-Decision/Glob activation modes). As of 2026-08-26, `.agents/rules/work-invariants.md` is populated (this framework's first use of the mechanism) — see "2026-08-26 research: cross-tool drift/re-anchoring" below for the format details and why activation mode can't be set via frontmatter. |

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

## 2026-08-26 research: cross-tool drift / re-anchoring (context-parity feature)

Findings from the `context-parity` feature's research, recorded here per this file's stated
purpose so they aren't re-derived from scratch.

- **Codex CLI — one-shot AGENTS.md load, no refresh.** `AGENTS.md` is built once per session and
  injected as ordinary conversation history; there is no refresh without restarting Codex
  (developers.openai.com/codex/guides/agents-md, checked 2026-08-26).
- **Codex CLI — 32 KiB truncation cap, no warning, declined as a fix.** `project_doc_max_bytes`
  silently drops concatenated `AGENTS.md` content past 32 KiB with no warning
  (openai/codex#7138, closed not-planned 2026-03-02, confirmed still current as of 2026-08-26).
  Not currently triggered by this framework's own content — the largest consuming repo's root
  `AGENTS.md` is 7,555 bytes.
- **Codex CLI — documented drift/forgetting mid-session.** Two dated, first-party-acknowledged
  issues (openai/codex#18517, openai/codex#3923) describe Codex drifting from or forgetting
  earlier instructions mid-session, with OpenAI attributing this partly to model training rather
  than confirming a purely architectural cause. Treat Codex mitigations in this framework as
  best-effort, not a guaranteed fix.
- **Antigravity — Always-On rules exist but have no in-file frontmatter for activation mode.**
  Confirmed directly from antigravity.google/docs/rules-workflows/ (fetched 2026-08-26): a rule
  is "simply a Markdown file"; activation mode (Manual / Always On / Model Decision / Glob) is
  set through the editor's Rules panel UI, not via a written frontmatter field in the file. Cross-
  checked against a real example (`JuliusBrussee/caveman` PR #117, `.agents/rules/caveman.md`),
  which ships with no frontmatter, confirming this isn't an omission specific to the docs page.
  `.agents/rules` is the current default location; `.agent/rules` is still supported as legacy.
  Rule files are capped at 12,000 characters.
- **Antigravity — no documented compaction guarantee for Always-On rules.** No official
  documentation or reverse-engineered evidence confirms Always-On rule content is protected from
  compaction under context pressure; the most detailed available internals analysis (a community
  gist reverse-engineering Antigravity's context handling) shows checkpoint-based pruning with no
  special-casing found for rules/AGENTS.md content. Treat as "best available lever," not proven.
- **Antigravity — gitignored `.agents/`/`.agent/` silently disables rule loading.** Rules fail to
  load entirely if the rules directory is gitignored, even with "Agent Gitignore Access" enabled
  (dyoshikawa/rulesync#981). Verified this repo's own `.gitignore` contains only
  `INITIAL_CONTEXT.md` — no exclusion of `.agents/`, `.agent/`, or (in this canonical-source repo)
  the top-level `rules/` directory that maps to it via subtree.
- **Antigravity — hard context-limit crashes reported, not graceful compaction.** Users report
  hitting hard context-limit crashes rather than graceful auto-compaction, with Antigravity
  explicitly contrasted unfavorably against Claude Code's and Cursor's auto-compaction (Google AI
  Developer Forum thread, 2026-03-03; acknowledged by Google with no fix/timeline given).
- **Claude Code — its own unresolved compaction report.** An unresolved, stale-closed GitHub
  issue (anthropics/claude-code#24460) reports `CLAUDE.md` rules vanishing after `/compact` in
  some real sessions, with no maintainer technical rebuttal, only auto-closure for inactivity.
  Claude Code's design intent is closer to a protected system-level invariant for `CLAUDE.md`
  than the other two tools, but this is not airtight — hence this framework's stage-transition
  re-anchoring (see `AGENTS.md`'s "Re-Anchor at Every Stage Transition") is worded tool-
  agnostically rather than treating Claude Code as immune.

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

Last verified: 2026-08-26, against official docs for each tool at that date (Antigravity rule
format re-confirmed directly from antigravity.google/docs/rules-workflows/ on 2026-08-26; see
"2026-08-26 research" above for the rest of that pass's sourcing).
Re-verify before trusting this table if it's more than a few months old —
this space moves fast.
