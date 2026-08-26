# $work Invariants (Antigravity Always-On rule)

> Setup note (not part of the rule content): per antigravity.google/docs/rules-workflows/,
> Antigravity rule files carry no in-file frontmatter field for activation mode — a rule is
> "simply a Markdown file," and activation (Manual / Always On / Model Decision / Glob) is set
> through the editor's Customizations → Rules panel when the rule is added. Set this rule's
> Activation Mode to **Always On** there. This note is a setup instruction, not a rule for the
> agent to follow.

This project uses the `$work` workflow (`.agents/skills/work/SKILL.md`). Re-anchor to these
invariants every turn, and especially after any compaction or long gap in the session.

## Never commit directly to the base/main branch

All code and doc changes — including `$work`'s own `docs/<feature>/*.yaml` artifacts — land via
a feature branch and PR, under every mode, including quick-fix. Never commit or push directly to
`main`/`master`. If you are about to edit a tracked file while still on the base branch, stop and
create or switch to the feature branch first (`work/<feature-slug>`, created immediately after
mode classification, before any `docs/<feature>/*.yaml` file is written).

## Four internal stages

`$work` always operates in exactly one of:

1. discovery
2. planning
3. execution
4. verification

These are internal states, not user-facing commands. A mode may make some stages minimal; it
never skips them silently.

## Re-anchor before acting

Before taking any nontrivial action, check `docs/<feature>/status.yaml` for the current `stage`,
`objective`, `branching.branch_name`, and `next_action`. Re-read it again at every stage
transition (discovery→planning, planning→execution, execution→verification) — not only once at
session start. If status.yaml and your own working memory disagree, trust status.yaml and update
it.

## Checkpoint before execution

Never move from planning into execution without an explicit user decision (proceed / revise plan
/ narrow scope / stop) recorded in status.yaml's `checkpoint` block.

See `.agents/AGENTS.md` and `.agents/skills/work/SKILL.md` for the full process — this file is a
condensed reminder, not a replacement for either.
