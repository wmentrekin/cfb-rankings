#!/usr/bin/env bash
# Links this framework's skills into Claude Code's native skill directory.
#
# .agents/ is imported via `git subtree`, so everything in this repo lands nested under
# .agents/ in the consuming project. Claude Code (.claude/skills/) looks for skills outside
# that prefix, so this entry point can't be shipped by the subtree itself -- this script
# creates it once, in the consuming repo, after import.
#
# Codex CLI and Antigravity need nothing: both natively scan .agents/skills/ (and
# .agents/agents/*/agent.md) directly. This script exists solely because Claude Code
# does not. See .agents/references/provider-notes.md for the verified mechanics per tool.
#
# Safe to re-run any time, including after `git subtree pull` if skills are added or removed.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
skills_src="$repo_root/.agents/skills"

if [ ! -d "$skills_src" ]; then
  echo "error: $skills_src not found -- run this from a repo with .agents/ imported" >&2
  exit 1
fi

mkdir -p "$repo_root/.claude/skills"

for skill_path in "$skills_src"/*/; do
  skill_name="$(basename "$skill_path")"
  link_path="$repo_root/.claude/skills/$skill_name"
  rel_target="../../.agents/skills/$skill_name"
  if [ -L "$link_path" ] || [ -e "$link_path" ]; then
    rm -rf "$link_path"
  fi
  ln -s "$rel_target" "$link_path"
  echo "linked .claude/skills/$skill_name -> $rel_target"
done
