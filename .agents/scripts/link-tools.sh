#!/usr/bin/env bash
# Links this framework's skills into Claude Code's and Codex's native skill directories.
#
# .agents/ is imported via `git subtree`, so everything in this repo lands nested under
# .agents/ in the consuming project. Claude Code (.claude/skills/) and Codex (.codex/skills/)
# look for skills outside that prefix, so those entry points can't be shipped by the subtree
# itself -- this script creates them once, in the consuming repo, after import.
#
# Antigravity needs no such step: it natively scans .agents/workflows/, .agents/rules/, and
# .agents/agents/*/agent.md directly.
#
# Safe to re-run any time, including after `git subtree pull` if skills are added or removed.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
skills_src="$repo_root/.agents/skills"

if [ ! -d "$skills_src" ]; then
  echo "error: $skills_src not found -- run this from a repo with .agents/ imported" >&2
  exit 1
fi

for target_root in .claude/skills .codex/skills; do
  mkdir -p "$repo_root/$target_root"
done

for skill_path in "$skills_src"/*/; do
  skill_name="$(basename "$skill_path")"
  for target_root in .claude/skills .codex/skills; do
    link_path="$repo_root/$target_root/$skill_name"
    rel_target="../../.agents/skills/$skill_name"
    if [ -L "$link_path" ] || [ -e "$link_path" ]; then
      rm -rf "$link_path"
    fi
    ln -s "$rel_target" "$link_path"
    echo "linked $target_root/$skill_name -> $rel_target"
  done
done
