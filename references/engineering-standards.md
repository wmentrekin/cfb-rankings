# Engineering Standards

Use this checklist during execution (all domain and generalist agents) and during code review.

## File Organization

- [ ] New files land where the existing project structure would predict, not in a new ad hoc
      location
- [ ] Naming matches the surrounding convention (case style, prefixes/suffixes, pluralization)
- [ ] No orphaned files left behind by moves/renames

## Code Quality

- [ ] No dead code, commented-out code, or unused imports/variables introduced
- [ ] No premature abstraction — duplication across two or three call sites is fine; don't
      generalize until a real third use case demands it
- [ ] Reuses existing helpers/utilities/patterns instead of reimplementing them
- [ ] Error handling matches what the codebase already does for similar cases — no invented
      fallback paths for scenarios that can't happen
- [ ] No unrequested scope creep: refactors, cleanups, or "while I'm here" changes outside the
      task's owned scope

## Documentation

- [ ] Comments explain non-obvious *why*, not restate *what* well-named code already shows
- [ ] Docs that describe changed behavior are updated in the same change, not left stale

## Consistency

- [ ] Matches existing formatting/lint configuration rather than introducing a new style
- [ ] Consistent with how similar features are already structured elsewhere in the repo
