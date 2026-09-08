# Repository Synchronization Preflight

Run this preflight before discovery, artifact creation, branch creation, repository research, or
implementation. It applies whether the agent is working in a sandbox, a local checkout, or a
worktree.

## Scope Inventory

Identify every Git repository that may be read from, changed, used as a dependency, or used as a
source for generated output during the work. Record each repository and its relevant base branch
in `docs/<feature>/status.yaml`.

Do not assume the current working directory is the only repository involved. Ask the user when
repository scope cannot be determined safely.

## Required Checks

For each repository:

1. identify the repository root, current branch, relevant base branch, remotes, and worktree state
2. check for tracked and untracked changes before attempting synchronization
3. fetch the relevant remote branches
4. synchronize the relevant base branch with a fast-forward-only pull when operating in its clean
   primary checkout
5. in a feature worktree or detached checkout, fetch and compare against the remote base instead
   of switching branches or pulling into the worktree
6. record the local commit before synchronization, fetched remote commit, local commit after
   synchronization, and result before any other workflow action

## Safety Rules

- Never discard, stash, commit, or overwrite existing user changes merely to complete preflight.
- Never pull into a dirty working tree without explicit user direction.
- Never use a merge pull; use fast-forward-only synchronization.
- Never switch a user checkout away from an active feature branch just to update the base branch.
- If network access is unavailable, report the branch and remote state as unverified and ask the
  user whether to continue with the local state.
- If the base has diverged, stop and ask for direction. Do not rebase or merge automatically.
- A sandbox is not an exemption. Use its available remote/fetch facilities or report the exact
  limitation.

## Result States

Use one result for every repository:

- `current`: fetched and confirmed aligned with the relevant remote branch
- `updated`: fast-forwarded to the relevant remote branch
- `comparison_only`: worktree or feature checkout fetched and compared without switching
- `blocked_dirty`: local changes prevent safe synchronization
- `blocked_diverged`: local and remote histories diverged
- `unverified_network`: remote state could not be fetched
- `unverified_remote`: no usable remote or relevant branch was available

Any `blocked_*` result stops the workflow. Any `unverified_*` result requires an explicit user
decision before continuing.

## Chat Requirement

Report the repository inventory and synchronization outcome to the user before proceeding. Keep
the report concise when all repositories are current; explain blockers and choices when they are
not.
