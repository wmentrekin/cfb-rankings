# Branch and PR Workflow

`$work` owns the mechanical git/PR lifecycle for any mode that reaches execution. The user's
role is final approval and merge — agents never merge.

**Never commit directly to the base/main branch, under any mode.** All code and doc changes —
including `docs/<feature>/status.yaml` and every other `$work` artifact — land via a feature
branch and PR. This is a literal prohibition, not just an implication of the branch-then-PR
ordering below.

## Branch Creation

Create the feature branch immediately after mode classification, before any
`docs/<feature>/*.yaml` artifact is first written — for every mode, including investigation and
quick-fix:

- default: a single branch `work/<feature-slug>` off the current branch
- record it in `docs/<feature>/status.yaml` under `branching.branch_name` when status.yaml is first written
- `plan.yaml` (once it exists) copies the same value into its own `branching.branch_name` rather
  than re-deciding it — the branch already exists by planning time

**Accepted tradeoff:** because every mode now branches immediately, an investigation-mode
session that concludes with no follow-up work will still have created a branch that ends up
unused. This is expected, not a bug — it trades a small amount of local branch churn for closing
the gap where discovery/planning artifacts were written on the base branch. Cleanup: if a
session ends with no artifacts worth keeping, delete the unused local branch
(`git branch -D work/<feature-slug>`).

## Parallel Domains (Worktrees)

Use a worktree per domain only when the plan's `execution_batches` contain two or more tasks
that are all of:

- `parallelizable: true`
- assigned to different `assigned_domain` values
- touching non-overlapping `owned_scope`

For each qualifying task, create `git worktree add <path> -b work/<feature-slug>-<domain>
work/<feature-slug>` and hand that worktree's path to the corresponding domain agent. Record each
worktree under `branching.worktrees` in `plan.yaml`.

When a domain agent finishes its batch, merge `work/<feature-slug>-<domain>` back into
`work/<feature-slug>` and remove the worktree (`git worktree remove`). Do this before opening the
PR — the result is always **one integration PR per feature**, never one PR per domain.

Coupled or sequential tasks (dependencies, overlapping scope, or only one domain involved) stay
on the single shared branch — do not create a worktree just because a task exists.

## Opening the PR

Open the PR once `docs/<feature>/implementation-report.yaml` reflects completed execution for
the current batch(es), and before the verification stage runs — verification should happen as
real PR review activity, not only in chat.

- use the GitHub tools available in this environment (e.g. `create_pull_request`)
- record the PR number/URL in `docs/<feature>/status.yaml` (`pr.number`, `pr.url`, `pr.status`)
  and in `docs/<feature>/implementation-report.yaml` (`pr`)
- base title/description on `docs/<feature>/requirements.yaml` and `plan.yaml`'s objective

## Verification on the PR

`reviewer` and `tester` post their findings as real PR review comments (e.g.
`pull_request_review_write` + `add_comment_to_pending_review`) in addition to their existing
structured chat output. This is additive, not a replacement for the chat-facing report.

## Loop-Cap Fixes

Fixes from a review/test loop (still capped at 2 cycles) push additional commits to the same
branch and PR. Never open a new branch or PR for a fix cycle within the same feature.

## Completion

`$work` is not complete for any mode that reached execution until the PR exists and is reported
to the user as ready for review. The orchestrator's handoff is always some form of "PR #N is
ready for your review" — it never merges the PR itself.

Once the user explicitly confirms they are merging the PR, delete the entire
`docs/<feature>/` directory from the branch (`git rm -r docs/<feature>/`) and push that removal
as the final commit before merge. `docs/<feature>/status.yaml`, `requirements.yaml`, `plan.yaml`,
`implementation-report.yaml`, and any `task-handoff.yaml` instances written to disk are `$work`'s
own live coordination scratch state, not project deliverables, and must not persist in the
merged history.

This scope is deliberately narrow: it applies only to `$work`'s own generated
`docs/<feature>/` directory. Any genuinely pre-existing project documentation the feature
touched (README, existing `docs/` content unrelated to this feature's scratch directory, etc.)
must remain updated in place and is never deleted by this step.
