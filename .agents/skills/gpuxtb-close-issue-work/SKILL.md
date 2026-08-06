---
name: gpuxtb-close-issue-work
description: Execute and hand off material gpuxtb work using GitHub issues as external memory, including priority and blocker discovery, acceptance ledgers, branches/worktrees, checkpoints, AI attribution, PR review, squash-merge gates, Epic updates, and honest Closes-versus-Refs decisions. Use when starting, resuming, checkpointing, reviewing, merging, or closing repository issue work.
---

# Close gpuxtb Issue Work

Keep enough durable state that another agent can continue without reconstructing the work from chat or Git history. Read the root `AGENTS.md`; it is authoritative for issue priority, attribution, review, and merge policy.

## Establish the Source of Truth

Before material work, run read-only discovery:

```bash
gh issue view 1 --comments
gh issue view <leaf> --comments
gh pr list --state open
git branch --show-current
git rev-parse HEAD
git status --short --branch
gh run list --limit 20
```

Read every directly blocking issue. Confirm priority labels, assignee, branch, scope, acceptance criteria, dependencies, and planned test matrix. If material work has no leaf issue, locate the correct existing issue or create a scoped one only when repository work and GitHub writes are authorized.

Use a dedicated branch or worktree from current `main`. Do not share overlapping files between concurrent editing agents.

## Maintain an Acceptance Ledger

Translate the issue body and blockers into a ledger before implementation:

| Criterion | Evidence needed | Status | Artifact or command |
| --- | --- | --- | --- |
| One issue acceptance item | Exact test, review, profile, legal decision, or package result | `PASS`, `FAIL`, `SKIP`, or `UNVERIFIED` | Commit, log, artifact, or URL |

Use `$gpuxtb-select-validation` to populate test evidence. `PASS` requires the intended backend and configuration to execute. Code landing, compile-only CUDA, a skipped test, or green unrelated CI does not satisfy a runtime, sanitizer, profile, package, legal, or conformance criterion.

Use `Closes #N` only when every ledger row passes. Otherwise use `Refs #N` and leave the remaining criteria explicit.

## Write Durable Checkpoints

At every material checkpoint, update the leaf issue with:

- branch, commit, and PR number;
- exact commands and pass/fail/skip/not-registered counts;
- decisions, units, public semantics, and important invariants;
- blockers and incomplete acceptance rows;
- the next concrete action.

Immediately before every GitHub comment, PR body, review, or commit, read the actual attribution values:

```bash
codex --version
rg -n '^(model|model_reasoning_effort)\s*=' ~/.codex/config.toml
```

Append the exact required attribution block or commit trailers from `AGENTS.md`. Never reuse values from an earlier checkpoint. Stop before writing if any value cannot be verified. Preserve attribution from other agents for retained work.

## Prepare and Review the PR

1. Keep commits focused and include the issue reference and verified attribution trailers.
2. Put the scope, decisions, exact validation, skips, and remaining gates in the PR body.
3. Review the complete diff and changed public behavior, not only the last commit.
4. Attach findings to precise changed lines; use minimal GitHub suggestion blocks when a safe complete fix is local.
5. Fix blockers on an authorized PR branch, rerun affected gates, and obtain independent review for risky changes.
6. Recheck conflicts and required CI at the final head.

Do not force-push shared work, weaken an acceptance gate, or merge because an earlier head was green.

## Merge and Hand Off

Squash-merge only when the final head is conflict-free, review-clean, and every required check is green. After merge:

1. Verify the squash subject, body, issue reference, and attribution trailers on the actual main commit.
2. Fast-forward local `main` without discarding unrelated work.
3. Update the leaf issue with the main SHA and final ledger.
4. Update Epic #1 with the main SHA, completed scope, remaining release gates, and next queue.
5. Remove only clean temporary branches/worktrees whose ownership and merge state are verified.

Leave the issue open when any acceptance item is `SKIP`, `UNVERIFIED`, or blocked. State the next action precisely enough for a new agent to execute it.
