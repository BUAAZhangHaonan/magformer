# Release Scope And Branch Cleanup Subplan

1. Confirm `master` is the checked-out branch and capture the dirty tree state.
2. Confirm whether `feature/ucn-msmformer-repair` is already merged into `master` so the release uses a normal forward commit, not a history rewrite.
3. Keep the existing repair code on `master`, commit the remaining working-tree changes there, and push `master` to origin.
4. Delete `feature/ucn-msmformer-repair` locally with `git branch -d`.
5. Delete `origin/feature/ucn-msmformer-repair` with `git push origin --delete feature/ucn-msmformer-repair`.
6. Verify that only `master` remains in normal use by checking local and remote branch lists again.
