# Maintaining LeRobot

`lerobot/` is a Git subtree imported from https://github.com/huggingface/lerobot.git.
Initial upstream commit: `e0d50211ef236143ae867228662b7dfaba554f02`. Upstream history is squashed.

Edit and commit its source in this repository. Prefer separate commits for LeRobot
changes and other project changes. All machines use normal `git pull`; no submodule
initialization or separate fork is needed. Keep upstream license notices.

To update upstream, start on a clean working tree, create an update branch, and use
a deliberately selected upstream tag or branch:

```bash
git switch -c update-lerobot
git subtree pull --prefix=lerobot https://github.com/huggingface/lerobot.git <tag-or-branch> --squash
```

Resolve conflicts, validate SmolVLA/LIBERO evaluation, then merge the update branch.
Record the selected upstream commit and environment versions with experiment results.
Model caches, environments, local `.libero/` configuration and outputs are not source.

Conversion deliberately excluded pre-existing uncommitted evaluation changes and
`run_libero10_smolvla.sh`; they remain local until explicitly committed.
