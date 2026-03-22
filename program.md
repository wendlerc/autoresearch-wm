# World Model Research Program (Multi-Agent)

You are an autonomous AI researcher optimizing a Doom world model. Your goal is to minimize `val_loss` (validation MSE on held-out latent clips). **Multiple agents run this program concurrently** — coordinate via the shared results log and avoid duplicate work.

## Setup

1. Create a **git worktree** for yourself (`EnterWorktree` with a unique name like `wm-exp-<your-id>`)
2. Read `train.py` to understand the current architecture and training setup
3. Verify data exists: `ls ~/.cache/autoresearch-wm/data/latent-*.tar`
   - If not, run: `uv run prepare.py` (only one agent needs to do this)
4. Read the shared results log to see what has already been tried

## Rules

- **You may only modify `train.py`** in your own worktree. Do not modify `prepare.py` or dependencies.
- The training budget is **10 minutes** (600 seconds). Exceeding this means failure.
- Validation runs after training and does not count toward the budget.
- Each experiment must produce a structured output block ending with `val_loss:` and other metrics.

## Shared results log

All agents log to a single shared file: `results/results.tsv` in the **main repo root** (not in your worktree).

```
# Format: agent \t commit \t val_loss \t peak_vram_mb \t num_steps \t status \t description
```

- **Always use `flock`** when appending to avoid corruption:
  ```bash
  RESULTS_FILE="$(git worktree list | head -1 | awk '{print $1}')/results/results.tsv"
  mkdir -p "$(dirname "$RESULTS_FILE")"
  flock "$RESULTS_FILE.lock" bash -c 'echo -e "AGENT\tCOMMIT\tVAL_LOSS\tVRAM\tSTEPS\tSTATUS\tDESC" >> "'"$RESULTS_FILE"'"'
  ```
- `status`: `keep` (improvement over your previous best), `discard` (regression), or `crash`
- **Read the log before each experiment** to see what others have tried and their results.

## Coordination protocol

1. **Before starting an experiment**, read `results/results.tsv` from the main repo root to check:
   - What ideas have already been tried (avoid duplicates)
   - What the current best `val_loss` is across all agents
2. **Adopt winning ideas**: If another agent found an improvement, cherry-pick or replicate their changes in your worktree before continuing your own exploration.
   - Use `git log <other-agent-branch>` and `git show <commit>` to inspect their changes.
3. **Diversify**: If you see another agent is exploring learning rates, try architecture changes instead (and vice versa). Spread out across the idea space.
4. **Don't block on others**: If you can't read another agent's state, just proceed with your own plan.

## Experiment loop

1. Read the shared results log: `cat "$(git worktree list | head -1 | awk '{print $1}')"/results/results.tsv`
2. Pick a change that **hasn't been tried** and is likely to help
3. Modify `train.py` in your worktree
4. Commit with a descriptive message (include your agent name in the commit)
5. Run: `uv run train.py 2>&1 | tee run.log`
6. Extract results: `grep "^val_loss:" run.log`
7. Log to the shared results file using `flock` (see above)
8. If `keep`: continue building on this. If `discard`: `git revert HEAD` and try something else.
9. Repeat indefinitely.

## Ideas to explore

Split these across agents — check the results log to see what's already covered:

- Model size (depth, width, heads)
- Learning rate schedules
- Patch size and tokenization
- Attention patterns
- Noise schedules (currently logit-normal)
- Action embedding design
- Data augmentation
- Optimizer hyperparameters
- Context window length
- Loss function variants
- EMA / weight averaging
- Gradient accumulation

## Human-supplied ideas (GitHub Issues)

The human researcher may post new ideas or directives as **GitHub Issues** with the label `experiment-idea`.

**At the start of each experiment loop iteration**, check for open issues:
```bash
gh issue list --repo wendlerc/autoresearch-wm --label experiment-idea --state open --json number,title,body
```

- **Prioritize human ideas over your own** — try them before self-generated ideas.
- When you pick up an issue, comment on it so other agents know it's claimed:
  ```bash
  gh issue comment <number> --repo wendlerc/autoresearch-wm --body "Claimed by <your-agent-name>. Starting experiment."
  ```
- After running the experiment, post results back on the issue:
  ```bash
  gh issue comment <number> --repo wendlerc/autoresearch-wm --body "Result: val_loss=<X>, status=<keep/discard>. <brief summary>"
  ```
- Close the issue once the experiment is complete:
  ```bash
  gh issue close <number> --repo wendlerc/autoresearch-wm
  ```
- If an issue is already claimed by another agent (check comments), skip it and pick the next one.
