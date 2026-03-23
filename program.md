# World Model Research Program (Multi-Agent)

**PROTOCOL_VERSION = 3** — if you haven't re-read this file recently, do so now. `git fetch origin multiagent && git show origin/multiagent:program.md > /tmp/program.md && cat /tmp/program.md`

You are an autonomous AI researcher optimizing a Doom world model. Your goal is to minimize `val_loss` (validation MSE on held-out latent clips). **Multiple agents run this program concurrently** — coordinate via the shared results log and claim board to avoid duplicate work.

## Setup

1. Create a **git worktree** for yourself (`EnterWorktree` with a unique name like `wm-exp-<your-id>`)
2. Read `train.py` to understand the current architecture and training setup
3. Verify data exists: `ls ~/.cache/autoresearch-wm/data/latent-*.tar`
   - If not, run: `uv run prepare.py` (only one agent needs to do this)
4. Read the shared results log and claim board:
   ```bash
   REPO_ROOT="$(git worktree list | head -1 | awk '{print $1}')"
   cat "$REPO_ROOT/results/results.tsv"
   cat "$REPO_ROOT/results/claims.tsv"
   ```
5. **Skip the baseline if it already exists**: If results.tsv already has entries, do NOT re-run unmodified train.py. Instead, find the best `val_loss` entry, check out or cherry-pick that commit's train.py into your worktree, and start improving from there.

## Rules

- **You may only modify `train.py`** in your own worktree. Do not modify `prepare.py` or dependencies.
- The training budget is **1 hour** (3600 seconds). Exceeding this means failure.
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

## Claim board

Before starting any experiment, **claim it** so other agents don't try the same thing. The claim board is at `results/claims.tsv` in the main repo root.

```
# Format: agent \t timestamp \t base_commit \t status \t description
```

**Before each experiment:**
```bash
REPO_ROOT="$(git worktree list | head -1 | awk '{print $1}')"
CLAIMS_FILE="$REPO_ROOT/results/claims.tsv"
BASE_COMMIT=$(git rev-parse --short HEAD)
flock "$CLAIMS_FILE.lock" bash -c 'echo -e "AGENT\t$(date -Iseconds)\t'"$BASE_COMMIT"'\tstarted\tDESCRIPTION" >> "'"$CLAIMS_FILE"'"'
```

**After each experiment:**
```bash
flock "$CLAIMS_FILE.lock" bash -c 'echo -e "AGENT\t$(date -Iseconds)\t'"$BASE_COMMIT"'\tdone\tDESCRIPTION" >> "'"$CLAIMS_FILE"'"'
```

- Always read claims.tsv before picking an idea — if someone has `started` but not `done` for an idea, **don't try the same thing**.
- `base_commit` shows which commit is being improved upon, so you can see the lineage of experiments.

## Coordination protocol

1. **Before starting an experiment**, read both `results/results.tsv` and `results/claims.tsv` to check:
   - What ideas have already been tried or are in-flight (avoid duplicates)
   - What the current best `val_loss` is across all agents
   - What base commit others are building on
2. **Adopt winning ideas**: If another agent found an improvement, cherry-pick or replicate their changes in your worktree before continuing your own exploration.
   - Use `git log <other-agent-branch>` and `git show <commit>` to inspect their changes.
3. **Diversify**: If you see another agent is exploring learning rates (check claims.tsv), try architecture changes instead. Spread out across the idea space.
4. **Don't block on others**: If you can't read another agent's state, just proceed with your own plan.

## Experiment loop

1. Read **both** shared logs:
   ```bash
   REPO_ROOT="$(git worktree list | head -1 | awk '{print $1}')"
   cat "$REPO_ROOT/results/results.tsv"
   cat "$REPO_ROOT/results/claims.tsv"
   ```
2. Pick a change that **hasn't been tried AND isn't currently claimed** by another agent
3. **Claim it** by appending to `results/claims.tsv` with status `started` (see Claim board section)
4. Modify `train.py` in your worktree
5. Commit with a descriptive message (include your agent name in the commit)
6. Run: `uv run train.py 2>&1 | tee run.log`
7. Extract results: `grep "^val_loss:" run.log`
8. Log to `results/results.tsv` using `flock` (see Shared results log section)
9. Mark claim as `done` in `results/claims.tsv`
10. If `keep`: continue building on this. If `discard`: `git revert HEAD` and try something else.
11. Repeat indefinitely.

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

### Claiming rules
- **Prioritize human ideas over your own** — try them before self-generated ideas.
- **Any open issue is fair game** — even if other agents have already tried it and failed. Previous comments show what was attempted; try a **different approach** to the same idea.
- When you pick up an issue, comment what specific approach you're taking:
  ```bash
  gh issue comment <number> --repo wendlerc/autoresearch-wm --body "Agent <name>: trying <specific approach>."
  ```
- After running the experiment, post results:
  ```bash
  gh issue comment <number> --repo wendlerc/autoresearch-wm --body "Agent <name> result: val_loss=<X>, status=<keep/discard>. <brief summary>"
  ```
- **Do NOT close issues** — only the human researcher closes them.
