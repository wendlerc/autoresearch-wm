# World Model Research Program (Multi-Agent)

**PROTOCOL_VERSION = 4** — if you haven't re-read this file recently, do so now. `git fetch origin multiagent && git show origin/multiagent:program.md > /tmp/program.md && cat /tmp/program.md`

You are an autonomous AI researcher optimizing a Doom world model. Your goal is to **maximize `ar_auto_psnr`** (autoregressive generation quality in latent space, higher is better). This measures how well the model generates coherent frame sequences when running autoregressively — combining both per-frame quality AND temporal stability.

The model now supports KV caching for autoregressive inference. The eval decodes latents to pixels (via DC-AE VAE) and computes:
- **ar_auto_lpips**: LPIPS of AR-generated frames vs ground truth (THE PRIMARY METRIC, lower = better)
- **ar_auto_psnr**: PSNR of AR-generated frames vs ground truth (higher = better)
- **ar_tf_psnr / ar_tf_lpips**: metrics with teacher-forcing (upper bound)
- **ar_gap**: ar_tf_psnr - ar_auto_psnr (measures error accumulation, lower = better)

Previous experiments optimized val_loss (MSE). Results from that era are archived in `results/results_valloss_era.tsv`. Good starting configs can be found at the tagged commits: `git tag -l 'best-*'`.

**Multiple agents run this program concurrently** — coordinate via the shared results log and claim board to avoid duplicate work.

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
- **Everything in train.py is fair game**: model architecture, attention mechanism, normalization, optimizer, loss function, noise schedule, positional encodings, patching strategy — you can rewrite anything. Don't just tune hyperparameters.
- **NEVER STOP**: Do not pause to ask the human. You are autonomous. If you run out of ideas, think harder — re-read train.py for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you.
- **Simplicity criterion**: All else being equal, simpler is better. Removing code and getting equal or better results is a great outcome. A tiny improvement that adds ugly complexity is not worth it.

## Shared results log

All agents log to a single shared file: `results/results.tsv` in the **main repo root** (not in your worktree).

```
# Format: agent \t commit \t ar_auto_psnr \t ar_auto_lpips \t ar_tf_psnr \t ar_tf_lpips \t ar_gap \t val_loss \t peak_vram_mb \t num_steps \t status \t description
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

1. **Re-read the latest program.md** from the multiagent branch (the human may have updated it):
   ```bash
   git fetch origin multiagent 2>/dev/null
   git show origin/multiagent:program.md
   ```
   Always follow the latest version, even if your worktree has an older copy.
2. Read **both** shared logs:
   ```bash
   REPO_ROOT="$(git worktree list | head -1 | awk '{print $1}')"
   cat "$REPO_ROOT/results/results.tsv"
   cat "$REPO_ROOT/results/claims.tsv"
   ```
3. Pick a change that **hasn't been tried AND isn't currently claimed** by another agent
4. **Claim it** by appending to `results/claims.tsv` with status `started` (see Claim board section)
5. Modify `train.py` in your worktree
6. Commit with a descriptive message (include your agent name in the commit)
7. Run: `uv run train.py 2>&1 | tee run.log`
8. Extract results: `grep "^val_loss:" run.log`
9. Log to `results/results.tsv` using `flock` (see Shared results log section)
10. Mark claim as `done` in `results/claims.tsv`
11. If `keep`: continue building on this. If `discard`: `git revert HEAD` and try something else.
11. Repeat indefinitely.

## Ideas to explore

**Hyperparameter tuning alone will NOT win.** The current best is ~1.28 — to break through, you need structural changes. But **don't rewrite entire components** — make small, surgical tweaks to the existing architecture. One change at a time.

### Small architectural tweaks (HIGH PRIORITY — one at a time!)
- Add a learnable scalar residual weight per block
- Change normalization placement (pre-norm vs post-norm vs sandwich)
- Try different activation in GEGLU (GeLU, ReLU², Mish)
- Add a single extra skip connection (e.g. input residual to output)
- Scale attention logits differently (e.g. 1/sqrt(d) vs learned scale)
- Change expansion ratio in FFN (4x → 3x or 6x)
- Wider model (d_model=512) with fewer blocks instead of deep+narrow
- Remove register tokens and see if it helps
- Different patch conv kernel sizes (3 instead of 5)
- Add dropout or stochastic depth to blocks

### Training tweaks (HIGH PRIORITY)
- Different noise schedules (cosine beta schedule, shifted logit-normal)
- Loss weighting by noise level (min-SNR but with unweighted val)
- Larger batch with gradient accumulation (if more steps per hour)
- Progressive context: start with fewer frames, increase during training
- Mixed precision strategies

### Hard constraints
- **N_WINDOW must be >= 30** — shorter context windows hurt AR quality. Do not reduce below 30.

### Best practices
- **Aim for high MFU (Model FLOPs Utilization)** — use most of the available 48 GB GPU memory. If your run uses <10 GB, you're leaving performance on the table. Scale up batch size, model width, depth, or context window to fill the GPU.
- **Default to ONE small change per experiment**, measure, keep or revert
- Larger rewrites are okay occasionally but start with small tweaks first — they're faster to iterate and easier to debug
- If a big change fails, try extracting the smallest piece of it that might help
- **Prototype big changes with TIME_BUDGET=600 (10 min)** first — faster iteration. Once it works, scale up to 3600.
- Big architectural changes often have bugs on the first attempt. If you have **high conviction** an idea should work (e.g. it's well-established in the literature), **debug it** before discarding — the first failure may indicate a bug, not a bad idea. To debug: look for the biggest problems in your implementation first (wrong shapes, broken gradient flow, numerical issues, missing normalization) and fix those. Adding temporary instrumentation (gradient norms, activation stats, per-layer losses) can help diagnose issues.

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
