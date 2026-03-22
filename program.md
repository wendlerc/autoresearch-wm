# World Model Research Program

You are an autonomous AI researcher optimizing a Doom world model. Your goal is to minimize `val_loss` (validation MSE on held-out latent clips).

## Setup

1. Create a new branch for your experiment
2. Read `train.py` to understand the current architecture and training setup
3. Verify data exists: `ls ~/.cache/autoresearch-wm/data/latent-*.tar`
4. Initialize `results.tsv` if it doesn't exist

## Rules

- **You may only modify `train.py`**. Do not modify `prepare.py` or dependencies.
- The training budget is **10 minutes** (600 seconds). Exceeding this means failure.
- Validation runs after training and does not count toward the budget.
- Each experiment must produce a structured output block ending with `val_loss:` and other metrics.

## Experiment loop

1. Make a change to `train.py` (architecture, hyperparameters, training procedure, etc.)
2. Commit with a descriptive message
3. Run: `uv run train.py 2>&1 | tee run.log`
4. Extract results: `grep "^val_loss:" run.log`
5. Log to `results.tsv`: `commit\tval_loss\tmemory_gb\tstatus\tdescription`
   - status: `keep` (improvement), `discard` (regression), or `crash`
6. If `keep`: continue from this point. If `discard`: revert and try something else.
7. Repeat indefinitely.

## Ideas to explore

- Model size (depth, width, heads)
- Learning rate schedules
- Patch size and tokenization
- Attention patterns
- Noise schedules (currently logit-normal)
- Action embedding design
- Data augmentation
- Optimizer hyperparameters
- Context window length
