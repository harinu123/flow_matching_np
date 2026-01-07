# FlowNP port inside aa_informed-meta-learning-main

This repo includes a **ported** copy of the FlowNP (Flow Matching Neural Process) model under:

- `baselines/flow_np/`

The original INP code under `models/` is **untouched**.

## What's included

- `baselines/flow_np/attention.py`, `modules.py`, `fnp.py` — copied from FlowNP and patched to:
  - avoid name collisions with this repo's `models/`
  - remove the `attrdictionary` dependency (tiny local `AttrDict` provided)
  - make `flow_matching` optional (only needed for likelihood computation in eval mode)

- `baselines/train_flownp_trending_sinusoids.py` — standalone trainer that:
  - uses the INP sinusoids dataloader (`dataset/utils.py`)
  - converts batches into FlowNP's expected `(xc,yc,xt,yt)` format
  - optionally concatenates numeric knowledge onto x (`--use_knowledge`)

## Running

From repo root:

```bash
python baselines/train_flownp_trending_sinusoids.py \
  --dataset set-trending-sinusoids \
  --knowledge_type full \
  --use_knowledge \
  --batch_size 16 --epochs 50
```

If you want *no knowledge*:

```bash
python baselines/train_flownp_trending_sinusoids.py --dataset set-trending-sinusoids --batch_size 16
```

## Notes

- The INP collate function returns `x_target/y_target` containing *all* points, plus `extras["context_idx"]`.
  The trainer uses this to build FlowNP targets as the **complement** of the context indices.
- If you call `model.eval()` and forward, FlowNP's likelihood requires the optional `flow_matching` package.
  Training + sampling do not.
