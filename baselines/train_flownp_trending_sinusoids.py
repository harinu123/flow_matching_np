"""Train FlowNP (ported) on Informed Meta-Learning trending-sinusoids.

This script is intentionally self-contained and does NOT modify the original INP model code.
It uses the INP dataset + collate_fn, then converts batches into FlowNP's expected format.

Notes:
- FlowNP expects targets (xt, yt) to EXCLUDE context points.
  The INP dataloader returns x_target/y_target containing all points; `extras["context_idx"]`
  indicates which were used as context.
- Likelihood computation (tar_ll) in eval mode requires optional dependency `flow_matching`.
  Training + sampling do not.
"""

import argparse
from types import SimpleNamespace

import torch
import numpy as np

from dataset.utils import setup_dataloaders
from baselines.flow_np.fnp import FNP, AttrDict


def _set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _complement_targets(x_all, y_all, context_idx):
    """Return (xt, yt) that exclude context indices.

    x_all: [B, N, Dx]
    y_all: [B, N, Dy]
    context_idx: [B, Nc, 1] (indices into N)
    """
    B, N, _ = x_all.shape
    idx = context_idx.squeeze(-1)  # [B, Nc]
    mask = torch.ones((B, N), dtype=torch.bool, device=x_all.device)
    mask.scatter_(1, idx, False)

    # ragged -> stack (Nc varies across batches but fixed within a batch)
    xt = torch.stack([x_all[i][mask[i]] for i in range(B)], dim=0)
    yt = torch.stack([y_all[i][mask[i]] for i in range(B)], dim=0)
    return xt, yt


def _augment_x_with_knowledge(x, knowledge):
    """Concatenate numeric knowledge to x at each point.

    x: [B, N, Dx]
    knowledge: [B, K] or [B] (numeric)
    """
    if knowledge.dim() == 1:
        knowledge = knowledge.unsqueeze(-1)
    k = knowledge.unsqueeze(1).expand(-1, x.shape[1], -1)
    return torch.cat([x, k], dim=-1)


@torch.no_grad()
def quick_eval(model, val_loader, device, use_knowledge=False, max_batches=20):
    model.eval()
    losses = []
    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break
        (xc, yc), (x_all, y_all), knowledge, extras = batch
        xc, yc, x_all, y_all = xc.to(device), yc.to(device), x_all.to(device), y_all.to(device)
        context_idx = extras["context_idx"].to(device)

        xt, yt = _complement_targets(x_all, y_all, context_idx)

        if use_knowledge:
            if not torch.is_tensor(knowledge):
                raise ValueError("knowledge_type produced non-tensor knowledge; use numeric knowledge_type (e.g. full/abc/b/min_max/... ).")
            knowledge = knowledge.to(device)
            xc_in = _augment_x_with_knowledge(xc, knowledge)
            xt_in = _augment_x_with_knowledge(xt, knowledge)
        else:
            xc_in, xt_in = xc, xt

        out = model(AttrDict(xc=xc_in, yc=yc, xt=xt_in, yt=yt))
        losses.append(out.loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="set-trending-sinusoids",
                   choices=["set-trending-sinusoids", "set-trending-sinusoids-dist-shift"])
    p.add_argument("--knowledge_type", type=str, default="full")
    p.add_argument("--use_knowledge", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # data / batching
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--min_num_context", type=int, default=3)
    p.add_argument("--max_num_context", type=int, default=15)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--x_sampler", type=str, default="uniform")
    p.add_argument("--noise", type=float, default=0.0)

    # model
    p.add_argument("--dim_posenc", type=int, default=16)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--emb_depth", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--timesteps", type=int, default=100)

    # optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)

    args = p.parse_args()
    _set_seed(args.seed)

    # Build a config object compatible with dataset.utils
    cfg = SimpleNamespace(
        dataset=args.dataset,
        knowledge_type=args.knowledge_type,
        batch_size=args.batch_size,
        min_num_context=args.min_num_context,
        max_num_context=args.max_num_context,
        num_samples=args.num_samples,
        noise=args.noise,
        x_sampler=args.x_sampler,
    )
    train_loader, val_loader, test_loader, _ = setup_dataloaders(cfg)

    # dim_x: 1 (time) (+ knowledge_dim if used)
    dim_x = 1
    knowledge_dim = 0
    if args.use_knowledge:
        # Infer numeric knowledge dimensionality from one batch
        b0 = next(iter(train_loader))
        knowledge0 = b0[2]
        if not torch.is_tensor(knowledge0):
            raise ValueError("knowledge_type produced non-tensor knowledge; use numeric knowledge_type.")
        if knowledge0.dim() == 1:
            knowledge_dim = 1
        else:
            knowledge_dim = int(knowledge0.shape[-1])
        dim_x = dim_x + knowledge_dim

    model = FNP(
        dim_x=dim_x,
        dim_y=1,
        dim_posenc=args.dim_posenc,
        d_model=args.d_model,
        emb_depth=args.emb_depth,
        dim_feedforward=args.dim_feedforward,
        nhead=args.nhead,
        dropout=args.dropout,
        num_layers=args.num_layers,
        timesteps=args.timesteps,
    ).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for (xc, yc), (x_all, y_all), knowledge, extras in train_loader:
            xc, yc = xc.to(args.device), yc.to(args.device)
            x_all, y_all = x_all.to(args.device), y_all.to(args.device)
            context_idx = extras["context_idx"].to(args.device)

            xt, yt = _complement_targets(x_all, y_all, context_idx)

            if args.use_knowledge:
                knowledge = knowledge.to(args.device)
                xc_in = _augment_x_with_knowledge(xc, knowledge)
                xt_in = _augment_x_with_knowledge(xt, knowledge)
            else:
                xc_in, xt_in = xc, xt

            out = model(AttrDict(xc=xc_in, yc=yc, xt=xt_in, yt=yt))
            loss = out.loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % args.log_every == 0:
                print(f"[epoch {epoch:03d} step {step:06d}] loss={loss.item():.6f}")
            step += 1

        val_loss = quick_eval(model, val_loader, args.device, use_knowledge=args.use_knowledge, max_batches=20)
        print(f"[epoch {epoch:03d}] val_loss={val_loss:.6f}")

    # Optional: save
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
    }, "flownp_trending_sinusoids.pt")
    print("Saved: flownp_trending_sinusoids.pt")


if __name__ == "__main__":
    main()
