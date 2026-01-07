#!/usr/bin/env python3
"""
Train FlowNP (Flow Neural Process) on the IML "set-trending-sinusoids-dist-shift" dataset,
saving checkpoints in the same ./saves/<project>/<run_prefix>_<N>/ format used by this repo.

This script is intentionally separate from models/train.py to keep the original INP training untouched.

Example:
  python baselines/train_flownp_trending_sinusoids_dist_shift.py \
    --project-name INPs_sinusoids \
    --run-name-prefix flownp_dist_shift \
    --dataset set-trending-sinusoids-dist-shift \
    --knowledge-type b \
    --use-knowledge True \
    --batch-size 64 \
    --num-epochs 200 \
    --lr 1e-4 \
    --noise 0.2 \
    --min-num-context 0 \
    --max-num-context 10 \
    --num-targets 100 \
    --test-num-z-samples 32 \
    --seed 0 \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch

# Add repo root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from config import Config
from dataset.utils import setup_dataloaders
from models.flownp import FlowNP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Save/run bookkeeping
    p.add_argument("--project-name", type=str, default="INPs_sinusoids")
    p.add_argument("--run-name-prefix", type=str, default="flownp_dist_shift")
    p.add_argument("--seed", type=int, default=0)

    # Data
    p.add_argument("--dataset", type=str, default="set-trending-sinusoids-dist-shift")
    p.add_argument("--knowledge-type", type=str, default="b")
    p.add_argument("--use-knowledge", type=lambda x: str(x).lower() in ["1", "true", "yes"], default=True)
    p.add_argument("--min-num-context", type=int, default=0)
    p.add_argument("--max-num-context", type=int, default=10)
    p.add_argument("--num-targets", type=int, default=100)
    p.add_argument("--x-sampler", type=str, default="uniform")

    # Model IO dims (sinusoids are 1->1)
    p.add_argument("--input-dim", type=int, default=1)
    p.add_argument("--output-dim", type=int, default=1)

    # Optimization
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # FlowNP evaluation noise (used for likelihood in evaluation)
    p.add_argument("--noise", type=float, default=0.2)

    # Number of samples used by evaluation notebook (mixture components)
    p.add_argument("--test-num-z-samples", type=int, default=32)

    # Device
    p.add_argument("--device", type=str, default=None)

    # FlowNP hyperparameters (optional)
    p.add_argument("--flownp-dim-posenc", type=int, default=20)
    p.add_argument("--flownp-d-model", type=int, default=64)
    p.add_argument("--flownp-emb-depth", type=int, default=4)
    p.add_argument("--flownp-dim-feedforward", type=int, default=128)
    p.add_argument("--flownp-nhead", type=int, default=4)
    p.add_argument("--flownp-dropout", type=float, default=0.0)
    p.add_argument("--flownp-num-layers", type=int, default=6)
    p.add_argument("--flownp-timesteps", type=int, default=100)

    return p.parse_args()


def pick_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_save_dir(project_name: str, run_prefix: str) -> str:
    base = f"./saves/{project_name}"
    os.makedirs(base, exist_ok=True)

    existing = []
    for x in os.listdir(base):
        if x.startswith(run_prefix + "_"):
            try:
                existing.append(int(x.split("_")[-1]))
            except ValueError:
                pass
    save_no = (max(existing) + 1) if len(existing) > 0 else 0
    save_dir = f"{base}/{run_prefix}_{save_no}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


@torch.no_grad()
def eval_epoch(model: FlowNP, data_loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in data_loader:
        context, target, knowledge, extras = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        loss = model.flow_matching_loss(x_context, y_context, x_target, y_target)
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses)) if len(losses) else float("inf")


def main():
    args = parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = pick_device(args.device)
    print(f"[FlowNP] Using device: {device}")

    # Build config (stored for evaluation loader)
    config = Config(**vars(args))
    config.model_type = "flownp"
    config.device = str(device)

    save_dir = make_save_dir(config.project_name, config.run_name_prefix)
    config.write_config(f"{save_dir}/config.toml")
    print(f"[FlowNP] Saving to: {save_dir}")

    # Data
    train_loader, val_loader, test_loader, extras = setup_dataloaders(config)

    # Model
    model = FlowNP(config).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val = float("inf")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            context, target, knowledge, extras = batch
            x_context, y_context = context
            x_target, y_target = target
            x_context = x_context.to(device)
            y_context = y_context.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            opt.zero_grad(set_to_none=True)
            loss = model.flow_matching_loss(x_context, y_context, x_target, y_target)
            loss.backward()

            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            opt.step()
            epoch_losses.append(loss.detach().cpu().item())

        train_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else float("inf")
        val_loss = eval_epoch(model, val_loader, device)

        print(f"[FlowNP] epoch {epoch+1:04d}/{config.num_epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{save_dir}/model_best.pt")

    # Save last
    torch.save(model.state_dict(), f"{save_dir}/model_last.pt")
    print(f"[FlowNP] Done. best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
