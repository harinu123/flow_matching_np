"""
FlowNP (Flow Neural Process) integration for the Informed Meta-Learning (INP) codebase.

This module provides:
- A lightweight re-implementation of the FlowNP "FNP" core model (inspired by flowNP-master/models/fnp.py),
  stripped of external dependencies (attrdictionary, flow_matching) while keeping:
    * flow-matching training loss (velocity matching MSE)
    * a discrete-time sampler for predictions

- A wrapper class `FlowNP` exposing the same forward signature as `models.inp.INP`
  so it can be evaluated with the existing evaluation utilities (NLL over samples).

Compatibility contract with this repo:
    model(x_context, y_context, x_target, y_target=None, knowledge=None)
        -> (p_yCc, z_samples, q_zCc, q_zCct)

For FlowNP:
- We treat each generated sample as a mixture component by returning a batched distribution
  whose batch dimension includes num_samples.
- We set q_zCc = q_zCct = None so the existing NLL falls back to the non-importance-weighted estimator.

Important:
- This implementation ignores "knowledge" at model level. This is intentional: FlowNP is compared as a
  model-free baseline under distribution shift.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


def build_mlp(dim_in: int, dim_hid: int, dim_out: int, depth: int) -> nn.Sequential:
    if depth < 2:
        raise ValueError("depth must be >= 2")
    layers = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth - 2):
        layers += [nn.Linear(dim_hid, dim_hid), nn.ReLU(True)]
    layers += [nn.Linear(dim_hid, dim_out)]
    return nn.Sequential(*layers)


def comp_posenc(dim_posenc: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Simple sinusoidal positional encoding over the last dimension of `pos`.
    Returns a flattened encoding.
    """
    if dim_posenc <= 0:
        raise ValueError("dim_posenc must be positive")

    d = pos.shape[-1]
    half = dim_posenc // 2
    if half == 0:
        return pos

    device = pos.device
    freqs = torch.pow(2.0, torch.arange(half, device=device).float())  # [half]
    angles = pos.unsqueeze(-1) * freqs  # [..., d, half]
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    enc = torch.cat([sin, cos], dim=-1)  # [..., d, 2*half]
    if dim_posenc % 2 == 1:
        enc = torch.cat([enc, pos.unsqueeze(-1)], dim=-1)  # [..., d, 2*half+1]
    return enc.reshape(*pos.shape[:-1], d * enc.shape[-1])


class _PredictVelocity(nn.Module):
    def __init__(self, encode_fn, dim_posenc: int):
        super().__init__()
        self._encode = encode_fn
        self._dim_posenc = dim_posenc

    def forward(self, x: torch.Tensor, t: torch.Tensor, batch: SimpleNamespace) -> torch.Tensor:
        # x is current state (yt at time t), flattened
        yt = x.reshape(batch.xt.shape[0], batch.xt.shape[1], -1)
        if t.dim() == 0:
            t = t.repeat((yt.shape[0], yt.shape[1], 1)).to(x.device)

        # context inputs get "1" time indicator; target gets random t
        xc = torch.cat(
            (batch.xc, torch.ones(list(batch.xc.shape[:-1]) + [1], device=x.device)),
            dim=-1,
        )
        xt = torch.cat((batch.xt, t), dim=-1)

        pred = self._encode(
            xc=comp_posenc(self._dim_posenc, xc),
            xt=comp_posenc(self._dim_posenc, xt),
            yc=batch.yc,
            yt=yt,
        )
        return pred.reshape(x.shape)


class FNP(nn.Module):
    """
    Minimal Flow Neural Process core:
    - Transformer encoder over concatenated (context tokens, target tokens)
    - Predicts a velocity field v_theta(y_t, t, x, context)
    - Trains with flow-matching MSE: E||v_theta - (y1 - y0)||^2
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_posenc: int = 20,
        d_model: int = 64,
        emb_depth: int = 4,
        dim_feedforward: int = 128,
        nhead: int = 4,
        dropout: float = 0.0,
        num_layers: int = 6,
        timesteps: int = 100,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.dim_posenc = int(dim_posenc)
        self.dim_x = int(dim_x)
        self.dim_y = int(dim_y)

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y),
        )

        # Each token sees: posenc(x, tflag) + y (either context y or evolving y_t)
        self.embedder = build_mlp((dim_x + 1) * dim_posenc + dim_y, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        def encode(xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
            x_y_ctx = torch.cat((xc, yc), dim=-1)
            x_y_tar = torch.cat((xt, yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar), dim=1)

            num_tar = xt.shape[1]
            embeddings = self.embedder(inp)
            encoded = self.encoder(embeddings)[:, -num_tar:]
            out = self.predictor(encoded)
            return out

        self.encode = encode
        self.predict_velocity = _PredictVelocity(self.encode, dim_posenc)

    def forward(self, batch: SimpleNamespace) -> SimpleNamespace:
        outs = SimpleNamespace()
        # Training loss
        y0 = torch.randn_like(batch.yt)
        t = torch.rand(size=(y0.shape[0], y0.shape[1], 1), device=y0.device)
        yt = t * batch.yt + (1 - t) * y0

        pred = self.predict_velocity(yt, t, batch)
        outs.loss = F.mse_loss(pred, batch.yt - y0)
        return outs

    @torch.no_grad()
    def predict(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        """
        Discrete-time Euler sampler to generate y samples at target x's.
        Returns: [num_samples, B, Nt, Dy]
        """
        B, Nt, Dy = xt.shape[0], xt.shape[1], yc.shape[2]
        Ns = int(num_samples)

        # Repeat per sample
        xc_rep = xc.repeat((Ns, 1, 1))
        yc_rep = yc.repeat((Ns, 1, 1))
        xt_all = torch.cat((xc_rep, xt.repeat((Ns, 1, 1))), dim=1)

        # Start from noise for *all* points (context + target), then slice targets out
        yt = torch.randn((Ns * B, xt_all.shape[1], Dy), device=xt.device)

        # Context positions get a fixed indicator time=1
        xct = torch.cat(
            (xc_rep, torch.ones((xc_rep.shape[0], xc_rep.shape[1], 1), device=xc_rep.device)),
            dim=-1,
        )

        T = self.timesteps
        for t in range(T):
            tt = torch.tensor(t / T, device=yt.device).repeat((yt.shape[0], yt.shape[1], 1))
            xtt = torch.cat((xt_all, tt), dim=-1)

            pred = self.encode(
                xc=comp_posenc(self.dim_posenc, xct),
                xt=comp_posenc(self.dim_posenc, xtt),
                yc=yc_rep,
                yt=yt,
            )

            # Small heuristic noise schedule (keeps sampling stochastic but stable)
            alpha = 1.0 + (t / T) * (1.0 - t / T)
            sigma = 0.2 * (t / T * (1.0 - t / T)) ** 0.5
            yt = yt + (alpha * pred + sigma * torch.randn_like(yt)) / T

        # Slice out the target portion
        y_tar = yt[:, xc_rep.shape[1]:].reshape(Ns, B, Nt, Dy)
        return y_tar


class FlowNP(nn.Module):
    """
    INP-compatible wrapper around FNP:
    - forward returns a batched distribution over target y
    - training uses flow-matching loss via flow_matching_loss()
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.core = FNP(
            dim_x=int(getattr(config, "input_dim", 1)),
            dim_y=int(getattr(config, "output_dim", 1)),
            dim_posenc=int(getattr(config, "flownp_dim_posenc", 20)),
            d_model=int(getattr(config, "flownp_d_model", 64)),
            emb_depth=int(getattr(config, "flownp_emb_depth", 4)),
            dim_feedforward=int(getattr(config, "flownp_dim_feedforward", 128)),
            nhead=int(getattr(config, "flownp_nhead", 4)),
            dropout=float(getattr(config, "flownp_dropout", 0.0)),
            num_layers=int(getattr(config, "flownp_num_layers", 6)),
            timesteps=int(getattr(config, "flownp_timesteps", 100)),
        )

        # Used only for evaluation likelihood (treats samples as mixture components)
        noise = float(getattr(config, "noise", 0.2))
        self.obs_noise = noise if noise > 0 else 0.2

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        knowledge: Optional[torch.Tensor] = None,
    ) -> Tuple[Independent, torch.Tensor, None, None]:
        num_samples = int(getattr(self.config, "test_num_z_samples", 32))
        y_samples = self.core.predict(x_context, y_context, x_target, num_samples=num_samples)

        scale = torch.full_like(y_samples, self.obs_noise)
        p_yCc = Independent(Normal(loc=y_samples, scale=scale), 1)

        # Placeholder latents for compatibility with NLL signature
        z_samples = torch.zeros((num_samples, x_target.shape[0], 1), device=x_target.device)
        return p_yCc, z_samples, None, None

    def flow_matching_loss(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:
        batch = SimpleNamespace(xc=x_context, yc=y_context, xt=x_target, yt=y_target)
        outs = self.core(batch)
        return outs.loss
