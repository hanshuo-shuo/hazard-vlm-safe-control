"""
Minimal conditional DDPM for vector data.

We model x0 = [copilot_obs, action] as a single vector.
Conditioning uses the first `cond_dim` dimensions:
- forward diffusion: optionally remove noise from cond dims (x_t[:cond_dim] = x0[:cond_dim])
- training loss: only on non-conditional dims (recommended; avoids impossible noise prediction)
- reverse diffusion: enforce condition each step by overwriting x[:cond_dim] = cond (naive conditioning)

This file is intentionally self-contained and avoids wandb/params_proto/etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager


def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Extract values from a 1-D tensor `a` at indices `t` and reshape for broadcast to `x`.
    a: (T,)
    t: (B,)
    x: (B, D)
    returns: (B, 1)
    """
    out = a.gather(0, t.long())
    while out.ndim < x.ndim:
        out = out.unsqueeze(-1)
    return out


def make_beta_schedule(
    schedule: Literal["linear", "sigmoid"], n_timesteps: int, start: float, end: float
) -> torch.Tensor:
    if n_timesteps <= 1:
        raise ValueError("n_timesteps must be > 1")
    start = float(start)
    end = float(end)
    if schedule == "linear":
        betas = torch.linspace(start, end, n_timesteps, dtype=torch.float32)
    elif schedule == "sigmoid":
        # common simple sigmoid schedule
        steps = torch.linspace(-6, 6, n_timesteps, dtype=torch.float32)
        betas = torch.sigmoid(steps) * (end - start) + start
    else:
        raise ValueError(f"Unknown beta schedule: {schedule!r}")
    return betas.clamp(1e-8, 0.999)


class TimeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_steps: int):
        super().__init__()
        self.num_steps = int(num_steps)
        self.t_embed = nn.Embedding(self.num_steps, hidden_dim)
        self.fc1 = nn.Linear(int(input_dim) + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(input_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B,)
        te = self.t_embed(t.long())
        h = torch.cat([x, te], dim=-1)
        h = F.silu(self.fc1(h))
        h = F.silu(self.fc2(h))
        return self.fc3(h)


@dataclass
class RunningNorm:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def from_data(x: np.ndarray, eps: float = 1e-6) -> "RunningNorm":
        xt = torch.as_tensor(x, dtype=torch.float32)
        mean = xt.mean(dim=0)
        std = xt.std(dim=0).clamp_min(float(eps))
        return RunningNorm(mean=mean, std=std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class EMA:
    def __init__(self, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module) -> None:
        self.shadow = {}
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        if not self.shadow:
            self.register(module)
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name].data)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}


@dataclass(frozen=True)
class DiffusionConfig:
    x_dim: int
    cond_dim: int
    num_steps: int = 50
    beta_schedule: Literal["linear", "sigmoid"] = "sigmoid"
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 256
    lr: float = 1e-3
    ema_decay: float = 0.995


class ConditionalDDPM:
    def __init__(self, cfg: DiffusionConfig, *, device: torch.device, norm: RunningNorm | None = None):
        self.cfg = cfg
        self.device = device

        betas = make_beta_schedule(cfg.beta_schedule, cfg.num_steps, cfg.beta_min, cfg.beta_max).to(device)
        self.betas = betas
        self.alphas = (1.0 - betas).clamp(1e-8, 1.0)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.model = TimeMLP(input_dim=cfg.x_dim, hidden_dim=cfg.hidden_dim, num_steps=cfg.num_steps).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=float(cfg.lr))
        self.ema = EMA(cfg.ema_decay)
        self.ema.register(self.model)

        self.norm = norm  # optional per-dimension standardization for x0

    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return x
        return self.norm.normalize(x)

    def _maybe_denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None:
            return x
        return self.norm.denormalize(x)

    @contextmanager
    def ema_scope(self):
        """
        Temporarily copy EMA weights into the model for sampling.
        """
        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        try:
            if self.ema.shadow:
                # copy EMA shadow to params
                self.ema.copy_to(self.model)
            yield
        finally:
            self.model.load_state_dict(backup)

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        plus optional: keep condition dims noise-free in x_t.
        """
        x0 = x0.to(self.device).float()
        t = t.to(self.device).long()
        x0n = self._maybe_norm(x0)

        eps = torch.randn_like(x0n)
        a = _extract(self.sqrt_alpha_bars, t, x0n)
        am1 = _extract(self.sqrt_one_minus_alpha_bars, t, x0n)
        xt = a * x0n + am1 * eps

        if int(self.cfg.cond_dim) > 0:
            cd = int(self.cfg.cond_dim)
            xt[..., :cd] = x0n[..., :cd]
            # target noise is undefined for cond dims when we overwrite x_t; set to zero and mask in loss.
            eps[..., :cd] = 0.0

        return xt, eps

    @torch.no_grad()
    def q_sample_at(self, x0: torch.Tensor, *, t_int: int, eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper for forward diffusion at a single integer timestep.
        eps can be provided for determinism; otherwise sampled from N(0, I).
        """
        b = int(x0.shape[0])
        t = torch.full((b,), int(t_int), device=self.device, dtype=torch.long)
        x0 = x0.to(self.device).float()
        x0n = self._maybe_norm(x0)

        if eps is None:
            eps = torch.randn_like(x0n)
        else:
            eps = eps.to(self.device).float()

        a = _extract(self.sqrt_alpha_bars, t, x0n)
        am1 = _extract(self.sqrt_one_minus_alpha_bars, t, x0n)
        xt = a * x0n + am1 * eps

        if int(self.cfg.cond_dim) > 0:
            cd = int(self.cfg.cond_dim)
            xt[..., :cd] = x0n[..., :cd]
            eps[..., :cd] = 0.0

        return xt, eps

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        b = int(x0.shape[0])
        t = torch.randint(0, int(self.cfg.num_steps), (b,), device=self.device)
        xt, eps = self.q_sample(x0, t)
        pred = self.model(xt, t)
        err = (eps - pred)

        if int(self.cfg.cond_dim) > 0:
            cd = int(self.cfg.cond_dim)
            err = err[..., cd:]

        return (err * err).mean()

    def train_step(self, x0: torch.Tensor) -> float:
        self.model.train()
        loss = self.loss(x0)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        self.ema.update(self.model)
        return float(loss.detach().cpu().item())

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t_int: int, *, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Reverse step using epsilon prediction:
          x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-a_bar_t) * eps_theta) + sigma_t * z
        """
        self.model.eval()
        t = torch.full((x.shape[0],), int(t_int), device=self.device, dtype=torch.long)
        x = x.to(self.device).float()

        if cond is not None and int(self.cfg.cond_dim) > 0:
            cd = int(self.cfg.cond_dim)
            x[..., :cd] = cond[..., :cd]

        eps_theta = self.model(x, t)

        beta_t = _extract(self.betas, t, x)
        alpha_t = _extract(self.alphas, t, x)
        a_bar_t = _extract(self.alpha_bars, t, x)

        coef = (1.0 - alpha_t) / torch.sqrt((1.0 - a_bar_t).clamp_min(1e-8))
        mean = (1.0 / torch.sqrt(alpha_t.clamp_min(1e-8))) * (x - coef * eps_theta)
        mean = mean.clamp(-10.0, 10.0)  # prevent numerical explosion during reverse chain

        if t_int == 0:
            out = mean
        else:
            z = torch.randn_like(x)
            sigma = torch.sqrt(beta_t.clamp_min(1e-8))
            out = mean + sigma * z
            out = out.clamp(-10.0, 10.0)

        if cond is not None and int(self.cfg.cond_dim) > 0:
            cd = int(self.cfg.cond_dim)
            out[..., :cd] = cond[..., :cd]

        return out

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x0 in *data space* (denormalized if norm provided).
        If cond is provided, it should be in data space and shape (B, cond_dim).
        """
        b = int(batch_size)
        x = torch.randn((b, int(self.cfg.x_dim)), device=self.device, dtype=torch.float32)

        cond_n = None
        if cond is not None:
            cond = cond.to(self.device).float()
            # Build a full vector to normalize consistently
            if self.norm is not None:
                # normalize only the condition part using stored mean/std
                cd = int(self.cfg.cond_dim)
                cond_n = (cond - self.norm.mean[:cd].to(self.device)) / self.norm.std[:cd].to(self.device)
            else:
                cond_n = cond

            # overwrite at start too
            if int(self.cfg.cond_dim) > 0:
                x[..., : int(self.cfg.cond_dim)] = cond_n

        for t in reversed(range(int(self.cfg.num_steps))):
            x = self.p_sample(x, t, cond=cond_n)

        # x is in normalized space if norm is used
        x0n = x
        x0 = self._maybe_denorm(x0n)
        return x0

    @torch.no_grad()
    def p_sample_loop_from(
        self,
        x_t: torch.Tensor,
        *,
        start_t: int,
        cond: torch.Tensor | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion from a provided x_t at timestep start_t down to x_0.
        `cond` is in data space shape (B, cond_dim).
        """
        x = x_t.to(self.device).float()

        cond_n = None
        if cond is not None:
            cond = cond.to(self.device).float()
            if self.norm is not None and int(self.cfg.cond_dim) > 0:
                cd = int(self.cfg.cond_dim)
                cond_n = (cond - self.norm.mean[:cd].to(self.device)) / self.norm.std[:cd].to(self.device)
            else:
                cond_n = cond

        ctx = self.ema_scope() if use_ema else contextmanager(lambda: (yield))()
        with ctx:
            for t in reversed(range(int(start_t) + 1)):
                x = self.p_sample(x, t, cond=cond_n)

        # x is in normalized space if norm used
        x0 = self._maybe_denorm(x)
        return x0

    @torch.no_grad()
    def refine_action(
        self,
        *,
        copilot_obs: torch.Tensor,
        pilot_action: torch.Tensor,
        k: int,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        VLM-guided control primitive:
        - build x0=[copilot_obs, pilot_action] in data space
        - forward diffuse to timestep k (keeping condition dims fixed)
        - reverse sample back to x0_hat
        - return action part from x0_hat

        Shapes:
        - copilot_obs: (B, cond_dim)
        - pilot_action: (B, act_dim)
        returns: (B, act_dim)
        """
        cd = int(self.cfg.cond_dim)
        x_dim = int(self.cfg.x_dim)
        if int(copilot_obs.shape[-1]) != cd:
            raise ValueError(f"copilot_obs dim {int(copilot_obs.shape[-1])} != cond_dim {cd}")

        x0 = torch.cat([copilot_obs, pilot_action], dim=-1).to(self.device).float()
        if int(x0.shape[-1]) != x_dim:
            raise ValueError(f"x0 dim {int(x0.shape[-1])} != x_dim {x_dim}")

        k = int(max(0, min(int(self.cfg.num_steps) - 1, int(k))))
        if k == 0:
            # no refine: just return pilot action
            return pilot_action.to(self.device).float()

        xk, _eps = self.q_sample_at(x0, t_int=k, eps=None)
        x0_hat = self.p_sample_loop_from(xk, start_t=k, cond=copilot_obs, use_ema=use_ema)
        act_dim = int(x_dim - cd)
        return x0_hat[..., cd : cd + act_dim]

    def state_dict(self) -> dict:
        return {
            "cfg": vars(self.cfg),
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "ema": self.ema.state_dict(),
            "norm": None
            if self.norm is None
            else {
                "mean": self.norm.mean.detach().cpu(),
                "std": self.norm.std.detach().cpu(),
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model"])
        if "optim" in sd and sd["optim"] is not None:
            self.optim.load_state_dict(sd["optim"])
        if "ema" in sd and sd["ema"] is not None:
            self.ema.load_state_dict(sd["ema"])
        if sd.get("norm") is not None:
            mean = sd["norm"]["mean"].float()
            std = sd["norm"]["std"].float()
            self.norm = RunningNorm(mean=mean, std=std)
