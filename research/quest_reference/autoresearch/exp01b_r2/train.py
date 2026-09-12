"""The only file an EXP-01B-R2 autoresearch agent may modify.

Candidate scope includes model architecture, loss, augmentation, optimizer,
batch size, and schedule.  The fixed runner supplies the data, seed, device, and
maximum epoch/time budget.  The returned model must map normalized RGB tensors
of shape ``[N, 3, 64, 64]`` to finite logits of shape ``[N, 2, 16, 16]``.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class CandidateFieldModel(nn.Module):
    """Frozen EXP-01B ResNet-18 baseline, moved here as the editable candidate."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.lateral4 = nn.Conv2d(512, 64, 1)
        self.lateral3 = nn.Conv2d(256, 64, 1)
        self.lateral2 = nn.Conv2d(128, 64, 1)
        self.lateral1 = nn.Conv2d(64, 64, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(self.stem(rgb))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        value = self.lateral4(x4)
        value = F.interpolate(value, x3.shape[-2:], mode="bilinear", align_corners=False) + self.lateral3(x3)
        value = F.interpolate(value, x2.shape[-2:], mode="bilinear", align_corners=False) + self.lateral2(x2)
        value = F.interpolate(value, x1.shape[-2:], mode="bilinear", align_corners=False) + self.lateral1(x1)
        return self.refine(value)


def build_candidate(seed: int) -> nn.Module:
    """Construct one candidate deterministically for the supplied model seed."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return CandidateFieldModel()


def field_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Baseline BCE + soft Dice loss."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    total = bce + dice
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach())}


def _rgb_tensor(scenes: Sequence[Any]) -> torch.Tensor:
    return torch.tensor(np.asarray([item.rgb for item in scenes]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def _target_tensor(scenes: Sequence[Any]) -> torch.Tensor:
    return torch.tensor(np.asarray([item.field_target for item in scenes]), dtype=torch.float32)


def _augment_palette(
    rgb: torch.Tensor,
    *,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Randomize color and luminance polarity without changing terrain layout."""
    count = len(rgb)
    grayscale = rgb.mean(dim=1, keepdim=True)
    saturation = torch.tensor(
        rng.uniform(0.0, 1.4, size=(count, 1, 1, 1)),
        dtype=rgb.dtype,
        device=rgb.device,
    )
    value = grayscale + saturation * (rgb - grayscale)

    polarity = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    polarity = torch.tensor(
        polarity[:, None, None, None],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    value = 0.5 + polarity * (value - 0.5)
    mean = value.mean(dim=(-2, -1), keepdim=True)
    contrast = torch.tensor(
        rng.uniform(0.65, 1.35, size=(count, 1, 1, 1)),
        dtype=rgb.dtype,
        device=rgb.device,
    )
    gains = torch.tensor(
        rng.uniform(0.7, 1.3, size=(count, 3, 1, 1)),
        dtype=rgb.dtype,
        device=rgb.device,
    )
    brightness = torch.tensor(
        rng.uniform(-0.15, 0.15, size=(count, 1, 1, 1)),
        dtype=rgb.dtype,
        device=rgb.device,
    )
    value = (value - mean) * contrast + mean
    value = value * gains + brightness

    permutations = np.tile(np.arange(3), (count, 1))
    for index in range(count):
        if rng.random() < 0.4:
            permutations[index] = rng.permutation(3)
    channels = torch.tensor(permutations, dtype=torch.long, device=rgb.device)
    channels = channels[:, :, None, None].expand(-1, -1, rgb.shape[-2], rgb.shape[-1])
    return value.gather(1, channels).clamp_(0.0, 1.0)


def _validation_loss(
    model: nn.Module,
    rgb: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rgb), batch_size):
            logits = model(rgb[start:start + batch_size].to(device, non_blocking=True))
            loss, _ = field_loss(logits, target[start:start + batch_size].to(device, non_blocking=True))
            values.append(float(loss))
    return float(np.mean(values))


def train_candidate(
    train_scenes: Sequence[Any],
    validation_scenes: Sequence[Any],
    *,
    seed: int,
    device: torch.device,
    budget: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any]]:
    """Train the candidate within the fixed runner-supplied budget."""
    started = time.monotonic()
    max_epochs = int(budget["max_epochs"])
    max_seconds = float(budget["max_train_seconds_per_seed"])
    batch_size = 20
    patience = min(6, max_epochs)

    model = build_candidate(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_rgb, train_target = _rgb_tensor(train_scenes), _target_tensor(train_scenes)
    validation_rgb, validation_target = _rgb_tensor(validation_scenes), _target_tensor(validation_scenes)
    rng = np.random.default_rng(int(seed))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss, stale = math.inf, 0
    history: list[dict[str, Any]] = []

    for epoch in range(max_epochs):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError(f"candidate exceeded {max_seconds:.0f}s training budget")
        model.train()
        order = rng.permutation(len(train_scenes))
        losses = []
        components = []
        for start in range(0, len(order), batch_size):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError(f"candidate exceeded {max_seconds:.0f}s training budget")
            index = torch.tensor(order[start:start + batch_size], dtype=torch.long)
            rgb = train_rgb[index].to(device, non_blocking=True)
            target = train_target[index].to(device, non_blocking=True)
            rgb = _augment_palette(rgb, rng=rng)
            logits = model(rgb)
            loss, detail = field_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            components.append(detail)

        validation_loss = _validation_loss(
            model,
            validation_rgb,
            validation_target,
            device=device,
            batch_size=64,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "bce": float(np.mean([item["bce"] for item in components])),
            "dice": float(np.mean([item["dice"] for item in components])),
        }
        history.append(row)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state, strict=True)
    with torch.no_grad():
        model.refine[-1].bias.sub_(2.0)
    model.to(device).eval()
    return model, {
        "epochs": len(history),
        "best_validation_loss": float(best_loss),
        "batch_size": batch_size,
        "post_training_logit_shift": -2.0,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
