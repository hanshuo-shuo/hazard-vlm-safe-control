"""Optional small dense student. Inference consumes RGB only."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialRequirementNet(nn.Module):
    revision = "small-cnn-spatial-v1"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,5,padding=2), nn.ReLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(24,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,3,1),
        )

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("expected BCHW RGB tensor")
        logits = self.encoder(rgb)
        return F.interpolate(logits,size=rgb.shape[-2:],mode="bilinear",align_corners=False)

    @torch.no_grad()
    def predict_field(self, rgb):
        return self(rgb).sigmoid()
