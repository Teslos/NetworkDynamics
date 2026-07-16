from __future__ import annotations

from torch import Tensor, nn


class LinearReadout(nn.Module):
    """Linear softmax classifier for frozen reservoir features."""

    def __init__(self, feature_dim: int, n_classes: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if n_classes <= 1:
            raise ValueError("n_classes must be at least 2")
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, feature_dim]")
        return self.classifier(features)
