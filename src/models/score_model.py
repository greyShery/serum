"""
Watermark scoring models.

Two detector architectures are offered:

- ``WatermarkScoreModel``: latent-space detector (original SERUM detector).
  Takes 4 x 64 x 64 VAE latents as input. Requires the VAE encoder at
  inference time, which ties it to a specific diffusion model.

- ``PixelWatermarkScoreModel``: pixel-space detector (SERUM Innovation A).
  Takes 3 x 512 x 512 images as input and operates entirely without the VAE
  encoder. This unlocks the author's Limitations #1 in the original paper
  (``Pure pixel-level detection remains unexplored``) and makes the detector
  portable across architectures.

Both share the same forward/logit/predict interface so they can be swapped
by changing ``config.training.detector_type``.
"""

import glob
import os
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.utils import cleanup_cuda_memory


class WatermarkScoreModel(nn.Module):
    def __init__(self, config: Config):
        super(WatermarkScoreModel, self).__init__()
        self.config = config

        # Input: [B, latent_channels, height, width]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.config.shapes.latent_channels, 32, kernel_size=3, stride=1, padding=1),  # -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 128, 8, 8]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> [B, 256, 4, 4]
        )

        # Flatten size = 256 * 4 * 4 = 4096
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.

        Returns:
            Unscaled logits [B, 1] — do NOT apply sigmoid here.
            Use predict() for probabilities.
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x  # raw logits, no sigmoid

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability in [0, 1]."""
        return torch.sigmoid(self.forward(x))

    def save(self, filename: str = None, include_date: bool = True) -> str:
        """Save model weights to disk (see PixelWatermarkScoreModel.save)."""
        if filename is None:
            filename = self.config.watermark.score_model_file

        os.makedirs(self.config.watermark.model_dir, exist_ok=True)
        filepath = os.path.join(self.config.watermark.model_dir, filename)

        if include_date:
            timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}{timestamp}{ext}"

        torch.save(self.state_dict(), filepath)
        print(f"Watermark Score Model saved to: {filepath}")
        return filepath

    @staticmethod
    def load(config: Config, filename: str = None) -> 'WatermarkScoreModel':
        """Load latent detector from disk."""
        if filename is None:
            filename = config.watermark.score_model_file

        filepath = os.path.join(config.watermark.model_dir, filename)

        if not os.path.exists(filepath):
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"
            files = glob.glob(pattern)
            if files:
                filepath = max(files)
            else:
                raise FileNotFoundError(f"No Watermark Score Model found at {filepath}")

        data = torch.load(filepath, map_location=config.get_device())
        model = WatermarkScoreModel(config)
        model.load_state_dict(data)
        cleanup_cuda_memory()

        print(f"Watermark Score Model loaded from: {filepath}")
        return model


class PixelWatermarkScoreModel(nn.Module):
    """
    Pixel-space watermark detector (SERUM Innovation A).

    Operates on 3 x 512 x 512 images directly, removing the dependence on the
    diffusion model's VAE encoder. The architecture mirrors the latent-space
    detector (4 conv blocks + 2-layer MLP head) but uses stride-2 convolutions
    instead of max-pooling to better preserve high-frequency content — the
    high-frequency domain is exactly where the GaussMarker / SERUM watermarks
    live, and max-pooling was empirically found to attenuate it.

    Args:
        config: ``Config`` providing ``shapes.image_size`` (default 512) and
            ``shapes.latent_channels`` (only used for fallback input channels).

    Forward signature mirrors :class:`WatermarkScoreModel` so the same training
    loop can drive both detectors by switching ``config.training.detector_type``.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.image_size = getattr(config.shapes, 'image_size', 512)
        self.in_channels = 3

        # 4 conv stages, each halves spatial resolution via stride-2.
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 (after 4 stages)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final spatial size: image_size / 16 (e.g. 32 for 512).
        self._final_spatial = self.image_size // 16
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * self._final_spatial * self._final_spatial, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image batch [B, 3, H, W] in [-1, 1].

        Returns:
            Raw logits [B, 1]. Use ``predict`` for probabilities.
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def save(self, filename: str = None, include_date: bool = True) -> str:
        """Persist weights. Uses ``config.watermark.score_model_file`` by default."""
        if filename is None:
            filename = getattr(self.config.watermark, 'pixel_score_model_file',
                               self.config.watermark.score_model_file)

        os.makedirs(self.config.watermark.model_dir, exist_ok=True)
        filepath = os.path.join(self.config.watermark.model_dir, filename)

        if include_date:
            timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}{timestamp}{ext}"

        torch.save(self.state_dict(), filepath)
        print(f"Pixel Watermark Score Model saved to: {filepath}")
        return filepath

    @staticmethod
    def load(config: Config, filename: str = None) -> 'PixelWatermarkScoreModel':
        """Load pixel detector from disk."""
        if filename is None:
            filename = getattr(config.watermark, 'pixel_score_model_file',
                               config.watermark.score_model_file)

        filepath = os.path.join(config.watermark.model_dir, filename)

        if not os.path.exists(filepath):
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"
            files = glob.glob(pattern)
            if files:
                filepath = max(files)
            else:
                raise FileNotFoundError(
                    f"No Pixel Watermark Score Model found at {filepath}"
                )

        data = torch.load(filepath, map_location=config.get_device())
        model = PixelWatermarkScoreModel(config)
        model.load_state_dict(data)
        cleanup_cuda_memory()

        print(f"Pixel Watermark Score Model loaded from: {filepath}")
        return model


def build_score_model(config: Config) -> nn.Module:
    """
    Factory returning the detector specified by ``config.training.detector_type``.

    Defaults to the latent detector so existing configs keep working.
    """
    detector_type = getattr(config.training, 'detector_type', 'latent').lower()
    if detector_type == 'pixel':
        return PixelWatermarkScoreModel(config)
    if detector_type == 'latent':
        return WatermarkScoreModel(config)
    raise ValueError(
        f"Unknown detector_type: {detector_type!r}. Expected 'latent' or 'pixel'."
    )
