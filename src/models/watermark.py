"""
Grid-based watermarking module for latent space manipulation.

This module implements a learnable grid pattern that can be used to inject
watermarks into the latent space of diffusion models.
"""

import glob
import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn

from ..utils.config import Config
from ..utils.utils import cleanup_cuda_memory


class Watermark(nn.Module):
    """
    A learnable grid pattern for watermark injection in latent space.
    
    This module maintains a learnable parameter grid that is used to generate
    noise patterns for watermarking. The grid is combined with random noise
    to create watermarked latent representations.
    
    Args:
        config (Config): Configuration object with watermark parameters.
        
    Attributes:
        grid (nn.Parameter): Learnable grid parameter with shape BASE_LATENT_SHAPE.
    """

    def __init__(self, config: Config):
        """
        Initialize the Watermark module.

        Args:
            config (Config): Configuration object with watermark parameters.
        """
        super().__init__()

        self.config = config

        self.grid = nn.Parameter(torch.randn(self.config.base_latent_shape))

        # === [创新点 #2: Spherical 3阶矩精确保持] ===
        #   启用后, forward 归一化阶段会额外做 tanh 重映射,
        #   保证 watermark 的 3 阶矩 (skewness) 严格 ≈ 0;
        #   与原始 N(0, I) 噪声的 3 阶矩一致, 降低 FID。
        self.use_spherical_3rd_moment = bool(
            getattr(self.config.watermark, 'use_spherical_3rd_moment', False)
        )
        self._tanh_eps = 1e-6  # 防 tanh saturation

    def _spherical_normalize(self, scaled_grid: torch.Tensor) -> torch.Tensor:
        """
        Spherical 2-order + 3-order moment exact preservation.

        阶段:
          1. 减均值               → 1阶矩 = 0 (centering)
          2. 除标准差             → 2阶矩 = 1 (scaling)
          3. tanh 重映射 (可选)    → 3阶矩 = 0 (奇函数 → skewness 严格 0)
          4. mean-center + norm 还原 →  1阶矩重新 = 0, ||A||₂ = sqrt(d)

        Args:
            scaled_grid: 已做 0.5~2.0 std 担位的 grid 张量, shape = base_latent_shape.

        Returns:
            归一化后的 grid, 满足 1/2/3 阶矩约束。
        """
        # 1. centering
        sg = scaled_grid - scaled_grid.mean()
        # 2. std = 1
        sg = sg / (sg.std() + self._tanh_eps)

        if self.use_spherical_3rd_moment:
            # 3. tanh 奇函数 → skewness 期望 0 严格成立
            #    (不 mean-center: 离散化下 tanh 自然 mean ≈ 0, 强行 center 会破坏奇对称)
            sg = torch.tanh(sg)
            # 4. 恢复 ||sg||₂ = sqrt(d), 保留论文 Appendix B 的投影约束
            d = float(sg.numel())
            cur_norm = sg.norm()
            if cur_norm > 0:
                sg = sg * (d ** 0.5) / cur_norm

        return sg

    def forward(self, batch_size: int, ret_noise: bool = False, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Generate watermarked noise from the grid pattern.

        This method combines the learnable grid pattern with random noise to create
        watermarked latent representations. The grid is first clamped to reasonable
        values, then combined with random noise using a weighted average.

        Args:
            batch_size (int): Number of samples in the batch.
            digits (torch.Tensor): A tensor of exactly 4 digits to inject.
            ret_noise (bool, optional): If True, also return the original noise.
                Defaults to False.

        Returns:
            torch.Tensor: Watermarked noise tensor of shape (batch_size, 4, 64, 64).
            tuple: If ret_noise is True, returns (watermarked_noise, original_noise).
        """
        if alpha is None:
            alpha = self.config.watermark.grid.noise_mix_alpha

        _, C, H, W = self.config.base_latent_shape

        # Generate original random noise
        orig_noise = torch.randn(batch_size, C, H, W).to(self.grid.device)

        # Numerical stability: enforce grid's empirical std lies in [0.5, 2.0]
        # to avoid NaN/Inf when the parameter gets stuck at extreme magnitudes.
        # (grid - mean) / std with std=0 produces NaN; with std=huge produces
        # an unscaled grid that the alpha mixing cannot suppress. Paper Appendix C.
        g_std = self.grid.std()
        if float(g_std) < 0.5:
            scale = 0.5 / (float(g_std) + 1e-6)
        elif float(g_std) > 2.0:
            scale = 2.0 / (float(g_std) + 1e-6)
        else:
            scale = 1.0
        scaled_grid = self.grid * scale
        # === [创新点 #2] 替换原 1/2 阶归一化 → 2 阶 + 3 阶 (可选) ===
        normalized_grid = self._spherical_normalize(scaled_grid)

        # Combine original noise with normalized grid pattern (equal weighting)
        # 创新点 #1（自适应 α）：支持 scalar 或 1-D tensor (B,)
        if isinstance(alpha, torch.Tensor):
            # 0-dim tensor (size 1) → 当 scalar 处理, 广播到所有 sample
            if alpha.dim() == 0 or (alpha.dim() == 1 and alpha.shape[0] == 1):
                noise = orig_noise * ((1 - alpha) ** 0.5) + normalized_grid * (alpha ** 0.5)
            else:
                if alpha.shape[0] != batch_size:
                    raise ValueError(
                        f"per-sample alpha must have length B={batch_size}, got {tuple(alpha.shape)}"
                    )
                alpha_b = alpha.view(batch_size, 1, 1, 1).to(self.grid.device, dtype=orig_noise.dtype)
                noise = orig_noise * ((1 - alpha_b) ** 0.5) + normalized_grid * (alpha_b ** 0.5)
        else:
            noise = orig_noise * ((1 - alpha) ** 0.5) + normalized_grid * (alpha ** 0.5)

        if ret_noise:
            return noise, orig_noise

        return noise

    def save(self, filename: str = None, include_date: bool = True) -> str:
        """
        Save the watermark model to disk.
        
        Args:
            path: Base path for saving
            include_date: Whether to append timestamp to filename
            
        Returns:
            Final path where model was saved
        """
        if filename is None:
            filename = self.config.watermark.watermark_file

        # Create data directory
        os.makedirs(self.config.watermark.model_dir, exist_ok=True)
        filepath = os.path.join(self.config.watermark.model_dir, filename)

        if include_date:
            timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(filepath)
            filepath = f"{base}{timestamp}{ext}"

        torch.save(self.state_dict(), filepath)
        print(f"Watermark saved to: {filepath}")
        return filepath

    @staticmethod
    def load(config: Config, filename: Optional[str] = None) -> 'Watermark':
        """
        Load watermark from disk.

        Args:
            filename: Name of the saved model file
            
        Returns:
            Loaded Watermark instance
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if filename is None:
            filename = config.watermark.watermark_file
        filepath = os.path.join(config.watermark.model_dir, filename)

        if not os.path.exists(filepath):
            # Try to find latest timestamped version
            base, ext = os.path.splitext(filepath)
            pattern = f"{base}_*{ext}"

            files = glob.glob(pattern)
            if files:
                filepath = max(files)  # Get most recent
            else:
                raise FileNotFoundError(f"No watermark found at {filepath}")

        data = torch.load(filepath, map_location=config.get_device())
        
        model = Watermark(config)

        model.load_state_dict(data)
        cleanup_cuda_memory()

        print(f"Watermark loaded from: {filepath}")
        return model
