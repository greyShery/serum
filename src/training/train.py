"""
Training module for watermark models.

This module provides a comprehensive Trainer class for pretraining and training
watermark scoring models with proper logging, checkpointing, and configuration management.
"""

import argparse
import copy
import gc
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .. import evaluation
from ..data import Dataset, Augment
from ..models import (
    WatermarkScoreModel,
    PixelWatermarkScoreModel,
    build_score_model,
    ModelWrapper,
    DiffusionModel,
    Watermark,
    encode,
    decode,
    get_vae,
)
from .aug_sampler import AugSampler
from ..utils.config import Config
from ..utils import im
from ..utils.utils import cleanup_cuda_memory, save_config_snapshot


class Trainer:
    """
    Trainer class for watermark models.
    
    Handles both pretraining (contrastive learning on clean data) and 
    watermark training (learning to distinguish watermarked vs clean images).
    """

    def __init__(self,
                 config: Config,
                 diffusion_model: nn.Module,
                 score_model: nn.Module,
                 watermark: nn.Module,
                 aug_sampler: AugSampler,
                 data: torch.Tensor,
                 data_aug: torch.Tensor,
                 prompts: List[str],
                 augment: Optional[Augment] = None):
        """
        Initialize the trainer.
        
        Args:
            score_model: Watermark scoring model to train
            data: Clean training data tensor
            data_aug: Augmented training data tensor
            prompts: List of training prompts
            diffusion_model: Diffusion model for generating watermarked images
            augment: Function to perturb images for augmentation
        """
        self.config = config
        self.diffusion_model = diffusion_model
        self.score_model = score_model
        self.watermark = watermark
        self.aug_sampler = aug_sampler
        self.data = data
        self.data_aug = data_aug
        self.prompts = prompts
        self.augment = augment if augment is not None else Augment()
        self.device = self.config.get_device()

        # Move model to device
        self.score_model = self.score_model.to(self.device).train().requires_grad_(True)
        self.watermark = self.watermark.to(self.device)

        # Detector mode: 'latent' (4 x 64 x 64 — needs VAE) or 'pixel' (3 x 512 x 512).
        self.detector_type = getattr(self.config.training, 'detector_type', 'latent').lower()
        if self.detector_type not in ('latent', 'pixel'):
            raise ValueError(
                f"Unsupported detector_type: {self.detector_type!r}. "
                "Expected 'latent' or 'pixel'."
            )
        print(f"Detector type: {self.detector_type}")

        # Create checkpoint directory
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)

        # Set random seed
        torch.manual_seed(self.config.training.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training.random_seed)

    def _prepare_dataloader(self, batch_size: int) -> None:
        """
        Prepare the data loader for training.
        
        Args:
            batch_size: Batch size for training
            test_split: Fraction of data to use for testing
        """

        flatten_tensors = lambda x: [b for a in x for b in a]

        gen_data = flatten_tensors(self.data)
        aug_data = flatten_tensors(self.data_aug)
        min_len = min(len(gen_data), len(aug_data))
        tensor_ds = TensorDataset(
            torch.stack(gen_data[:min_len]),
            torch.stack(aug_data[:min_len])
        )

        self.dataloader = DataLoader(tensor_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        print("Train samples:", len(tensor_ds))

    @torch.no_grad()
    def _prepare_watermark_loader(self, batch_size: int, watermark_batch_size: int, load_checkpoint: bool = True) -> None:
        # Pixel-space images at 3 x 512 x 512 are ~48x larger than latent
        # tensors (4 x 64 x 64). Caching ~15k of them as a single .pt would
        # cost ~42 GB on disk, so we always regenerate on-the-fly in pixel mode.
        if self.detector_type == 'pixel':
            self._setup_pixel_watermark_loader(batch_size, watermark_batch_size)
            return

        # Generate dataset filename based on configuration
        #   默认 (固定 α)：    watermarked_dataset_{N}_{alpha}_latent.pt
        #   创新点 #1 (自适应 α): watermarked_dataset_{N}_adaptive_latent.pt
        #   注：自适应 α 模式下, 文件名不嵌 α — 因为 α 是 per-sample 的
        use_adaptive = bool(getattr(self.config.watermark, 'adaptive_alpha', False))
        expected_size = (self.config.watermark.buffer.size // batch_size) * batch_size
        suffix = 'pixel' if self.detector_type == 'pixel' else 'latent'
        if use_adaptive:
            dataset_filename = f"watermarked_dataset_{expected_size}_adaptive_{suffix}.pt"
        else:
            dataset_filename = f"watermarked_dataset_{expected_size}_{self.config.watermark.grid.noise_mix_alpha}_{suffix}.pt"
        dataset_path = os.path.join(self.config.training.checkpoint_dir, dataset_filename)

        # Check if dataset already exists and has correct size
        if load_checkpoint and os.path.exists(dataset_path):
            print(f"Loading existing watermarked dataset from {dataset_path}")
            try:
                dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)

                if len(dataset) == expected_size:
                    print(f"Loaded {len(dataset)} watermarked images from existing dataset")

                    # Create dataloader
                    self.watermarked_loader = DataLoader(
                        dataset,
                        batch_size=watermark_batch_size,
                        shuffle=True,
                        drop_last=True
                    )

                    # Create iterator for training loop
                    self.watermarked_iter = iter(self.watermarked_loader)
                    return
                else:
                    print(f"Existing dataset has {len(dataset)} images, but expected {expected_size}. Regenerating...")
            except Exception as e:
                print(f"Error loading existing dataset: {e}. Regenerating...")

        # Generate new dataset
        batches_watermarked_images = []
        batches_watermarked_aug = []

        print("Generating watermarked images for training...")

        # Calculate how many full batches we can fit in buffer.size
        num_batches = self.config.watermark.buffer.size // batch_size
        use_adaptive = bool(getattr(self.config.watermark, 'adaptive_alpha', False))

        # === 创新点 #1（自适应 α）辅助函数 ===
        #   按 prompt token 长度映射 α ∈ [α_min, α_max]
        #   简单 prompt (token 短) → α 小 → 弱水印 → 保护图像细节
        #   复杂 prompt (token 长) → α 大 → 强水印 → 确保可检测
        alpha_min = float(getattr(self.config.watermark, 'adaptive_alpha_min', 0.3))
        alpha_max = float(getattr(self.config.watermark, 'adaptive_alpha_max', 0.7))
        alpha_scale = float(getattr(self.config.watermark, 'adaptive_alpha_scale', 25.0))

        def _prompt_to_alpha(prompt_list):
            """把 prompt list 映射成长度 B 的 α tensor"""
            if not use_adaptive:
                return float(self.config.watermark.grid.noise_mix_alpha)
            lengths = torch.tensor([len(p.split()) for p in prompt_list], dtype=torch.float32)
            # sigmoid 把 [0, +∞) 映射到 (0, 1), 然后 linear map 到 [alpha_min, alpha_max]
            # scale 控制拐点 — 25 表示 "平均 25 token 出现 α=0.5"
            norm = torch.sigmoid((lengths - alpha_scale) / 8.0)
            return alpha_min + (alpha_max - alpha_min) * norm

        for batch_idx in tqdm(range(num_batches)):
            # === 创新点 #1 关键修改 ===
            # 自适应 α 模式下, 我们需要先知道本 batch 用哪几个 prompt 才能算 α tensor
            # generate_full 默认会随机挑 prompt — 我们用 ret_prompts=True 拿到它们
            if use_adaptive:
                # 先用一个 dummy noise 跑一次只是为了拿到 prompts
                # 但更省事的做法: 直接手动 random sample prompts
                batch_prompts = [self.prompts[random.randint(0, len(self.prompts) - 1)] for _ in range(batch_size)]
                alpha_tensor = _prompt_to_alpha(batch_prompts).to(self.watermark.grid.device)
                watermark_noise = self.watermark(batch_size, alpha=alpha_tensor).half()
            else:
                alpha = self.config.watermark.grid.noise_mix_alpha
                watermark_noise = self.watermark(batch_size, alpha=alpha).half()
                batch_prompts = None

            # Generate watermarked images using diffusion model
            if use_adaptive and batch_prompts is not None:
                # 显式传 positive_prompts, 让 SD 用我们选好的 prompts
                watermarked_images = self.diffusion_model.generate_full(
                    batch_size=batch_size,
                    noise=watermark_noise,
                    x_ts_ret=False,
                    prompts=None,
                    positive_prompts=batch_prompts,
                    negative_prompts=[''] * batch_size,
                ).float()
            else:
                watermarked_images = self.diffusion_model.generate_full(
                    batch_size=batch_size,
                    noise=watermark_noise,
                    x_ts_ret=False,
                    prompts=self.prompts
                ).float()

            if self.detector_type == 'pixel':
                # Pixel detector: cache decoded images (3 x 512 x 512) directly.
                # Skip the VAE encode/decode round-trip that the latent pipeline uses.
                primary = decode(watermarked_images.half()).float()
                # Augment the *pixel* images, then keep them in pixel space.
                primary_aug = self.augment(primary, idx=self.aug_sampler.sample()).float()
            else:
                # Latent detector: keep the legacy latent encode/decode round-trip.
                images_aug = encode(self.augment(decode(watermarked_images.half()).float()).half()).float()
                primary = watermarked_images
                primary_aug = images_aug

            # Store watermarked images for later use
            batches_watermarked_images.append(primary.cpu())
            batches_watermarked_aug.append(primary_aug.cpu())

            del watermarked_images, primary, primary_aug, watermark_noise
            cleanup_cuda_memory()

        # Create dataset
        all_images = torch.cat(batches_watermarked_images, dim=0)
        all_images_aug = torch.cat(batches_watermarked_aug, dim=0)

        dataset = TensorDataset(all_images, all_images_aug)

        # Save the dataset directly
        print(f"Saving watermarked dataset to {dataset_path}")
        torch.save(dataset, dataset_path)

        print(f"Generated {len(dataset)} watermarked images for training buffer ({self.detector_type})")

        self.watermarked_loader = DataLoader(
            dataset,
            batch_size=watermark_batch_size,
            shuffle=True,
            drop_last=True
        )

        # Create iterator for training loop
        self.watermarked_iter = iter(self.watermarked_loader)
        cleanup_cuda_memory()

    def _get_watermarked_batch(self, aug_idx: Optional[int] = None,
                                batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch from watermarked dataloader.

        For the latent detector, batches are pre-computed and cached; for the
        pixel detector, batches are generated on-the-fly per call (much cheaper
        than caching 15k images at 3 x 512 x 512 on disk).
        """
        if self.detector_type == 'pixel':
            return self._generate_pixel_watermarked_batch(batch_size, aug_idx)

        try:
            images, images_aug = next(self.watermarked_iter)
            return images.to(self.device), images_aug.to(self.device)
        except StopIteration:
            # Reset iterator when we reach the end
            self.watermarked_iter = iter(self.watermarked_loader)
            images, images_aug = next(self.watermarked_iter)
            return images.to(self.device), images_aug.to(self.device)

    @torch.no_grad()
    def _generate_pixel_watermarked_batch(self, batch_size: int,
                                          aug_idx: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a single watermarked pixel batch on-the-fly.

        Args:
            batch_size: number of watermarked images to generate this step.
            aug_idx: passed in from the caller so ``aug_sampler.sample()`` is
                only called once per training step (the resulting index is used
                both for the augmented wm-batch and the augmented clean-batch).
        """
        if aug_idx is None:
            aug_idx = self.aug_sampler.sample()
        wm_noise = self.watermark(batch_size, alpha=self.config.watermark.grid.noise_mix_alpha).half()
        wm_latents = self.diffusion_model.generate_full(
            batch_size=batch_size,
            noise=wm_noise,
            x_ts_ret=False,
            prompts=self.prompts,
        ).float()
        wm_pixel = decode(wm_latents.half()).float().clamp(-1, 1)
        wm_pixel_aug = self.augment(wm_pixel, idx=aug_idx).float().clamp(-1, 1)
        del wm_noise, wm_latents
        cleanup_cuda_memory()
        return wm_pixel, wm_pixel_aug

    @torch.no_grad()
    def _setup_pixel_watermark_loader(self, batch_size: int, watermark_batch_size: int) -> None:
        """No-op for the pixel loader: batches are streamed lazily.

        We still set ``self.watermarked_iter`` so older code paths that test
        ``hasattr(self, 'watermarked_iter')`` keep working.
        """
        self.watermarked_loader = None
        self.watermarked_iter = None
        print(f"Pixel loader is streaming on-the-fly (no cache): {self.detector_type}")

    def _prepare_optimizer(self) -> None:
        # Initialize training optimizer
        if self.config.training.optimizer.lower() == 'Adam'.lower():
            cls = torch.optim.Adam
        elif self.config.training.optimizer.lower() == 'SGD'.lower():
            cls = torch.optim.SGD
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        params = [{"params": list(self.score_model.parameters()), **self.config.training.optimizer_score_model_args}]
        self.optimizer = cls(params)

    def _prepare_scheduler(self) -> None:
        if self.config.training.scheduler.lower() == 'ReduceLROnPlateau'.lower():
            cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.scheduler}")

        self.scheduler = cls(
            self.optimizer,
            **self.config.training.scheduler_args
        )

    def train_epoch(self, epoch: int, verbose: bool = True) -> Tuple[float, float, float]:
        """
        Run one training epoch.
        
        This function trains a binary classifier to distinguish between watermarked 
        and clean images. The model learns to predict 1 for watermarked images and 
        0 for clean images.
        
        Args:
            verbose: Whether to print loss information
            
        Returns:
            List of loss values for all batches in the epoch
        """
        # Set models to appropriate training modes
        self.score_model.train()
        self.watermark.eval()

        # Initialize metrics tracking for this epoch
        losses = []  # Track loss for each batch
        acc_w = []  # Track accuracy on watermarked (augmented) images
        acc_nw = []  # Track accuracy on non-watermarked (clean) images

        counter = 0
        # Iterate through batches of clean training data
        # x1, x2 are both clean images (original and augmented versions).
        # For the latent detector they are 4 x 64 x 64 latents; for the pixel
        # detector they are decoded to 3 x 512 x 512 images here, once per batch,
        # so the 6 forward heads never have to round-trip through VAE.
        for x1, x2 in tqdm(self.dataloader):
            # Move clean data to device
            x1 = x1.to(self.device).float()  # Clean images (original)
            x2 = x2.to(self.device).float()  # Clean images (augmented)

            if self.detector_type == 'pixel':
                x1 = decode(x1.half()).float()
                x2 = decode(x2.half()).float()
                x1 = x1.clamp(-1, 1)
                x2 = x2.clamp(-1, 1)

            aug_idx = self.aug_sampler.sample()

            # Get watermarked images from dataloader.
            # In pixel mode this generates a fresh batch on-the-fly (avoids
            # caching 15k images at 3 x 512 x 512 = ~42 GB on disk).
            wm_batch_size = min(self.config.training.aug_batch_size * 4,
                                self.config.training.watermark_batch_size)
            watermarked_images, watermarked_images_aug = self._get_watermarked_batch(
                aug_idx=aug_idx, batch_size=wm_batch_size,
            )

            aug_batch = self.config.training.aug_batch_size

            if self.detector_type == 'pixel':
                # Pixel detector: skip the latent encode/decode hop completely.
                aug_images = self.augment(
                    x1[:aug_batch].float(), idx=aug_idx
                ).float().clamp(-1, 1)
                aug_images_watermarked = self.augment(
                    watermarked_images[:aug_batch].float(), idx=aug_idx
                ).float().clamp(-1, 1)
            else:
                # Latent detector: keep the original round-trip pipeline.
                aug_images = encode(
                    self.augment(decode(x1[:aug_batch, ].half()).float(), idx=aug_idx).half()
                ).float()
                # Create augmented versions of watermarked images
                # This tests the model's robustness to image transformations
                # Pipeline: watermarked latents - decode to pixels - augment - encode back to latents
                aug_images_watermarked = encode(
                    self.augment(decode(watermarked_images[:aug_batch, ].half()).float(),
                                 idx=aug_idx).half()
                ).float()

            # === FORWARD PASS: Get model predictions on different image types ===
            inputs = [
                watermarked_images,
                watermarked_images_aug,
                x1,
                x2,
                aug_images,
                aug_images_watermarked,
            ]
            inputs = [t.to(self.device).float() for t in inputs]

            sizes = [t.size(0) for t in inputs]
            concat_in = torch.cat(inputs, dim=0)
            logits_all = self.score_model(concat_in)

            logits_images, logits_aug, logits_x1, logits_x2, \
                logits_aug_images, logits_aug_images_watermarked = torch.split(logits_all, sizes, dim=0)

            # Probabilities for aug_sampler mistake signals and accuracy metrics
            p_aug_images_watermarked = torch.sigmoid(logits_aug_images_watermarked)
            p_aug_images = torch.sigmoid(logits_aug_images)

            # === ACCURACY CALCULATION ===
            acc_aug_imgs_w = (p_aug_images_watermarked > 0.5).float().mean()
            acc_aug_imgs_nw = (p_aug_images < 0.5).float().mean()

            # === TARGET LABELS ===
            ones = torch.ones_like(logits_images)
            zeros = torch.zeros_like(logits_x1)

            # === WEIGHTED LOSS with focal component ===
            # Each term:  weight * focal(p, target)
            # focal(p, y) = -y * (1-p)^gamma * log(p)  -  (1-y) * p^gamma * log(1-p)
            # gamma=1 recovers standard BCEWithLogitsLoss.
            gamma = getattr(self.config.training, 'focal_gamma', 1.0)
            pos_w  = getattr(self.config.training, 'pos_weight', 1.25)   # watermark (positive) class weight
            neg_w  = getattr(self.config.training, 'neg_weight', 1.0)   # clean (negative) class weight

            def _focal_bce(logits, target_value, weight, gamma_):
                # Auto-build target with same shape as logits (avoids shape-mismatch
                # when different forward heads have different batch sizes).
                target = torch.full_like(logits, target_value)
                p = torch.sigmoid(logits)
                pt = torch.where(target > 0.5, p, 1 - p).clamp(min=1e-7, max=1 - 1e-7)
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                focal = bce * ((1 - pt) ** gamma_)
                return (weight * focal).mean()

            # ----------------------------------------------------------------
            # 创新点 1 (NovAUG-on): 6 路 focal loss (原始 SERUM 损失)
            # ----------------------------------------------------------------
            # Positive (watermarked) terms — label = 1
            L_watermarked       = _focal_bce(logits_images,                  1.0, pos_w, gamma)
            L_aug_watermarked   = _focal_bce(logits_aug,                    1.0, pos_w, gamma)
            L_aug_wm_custom     = _focal_bce(logits_aug_images_watermarked, 1.0, pos_w, gamma)

            # Negative (clean) terms — label = 0
            L_clean             = _focal_bce(logits_x1,                     0.0, neg_w, gamma)
            L_clean_aug         = _focal_bce(logits_x2,                     0.0, neg_w, gamma)
            L_clean_custom      = _focal_bce(logits_aug_images,             0.0, neg_w, gamma)

            loss_cls = (L_watermarked + L_aug_watermarked + L_aug_wm_custom +
                        L_clean * 0.8 + L_clean_aug * 0.8 + L_clean_custom * 0.8)

            # ----------------------------------------------------------------
            # 创新点 2 (NovCONS-IST): Augmentation consistency regularizer
            #   文章 idea: 同一 prompt 的水印图在 (clean aug) 前后，detector 分数应保持一致
            #   依据: SIDE_BY_SIDE.md §4.3 (mistake-as-signal) +  LDM 隐式 augment
            # ----------------------------------------------------------------
            loss_cons = logits_all.new_tensor(0.0)
            if getattr(self.config.training, 'use_consistency', True):
                w_cons = getattr(self.config.training, 'consistency_weight', 0.1)
                # 水印图 clean vs aug 的一致性（logits 层面 MSE）
                loss_cons_w = F.mse_loss(
                    logits_aug_images_watermarked.view(-1),
                    logits_images[:logits_aug_images_watermarked.shape[0]].view(-1),
                )
                # 干净图 clean vs aug 的一致性
                loss_cons_c = F.mse_loss(
                    logits_aug_images.view(-1),
                    logits_x1[:logits_aug_images.shape[0]].view(-1),
                )
                loss_cons = w_cons * (loss_cons_w + loss_cons_c)

            # ----------------------------------------------------------------
            # 创新点 3 (NovMARGIN): Margin-based contrastive loss (HiDDeN 2018 灵感)
            #   让 watermark logits 与 clean logits 至少隔开 margin 的距离
            #   依据: HiDDeN 论文 Eq.4 (margin ranking) + SERUM §3.2 latent-space detection
            # ----------------------------------------------------------------
            loss_margin = logits_all.new_tensor(0.0)
            if getattr(self.config.training, 'use_margin', True):
                w_margin = getattr(self.config.training, 'margin_weight', 0.05)
                margin = getattr(self.config.training, 'margin_value', 2.0)
                # 配对 logits (取较小 batch 大小)
                n = min(logits_images.shape[0], logits_x1.shape[0])
                pos_logits = logits_images[:n].view(-1)
                neg_logits = logits_x1[:n].view(-1)
                # hinge: max(0, margin - (pos - neg))
                loss_margin = w_margin * F.relu(
                    margin - (pos_logits - neg_logits)
                ).mean()

            # ----------------------------------------------------------------
            # 创新点 4 (NovALPHA): KL/alpha regularization (Appendix B 解析)
            #   论文 Appendix B: KL(η' || η) ∝ α² 而不依赖于 grid 具体值
            #   轻微正则化 alpha, 防止数值走偏
            # ----------------------------------------------------------------
            loss_alpha = logits_all.new_tensor(0.0)
            if getattr(self.config.training, 'use_alpha_reg', True):
                w_alpha = float(getattr(self.config.training, 'alpha_reg_weight', 1e-4))
                alpha_val = float(self.config.watermark.grid.noise_mix_alpha)
                alpha_t = torch.tensor(alpha_val, dtype=logits_all.dtype, device=logits_all.device)
                alpha_lo = torch.tensor(0.3, dtype=logits_all.dtype, device=logits_all.device)
                alpha_hi = torch.tensor(0.7, dtype=logits_all.dtype, device=logits_all.device)
                w_alpha_t = torch.tensor(w_alpha, dtype=logits_all.dtype, device=logits_all.device)
                loss_alpha = w_alpha_t * (
                    F.relu(alpha_lo - alpha_t) + F.relu(alpha_t - alpha_hi)
                )

            # ----------------------------------------------------------------
            # 创新点 5 (NovPER): TD-error-style priority for miss 反馈
            #   原文: aug_sampler 使用二元 mistake  -> 换成连续错误率
            #   依据: PER_Schaul2016 论文, mistake-as-signal 多任务思想
            # ----------------------------------------------------------------
            p_aug_images_watermarked = torch.sigmoid(logits_aug_images_watermarked)
            p_aug_images = torch.sigmoid(logits_aug_images)
            # 连续 优先级 (0~1), 适中错误率才"学"
            td_error_w = (0.5 - p_aug_images_watermarked).abs().mean().item()
            td_error_c = (p_aug_images - 0.5).abs().mean().item()
            td_error = max(td_error_w, td_error_c)
            self.aug_sampler.update(aug_idx, td_error)

            # === TOTAL LOSS ===
            loss = loss_cls + loss_cons + loss_margin + loss_alpha

            # === BACKWARD PASS ===
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # === ACCURACY CALCULATION ===
            acc_aug_imgs_w = (p_aug_images_watermarked > 0.5).float().mean()
            acc_aug_imgs_nw = (p_aug_images < 0.5).float().mean()

            # Store metrics for this batch
            losses.append(loss.item())

            # Track accuracy for all batches
            acc_w.append(acc_aug_imgs_w.item())  # % of watermarked images correctly identified
            acc_nw.append(acc_aug_imgs_nw.item())  # % of clean images correctly identified

            # === LOGGING ===
            log_interval = self.config.training.visualization.log_interval
            if verbose and counter % log_interval == 0:
                # Print running averages over recent batches
                recent_loss = np.mean(losses[-log_interval:]) if len(losses) >= log_interval else np.mean(losses)
                recent_acc_w  = np.mean(acc_w[-log_interval:])  if len(acc_w)  >= log_interval else np.mean(acc_w)
                recent_acc_nw = np.mean(acc_nw[-log_interval:]) if len(acc_nw) >= log_interval else np.mean(acc_nw)

                print(f"Batch {counter + 1}")
                print(f'Loss: {recent_loss:.9f}')
                print(f'Watermarked (augmented) Acc: {recent_acc_w:.9f}')
                print(f'Non-Watermarked (augmented) Acc: {recent_acc_nw:.9f}')
                print(f'  - cls={loss_cls.item():.6f}  cons={loss_cons.item():.6f}  '
                      f'margin={loss_margin.item():.6f}  alpha={loss_alpha.item():.6f}  '
                      f'td_err={td_error:.4f}')
                print(f'Final probs: {self.aug_sampler.get_probs()}')

            counter += 1

        # === EPOCH END: UPDATE LEARNING RATE ===
        self.scheduler.step(np.mean(losses))

        # === VALIDATION ON TEST SET ===
        # Set models to eval mode for validation
        self.score_model.eval()
        self.watermark.eval()

        # Evaluate performance on held-out clean data
        print(f"Epoch loss: {np.mean(losses)}")
        print(f"Epoch watermark accs: {np.mean(acc_w)}")

        # Test accuracy on clean images from test set
        print(f"Epoch non-watermark accs: {np.mean(acc_nw)}")

        print(f'Final probabilities: {self.aug_sampler.get_probs()}')

        return losses

    def train(self,
              epochs: int = None,
              batch_size: int = None,
              watermark_batch_size: int = None,
              full_eval_interval: int = None,
              verbose: bool = True,
              load_checkpoint: bool = True) -> List[Tuple[float, float, float]]:
        """
        Run full training.

        Args:
            epochs: Number of epochs to train (uses config default if None)
            batch_size: Batch size for training (uses config default if None)
            verbose: Whether to print progress
            load_checkpoint: Whether to load the latest checkpoint

        Returns:
            List of (total_loss, opt_loss, non_opt_loss) for each epoch
        """
        if epochs is None:
            epochs = self.config.training.epochs
        if batch_size is None:
            batch_size = self.config.training.train_batch_size
        if watermark_batch_size is None:
            watermark_batch_size = self.config.training.watermark_batch_size
        if full_eval_interval is None:
            full_eval_interval = self.config.training.full_eval_interval

        self.diffusion_model.eval()
        self._prepare_dataloader(batch_size=batch_size)
        self._prepare_optimizer()
        self._prepare_scheduler()

        # Load checkpoint if requested
        start_epoch = 0
        if load_checkpoint:
            start_epoch = self.load_checkpoint()
            if start_epoch > 0:
                print(f"Resuming training from epoch {start_epoch + 1}")

        self._prepare_watermark_loader(batch_size=batch_size, 
                                       watermark_batch_size=watermark_batch_size,
                                       load_checkpoint=load_checkpoint)

        if start_epoch == 0:
            self.save_checkpoint(0)  # Save initial state

        if full_eval_interval > 0:
            test_prompts = Dataset.get_prompts(self.config, use_test=True)
            is_first_eval = True

        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(self.config.training.random_seed)

        epoch_losses = []
        for epoch in range(start_epoch, epochs):
            print(f"\n[Train Epoch {epoch + 1}/{epochs}]")

            losses = self.train_epoch(epoch, verbose=verbose)
            epoch_losses.append(losses)

            if full_eval_interval > 0 and (epoch + 1) % full_eval_interval == 0:
                print("\nRunning full evaluation...")
                evaluation.Evaluation(
                    config=self.config,
                    diffusion_model=copy.deepcopy(self.diffusion_model),
                    score_model=copy.deepcopy(self.score_model),
                    watermark=copy.deepcopy(self.watermark),
                    prompts=test_prompts,
                    augment=copy.deepcopy(self.augment)
                ).run_full_eval(force_reload=is_first_eval)
                is_first_eval = False
                cleanup_cuda_memory()

            self.save_checkpoint(epoch + 1)

        return epoch_losses

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint including optimizer, scheduler, and training state.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'score_model_state_dict': self.score_model.state_dict(),
            'watermark_state_dict': self.watermark.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'random_state': torch.get_rng_state().byte()
        }

        if torch.cuda.is_available():
            # Save the random state for the specific device being used
            checkpoint['cuda_random_state'] = torch.cuda.get_rng_state(device=self.device).byte()
            checkpoint['device'] = str(self.device)  # Save which device was used

        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)

        # Also save the latest checkpoint
        latest_path = os.path.join(self.config.training.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(checkpoint, latest_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> int:
        """
        Load model checkpoint including optimizer, scheduler, and training state.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest checkpoint.
            
        Returns:
            The epoch number from the loaded checkpoint
        """
        if checkpoint_path is None:
            checkpoint_path = Trainer.get_checkpoint_path(self.config)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"No checkpoint found")
            return 0

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model states
        self.score_model.load_state_dict(checkpoint['score_model_state_dict'])
        self.watermark.load_state_dict(checkpoint['watermark_state_dict'])

        print("Model states restored from checkpoint")

        # Load optimizer and scheduler states (only if they exist)
        #if hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
        #    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #    print("Optimizer state restored from checkpoint")
        #else:
        #    print("Warning: Optimizer state not found in checkpoint, skipping optimizer state restoration")
        self._prepare_optimizer()

        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored from checkpoint")
        else:
            print("Warning: Scheduler state not found in checkpoint, skipping scheduler state restoration")

        # Load random states
        if 'random_state' in checkpoint:
            random_state = checkpoint['random_state']
            if isinstance(random_state, torch.Tensor):
                random_state = random_state.to(torch.uint8).cpu()
                torch.set_rng_state(random_state)
                print("RNG state restored from checkpoint")
            else:
                print("Warning: Invalid random state format, skipping RNG state restoration")
                random_state = None
        else:
            print("Warning: Random state not found in checkpoint, skipping RNG state restoration")

        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            cuda_random_state = checkpoint['cuda_random_state']
            if isinstance(cuda_random_state, torch.Tensor):
                cuda_random_state = cuda_random_state.to(torch.uint8).cpu()
                # Set random state for the specific device
                torch.cuda.set_rng_state(cuda_random_state, device=self.device)
                print(f"CUDA RNG state restored from checkpoint for device {self.device}")
            else:
                print("Warning: Invalid CUDA random state format, skipping CUDA RNG state restoration")
        else:
            print("Warning: CUDA random state not found in checkpoint, skipping CUDA RNG state restoration")

        epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch

    @staticmethod
    def get_checkpoint_path(config: Config) -> str:
        """
        Get the path to the latest available checkpoint.
        
        Args:
            config: Configuration object containing checkpoint directory
        
        Returns:
            Path to the latest checkpoint, or empty string if none found
        """
        checkpoint_dir = config.training.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            return ""

        # Try latest_checkpoint.pt first
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            return latest_path

        # Fallback to latest numbered checkpoint
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
                try:
                    # Extract epoch number from filename
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pt", "")
                    epoch_num = int(epoch_str)
                    checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, filename)))
                except ValueError:
                    # Skip files that don't have valid epoch numbers
                    continue

        if not checkpoint_files:
            return ""

        # Sort by epoch number and return the latest one
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        latest_epoch, latest_path = checkpoint_files[0]
        print(f"Found latest numbered checkpoint: {latest_path} (epoch {latest_epoch})")
        return latest_path

    @staticmethod
    def load_models_from_checkpoint(config: Config, checkpoint_path: str) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Load score model, watermark, and optimizer from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Tuple of (score_model, watermark, optimizer)
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        device = config.get_device()
        print(f"Loading models from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize models
        score_model = WatermarkScoreModel(config).to(device)
        watermark = Watermark(config).to(device)

        score_model.load_state_dict(checkpoint['score_model_state_dict'])
        watermark.load_state_dict(checkpoint['watermark_state_dict'])

        return score_model, watermark


def main() -> None:
    """
    Main training function that initializes models and starts training.
    """
    parser = argparse.ArgumentParser(description='Train watermark models')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (overrides default)')
    parser.add_argument('--include-eval', action='store_true', default=False,
                        help='Run evaluation at the end of training')
    parser.add_argument('--dont-load-checkpoint', action='store_true', default=False,
                        help='Do not load checkpoint at the start of training')

    args = parser.parse_args()

    # Load config if provided
    config = Config(args.config)

    # Save config snapshot at the beginning
    save_config_snapshot(config, "train_config")

    print("Initializing training...")
    device = config.get_device()

    get_vae(config)

    augment = Augment()

    diffusion_model = ModelWrapper(DiffusionModel(config).to(device)).to(device)
    watermark = Watermark(config).to(device)

    # Initialize watermark scoring model
    score_model = WatermarkScoreModel(config).to(device)

    print("Watermark Score Model:")
    print(f"Total parameters: {sum(p.numel() for p in score_model.parameters()):,}")

    train_data_gen, train_data_aug = Dataset.load_data_pair(config)
    train_prompts = Dataset.get_prompts(config, use_test=False)
    print(f"Sample prompt: {train_prompts[17]}")

    # Initialize trainer
    aug_sampler = AugSampler(len(augment), temp=config.training.aug_temp)
    trainer = Trainer(
        config=config,
        diffusion_model=diffusion_model,
        score_model=score_model,
        watermark=watermark,
        aug_sampler=aug_sampler,
        data=train_data_gen,
        data_aug=train_data_aug,
        prompts=train_prompts,
        augment=augment
    )

    print("Trainer initialized!")

    # Watermark training
    print("Training watermark model...")
    start_train_time = time.time()
    train_losses = trainer.train(verbose=False, load_checkpoint=not args.dont_load_checkpoint)
    print('Total training time: {:.2f} minutes'.format((time.time() - start_train_time) / 60))

    if train_losses:
        print(f"Training complete. Final loss: {np.mean(train_losses[-1]):.6f}")
    else:
        print("Training completed with no losses recorded.")

    # Save the models
    score_model.save()
    watermark.save()
    print("Models saved.")

    if args.include_eval:
        print("Running evaluation...")
        cleanup_cuda_memory()
        evaluation.Evaluation(
            config=config,
            diffusion_model=copy.deepcopy(diffusion_model),
            score_model=copy.deepcopy(score_model),
            watermark=copy.deepcopy(watermark),
            prompts=Dataset.get_prompts(config, use_test=True),
            augment=copy.deepcopy(augment)
        ).run_full_eval()


if __name__ == "__main__":
    main()
