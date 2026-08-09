"""
Adaptive augmentation sampler for training.

This module implements an adaptive sampling strategy for data augmentations,
adjusting probabilities based on model performance to focus on more challenging
augmentations during training.
"""

from typing import Optional, Union

import numpy as np


class AugSampler:
    """
    Adaptive augmentation sampler that adjusts sampling probabilities based on model mistakes.
    
    This class implements an adaptive strategy for selecting data augmentations during training.
    It maintains probabilities for each augmentation type and updates them based on whether
    the model makes mistakes on augmented samples, focusing more on challenging augmentations.
    
    Args:
        n_augs: Number of different augmentation types
        init_p: Initial probability for each augmentation
        eps: Minimum probability threshold to prevent any augmentation from being completely ignored
        base_lr_pos: Base learning rate for increasing probabilities (when mistakes are made)
        base_lr_neg: Base learning rate for decreasing probabilities (when no mistakes are made)
        boost: Multiplicative factor for adaptive learning rate
        beta: Exponent for adaptive learning rate calculation
        smoothing_eps: Small value added for numerical stability in probability calculation
        temp: Temperature parameter for probability softening
        seed: Random seed for reproducibility
    """

    def __init__(self, 
                 n_augs: int, 
                 init_p: float = 0.1, 
                 eps: float = 1e-3,
                 base_lr_pos: float = 0.2, 
                 base_lr_neg: float = 0.05, 
                 boost: float = 3.0, 
                 beta: float = 1.0,
                 smoothing_eps: float = 1e-3, 
                 temp: float = 1.0, 
                 seed: Optional[int] = None):
        self.n = n_augs
        self.p = np.full(n_augs, init_p, dtype=float)
        self.eps = eps
        self.base_lr_pos = base_lr_pos
        self.base_lr_neg = base_lr_neg
        self.boost = boost
        self.beta = beta
        self.smoothing_eps = smoothing_eps
        self.temp = temp
        if seed is not None:
            np.random.seed(seed)
        print("Aug temp is: ", self.temp)

    def sample(self, k: int = 1) -> Union[int, np.ndarray]:
        """
        Sample augmentation indices based on current probabilities.
        
        Args:
            k: Number of samples to draw
            
        Returns:
            Single index (if k=1) or array of indices (if k>1)
        """
        probs = (self.p + self.smoothing_eps) ** (1.0 / self.temp)
        probs = probs / probs.sum()
        choices = np.random.choice(self.n, size=k, replace=True, p=probs)
        return choices if k > 1 else int(choices[0])

    def get_probs(self) -> np.ndarray:
        """
        Get current normalized probabilities for all augmentations.
        
        Returns:
            Array of normalized probabilities
        """
        probs = (self.p + self.smoothing_eps) ** (1.0 / self.temp)
        probs = probs / probs.sum()
        return probs

    def update(self, idx: int, td_error: float) -> None:
        """
        Update probability for a specific augmentation based on model performance.

        Args:
            idx: Index of the augmentation to update
            td_error: Continuous TD-error-style signal in [0, 0.5].
                      0 = perfect prediction, 0.5 = random prediction.
                      Internally normalized to [0, 1] for boost scaling.
        """
        # Normalize td_error ∈ [0, 0.5] -> [0, 1]
        norm = float(min(1.0, max(0.0, float(td_error) * 2.0)))

        # Difficulty-weighted increase: only push up when sample is hard.
        # When norm == 0 (perfect prediction), lr_pos == 0, no push-up.
        # When norm == 1 (random), lr_pos = base_lr_pos * (1 + boost) * adapt.
        adapt = (1.0 - self.p[idx]) ** self.beta
        lr_pos = self.base_lr_pos * norm * (1.0 + self.boost * adapt)
        self.p[idx] = self.p[idx] + lr_pos * (1.0 - self.p[idx])

        # Difficulty-weighted decrease: push down when sample is easy.
        # When norm == 0 (easy), lr_neg = base_lr_neg * adapt_neg, full push-down.
        # When norm == 1 (hard), lr_neg == 0, no push-down.
        adapt_neg = (self.p[idx]) ** self.beta
        lr_neg = self.base_lr_neg * (1.0 - norm) * adapt_neg
        self.p[idx] = self.p[idx] - lr_neg * self.p[idx]

        # Clamp probability to prevent extreme values
        self.p[idx] = np.clip(self.p[idx], self.eps, 1.0 - self.eps)
