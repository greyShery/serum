"""
Model definitions for watermarking experiments.

This module contains the core models including the vision transformer
and watermark scoring model.
"""

from .diffusion import ModelWrapper, DiffusionModel
from .score_model import (
    WatermarkScoreModel,
    PixelWatermarkScoreModel,
    build_score_model,
)
from .vae import get_vae, encode, decode
from .watermark import Watermark

__all__ = [
    'WatermarkScoreModel', 'PixelWatermarkScoreModel', 'build_score_model',
    'Watermark',
    'ModelWrapper', 'DiffusionModel',
    'get_vae', 'encode', 'decode'
]
