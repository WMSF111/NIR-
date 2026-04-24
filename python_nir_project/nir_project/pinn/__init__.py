"""
PINN (Physics-Informed Neural Network) 模块
用于NIR光谱与E-nose数据的多模态物质降解预测
"""

from .dataset import (
    preprocess_nir,
    preprocess_enose,
    mixup_augmentation,
    delta_augmentation,
    prepare_pinn_dataset,
)
from .model import PINNNetwork
from .loss import PINNLoss, arrhenius_rate_constant
from .train import train_pinn, train_pinn_two_stage
from .evaluate import evaluate_pinn, plot_comparison, plot_kinetics

__all__ = [
    "preprocess_nir",
    "preprocess_enose",
    "mixup_augmentation",
    "delta_augmentation",
    "prepare_pinn_dataset",
    "PINNNetwork",
    "PINNLoss",
    "arrhenius_rate_constant",
    "train_pinn",
    "train_pinn_two_stage",
    "evaluate_pinn",
    "plot_comparison",
    "plot_kinetics",
]
