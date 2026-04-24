"""
PINN数据集准备模块

该模块负责NIR光谱和E-nose数据的预处理、数据增强和PINN训练数据集的准备。

主要功能：
- NIR光谱预处理（SNV、SG滤波、归一化）
- E-nose传感器数据特征提取和预处理
- Mixup数据增强（生成中间状态样本）
- Delta增强（构造变化趋势样本）
- PINN训练数据集的完整准备（训练集、验证集、配点集）
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List


def preprocess_nir(nir_spectrum: np.ndarray, snv: bool = True, sg_window: int = 15,
                   sg_order: int = 3, normalize: bool = True) -> np.ndarray:
    """
    NIR光谱预处理

    对NIR光谱进行标准化预处理，包括SNV变换、Savitzky-Golay滤波和平滑处理。

    Args:
        nir_spectrum (np.ndarray): 原始NIR光谱数据，形状为(n_samples, n_wavelengths)
        snv (bool): 是否应用标准正态变量变换(SNV)去除散射效应
        sg_window (int): Savitzky-Golay滤波的窗口大小，必须为奇数
        sg_order (int): Savitzky-Golay多项式阶数
        normalize (bool): 是否进行Min-Max归一化

    Returns:
        np.ndarray: 预处理后的NIR光谱数据
    """
    spectrum = nir_spectrum.copy().astype(np.float32)

    # SNV处理 (去散射效应)
    if snv:
        mean = spectrum.mean(axis=1, keepdims=True)
        std = spectrum.std(axis=1, keepdims=True)
        spectrum = (spectrum - mean) / (std + 1e-8)

    # SG滤波 (去噪)
    try:
        from scipy.signal import savgol_filter
        spectrum = savgol_filter(spectrum, sg_window, sg_order, axis=1)
    except:
        pass

    # Min-Max 归一化
    if normalize:
        spec_min = spectrum.min(axis=1, keepdims=True)
        spec_max = spectrum.max(axis=1, keepdims=True)
        spectrum = (spectrum - spec_min) / (spec_max - spec_min + 1e-8)

    return spectrum


def preprocess_enose(sensor_data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    E-nose传感器数据预处理

    从E-nose的时序响应数据中提取统计特征，并进行归一化处理。

    Args:
        sensor_data (np.ndarray): E-nose传感器响应数据
            形状为(n_samples, n_sensors, n_timesteps)或(n_samples, n_features)
        normalize (bool): 是否进行Min-Max归一化

    Returns:
        np.ndarray: 提取的特征向量，形状为(n_samples, n_features)
            特征包括：最大响应值、平均响应值、变异系数等
    """
    # 简单特征提取: 最大值、平均值、变异系数
    if sensor_data.ndim == 3:
        r_max = sensor_data.max(axis=2)  # [n_samples, n_sensors]
        r_mean = sensor_data.mean(axis=2)
        r_std = sensor_data.std(axis=2)
        r_cv = r_std / (r_mean + 1e-8)  # 变异系数

        features = np.concatenate([r_max, r_mean, r_cv], axis=1).astype(np.float32)
    else:
        features = sensor_data.astype(np.float32)

    # 归一化
    if normalize:
        feat_min = features.min(axis=0, keepdims=True)
        feat_max = features.max(axis=0, keepdims=True)
        features = (features - feat_min) / (feat_max - feat_min + 1e-8)

    return features


def mixup_augmentation(X1: np.ndarray, X2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       alpha: float = 0.2, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mixup数据增强

    通过在两个样本之间进行线性插值来生成中间状态的样本，
    这有助于模型学习更平滑的决策边界和更好的泛化能力。

    Args:
        X1 (np.ndarray): 时间点t1的特征数据
        X2 (np.ndarray): 时间点t2的特征数据
        y1 (np.ndarray): 时间点t1的标签
        y2 (np.ndarray): 时间点t2的标签
        alpha (float): Beta分布的参数，控制插值强度
        n_samples (int): 生成的增强样本数量

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - 增强后的特征数据
            - 增强后的标签数据
            - 使用的插值系数lambda
    """
    lambdas = np.random.beta(alpha, alpha, n_samples)

    X_new = []
    y_new = []

    for lam in lambdas:
        x = lam * X1 + (1 - lam) * X2
        y = lam * y1 + (1 - lam) * y2
        X_new.append(x)
        y_new.append(y)

    return np.array(X_new), np.array(y_new), lambdas


def delta_augmentation(X1: np.ndarray, X2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                       times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    差值增强

    构造特征和标签的变化量样本，让模型学习降解过程中的变化趋势，
    而不仅仅是静态的状态值。

    Args:
        X1 (np.ndarray): 时间点t1的特征数据
        X2 (np.ndarray): 时间点t2的特征数据
        y1 (np.ndarray): 时间点t1的标签（质量值）
        y2 (np.ndarray): 时间点t2的标签（质量值）
        times (np.ndarray): 对应的时间点[t1, t2]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - 特征差值 (X2 - X1)
            - 质量变化率 (y2 - y1) / delta_t
            - 时间差 delta_t
            - 特征平均值 (X1 + X2) / 2
    """
    delta_X = X2 - X1
    delta_y = y2 - y1
    delta_t = times[1] - times[0]

    # 变化率 (normalized by time)
    if delta_t > 0:
        rate_y = delta_y / delta_t
    else:
        rate_y = delta_y

    avg_X = (X1 + X2) / 2.0

    return delta_X, rate_y, delta_t, avg_X


def prepare_pinn_dataset(
    X_nir: np.ndarray,
    X_enose: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    temperatures: np.ndarray,
    train_ratio: float = 0.7,
    augment: bool = True,
    mixup_alpha: float = 0.2,
    mixup_samples: int = 100,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    准备PINN训练数据集

    将原始数据分割为训练集、验证集，并生成用于物理损失计算的配点集。
    支持数据增强以提高模型的泛化能力。

    Args:
        X_nir (np.ndarray): NIR光谱数据，形状为(n_samples, n_wavelengths)
        X_enose (np.ndarray): E-nose特征数据，形状为(n_samples, n_features)
        y (np.ndarray): 质量标签，形状为(n_samples,)
        times (np.ndarray): 采样时间，形状为(n_samples,)
        temperatures (np.ndarray): 温度条件，形状为(n_samples,)
        train_ratio (float): 训练集比例
        augment (bool): 是否进行数据增强
        mixup_alpha (float): Mixup增强的Beta分布参数
        mixup_samples (int): 每次Mixup生成的样本数

    Returns:
        Dict[str, Dict[str, torch.Tensor]]: 包含三个数据集的字典
            - 'train': 训练集数据
            - 'val': 验证集数据
            - 'collocation': 配点集数据（用于物理损失）
    """
    n_samples = len(y)
    n_train = int(n_samples * train_ratio)

    # 打乱并分割
    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    # 预处理
    X_nir_proc = preprocess_nir(X_nir)
    X_enose_proc = preprocess_enose(X_enose) if X_enose is not None else np.zeros((n_samples, 1))

    # 转换为torch张量
    def to_tensor(arr):
        return torch.from_numpy(arr).float()

    # 基础训练集
    train_data = {
        'X_nir': to_tensor(X_nir_proc[train_idx]),
        'X_enose': to_tensor(X_enose_proc[train_idx]),
        'times': to_tensor(times[train_idx].reshape(-1, 1)),
        'temperatures': to_tensor(temperatures[train_idx].reshape(-1, 1)),
        'y': to_tensor(y[train_idx]),
    }

    # 验证集
    val_data = {
        'X_nir': to_tensor(X_nir_proc[val_idx]),
        'X_enose': to_tensor(X_enose_proc[val_idx]),
        'times': to_tensor(times[val_idx].reshape(-1, 1)),
        'temperatures': to_tensor(temperatures[val_idx].reshape(-1, 1)),
        'y': to_tensor(y[val_idx]),
    }

    # 配点集 (用于计算物理损失)
    collocation_data = train_data.copy()

    # Mixup增强
    if augment and len(train_idx) > 1:
        # 随机选择成对的样本
        for _ in range(3):  # 做3轮增强
            i, j = np.random.choice(len(train_idx), 2, replace=False)

            X_nir_aug, y_aug, _ = mixup_augmentation(
                X_nir_proc[train_idx[i]], X_nir_proc[train_idx[j]],
                y[train_idx[i]], y[train_idx[j]],
                alpha=mixup_alpha, n_samples=mixup_samples
            )

            # 时间和温度也进行线性插值
            t_aug = np.random.uniform(
                times[train_idx[i]], times[train_idx[j]], mixup_samples
            )
            T_aug = np.random.uniform(
                temperatures[train_idx[i]], temperatures[train_idx[j]], mixup_samples
            )

            collocation_data['X_nir'] = torch.cat([
                collocation_data['X_nir'],
                to_tensor(X_nir_aug)
            ], dim=0)
            collocation_data['y'] = torch.cat([
                collocation_data['y'],
                to_tensor(y_aug)
            ], dim=0)
            collocation_data['times'] = torch.cat([
                collocation_data['times'],
                to_tensor(t_aug.reshape(-1, 1))
            ], dim=0)
            collocation_data['temperatures'] = torch.cat([
                collocation_data['temperatures'],
                to_tensor(T_aug.reshape(-1, 1))
            ], dim=0)

    return {
        'train': train_data,
        'val': val_data,
        'collocation': collocation_data,
    }
