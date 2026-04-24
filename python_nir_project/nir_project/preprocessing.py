"""
NIR光谱预处理模块

该模块提供了多种近红外(NIR)光谱预处理方法，包括：
- Savitzky-Golay滤波(SG)
- 多重散射校正(MSC)
- 标准正态变量变换(SNV)
- 基线校正： baseline_zero
- 去噪(Despiking)

这些预处理方法可以单独使用或组合使用，以改善光谱数据的质量和后续建模性能。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, medfilt
from typing import Optional


def apply_sg(X: np.ndarray, order: int, window: int) -> np.ndarray:
    """
    应用Savitzky-Golay滤波进行光谱平滑

    Savitzky-Golay滤波是一种基于多项式拟合的平滑滤波方法，
    能够保持光谱的峰形特征同时减少噪声。

    Args:
        X (np.ndarray): 输入光谱数据，形状为(n_samples, n_features)
        order (int): 多项式阶数，通常为2-4
        window (int): 窗口大小，必须为奇数且大于order

    Returns:
        np.ndarray: 平滑后的光谱数据，形状与输入相同

    Raises:
        ValueError: 当window不是奇数或小于等于order时抛出
    """
    if window % 2 == 0 or window <= order:
        raise ValueError('SG window must be odd and greater than order')
    X_out = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_out[i, :] = savgol_filter(X[i, :], window_length=window, polyorder=order, mode='interp')
    return X_out


def apply_msc(X: np.ndarray, reference_mode: str = 'mean') -> np.ndarray:
    """
    应用多重散射校正(Multiplicative Scatter Correction)

    MSC通过对每个光谱进行线性回归校正来减少散射效应。
    使用参考光谱(平均、中位数或第一个光谱)作为基准。

    Args:
        X (np.ndarray): 输入光谱数据，形状为(n_samples, n_features)
        reference_mode (str): 参考光谱计算方式
            - 'mean': 使用平均光谱作为参考
            - 'median': 使用中位数光谱作为参考
            - 'first': 使用第一个光谱作为参考

    Returns:
        np.ndarray: MSC校正后的光谱数据

    Raises:
        ValueError: 当reference_mode不支持时抛出
    """
    if reference_mode == 'mean':
        ref = np.nanmean(X, axis=0)
    elif reference_mode == 'median':
        ref = np.nanmedian(X, axis=0)
    elif reference_mode == 'first':
        ref = X[0, :]
    else:
        raise ValueError(f'Unsupported MSC reference mode: {reference_mode}')

    X_out = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i, :]
        valid = np.isfinite(row) & np.isfinite(ref)
        if np.count_nonzero(valid) < 2:
            X_out[i, :] = row
            continue
        p = np.polyfit(ref[valid], row[valid], 1)
        slope = p[0]
        intercept = p[1]
        if not np.isfinite(slope) or abs(slope) < 1e-12:
            X_out[i, :] = row - intercept
        else:
            corrected = (row - intercept) / slope
            corrected[~np.isfinite(corrected)] = row[~np.isfinite(corrected)]
            X_out[i, :] = corrected
    return X_out


def apply_snv(X: np.ndarray, mode: str = 'standard') -> np.ndarray:
    """
    应用标准正态变量变换(Standard Normal Variate)

    SNV通过减去均值并除以标准差来标准化每个光谱，
    从而减少光程长度和散射效应的影响。

    Args:
        X (np.ndarray): 输入光谱数据，形状为(n_samples, n_features)
        mode (str): SNV变换模式
            - 'standard': 标准SNV，使用均值和标准差
            - 'robust': 鲁棒SNV，使用中位数和MAD(中位数绝对偏差)

    Returns:
        np.ndarray: SNV变换后的光谱数据

    Raises:
        ValueError: 当mode不支持时抛出
    """
    X_out = np.empty_like(X)
    for i in range(X.shape[0]):
        row = X[i, :]
        if mode == 'standard':
            mean = np.nanmean(row)
            std = np.nanstd(row)
            if std < 1e-12:
                X_out[i, :] = row - mean
            else:
                X_out[i, :] = (row - mean) / std
        elif mode == 'robust':
            median = np.nanmedian(row)
            mad = np.nanmedian(np.abs(row - median))
            if mad < 1e-12:
                X_out[i, :] = row - median
            else:
                X_out[i, :] = (row - median) / mad
        else:
            raise ValueError(f'Unsupported SNV mode: {mode}')
    return X_out


def baseline_zero(X: np.ndarray, mode: str = 'none', scope: str = 'cropped_spectrum') -> np.ndarray:
    """
    执行基线校正，使光谱从零开始

    通过减去基线值来校正光谱的基线偏移。

    Args:
        X (np.ndarray): 输入光谱数据
        mode (str): 基线计算方式
            - 'none': 不进行基线校正
            - 'first_point': 使用第一个点的值作为基线
            - 'first_5_mean': 使用前5个点的平均值作为基线
        scope (str): 应用范围(目前未使用，保留兼容性)

    Returns:
        np.ndarray: 基线校正后的光谱数据

    Raises:
        ValueError: 当mode不支持时抛出
    """
    if mode == 'none':
        return X
    if mode == 'first_point':
        base = X[:, 0]
    elif mode == 'first_5_mean':
        n = min(5, X.shape[1])
        base = np.nanmean(X[:, :n], axis=1)
    else:
        raise ValueError(f'Unsupported baseline_zero_mode: {mode}')
    return X - base.reshape(-1, 1)


def despike(X: np.ndarray, mode: str = 'none') -> np.ndarray:
    """
    去除光谱中的尖峰噪声

    使用多种算法检测和去除光谱中的异常尖峰值。

    Args:
        X (np.ndarray): 输入光谱数据，形状为(n_samples, n_features)
        mode (str): 去噪模式
            - 'none': 不进行去噪
            - 'median3', 'median5', 'median7': 中位数滤波，窗口大小分别为3、5、7
            - 'local': 局部去噪，阈值为3
            - 'local_strong': 强局部去噪，阈值为2
            - 'jump_guard': 跳跃保护算法

    Returns:
        np.ndarray: 去噪后的光谱数据

    Raises:
        ValueError: 当mode不支持时抛出
    """
    if mode == 'none':
        return X
    if mode in {'median3', 'median5', 'median7'}:
        kernel = int(mode.replace('median', ''))
        return np.vstack([medfilt(row, kernel_size=kernel) for row in X])
    if mode in {'local', 'local_strong'}:
        threshold = 3 if mode == 'local' else 2
        return _local_despike(X, threshold)
    if mode == 'jump_guard':
        return _jump_guard(X)
    raise ValueError(f'Unsupported despike_mode: {mode}')


def _local_despike(X: np.ndarray, threshold: int) -> np.ndarray:
    """
    局部去噪算法的内部实现

    检测光谱中的局部尖峰，通过比较当前点与其邻域的关系来判断是否为噪声。

    Args:
        X (np.ndarray): 输入光谱数据
        threshold (int): 尖峰检测阈值

    Returns:
        np.ndarray: 去噪后的光谱数据
    """
    out = X.copy()
    for i in range(X.shape[0]):
        row = X[i, :].copy()
        for j in range(1, row.shape[0] - 1):
            left_v = row[j - 1]
            mid_v = row[j]
            right_v = row[j + 1]
            neigh_mean = (left_v + right_v) / 2
            neigh_diff = abs(right_v - left_v)
            spike_mag = abs(mid_v - neigh_mean)
            local_scale = max(neigh_diff, abs(mid_v - left_v), abs(mid_v - right_v), np.finfo(float).eps)
            if spike_mag > threshold * local_scale and np.sign(mid_v - left_v) != np.sign(right_v - mid_v):
                row[j] = neigh_mean
        out[i, :] = row
    return out


def _jump_guard(X: np.ndarray) -> np.ndarray:
    """
    跳跃保护去噪算法的内部实现

    检测光谱中的跳跃式尖峰，当点与其两个邻居的差值都很大时认为是噪声。

    Args:
        X (np.ndarray): 输入光谱数据

    Returns:
        np.ndarray: 去噪后的光谱数据
    """
    out = X.copy()
    for i in range(X.shape[0]):
        row = X[i, :].copy()
        for j in range(1, row.shape[0] - 1):
            left_v = row[j - 1]
            mid_v = row[j]
            right_v = row[j + 1]
            d1 = abs(mid_v - left_v)
            d2 = abs(mid_v - right_v)
            neigh_gap = abs(left_v - right_v)
            jump_ref = max(neigh_gap, np.finfo(float).eps)
            if d1 > 4 * jump_ref and d2 > 4 * jump_ref:
                row[j] = (left_v + right_v) / 2
        out[i, :] = row
    return out


def preprocess_spectrum(
    X: np.ndarray,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    msc_ref_mode: str = 'mean',
    snv_mode: str = 'standard',
    baseline_zero_mode: str = 'none',
    baseline_zero_scope: str = 'cropped_spectrum',
    despike_mode: str = 'none',
) -> np.ndarray:
    """
    对单个光谱数据集执行完整的预处理流程

    按照指定的顺序组合应用多种预处理方法。

    Args:
        X (np.ndarray): 输入光谱数据，形状为(n_samples, n_features)
        preproc_mode (str): 预处理模式组合，用'+'分隔
            - 'sg': Savitzky-Golay滤波
            - 'msc': 多重散射校正
            - 'snv': 标准正态变量变换
            - 'none': 不进行预处理
        sg_order (int): SG滤波的多项式阶数
        sg_window (int): SG滤波的窗口大小
        msc_ref_mode (str): MSC的参考光谱模式
        snv_mode (str): SNV变换模式
        baseline_zero_mode (str): 基线校正模式
        baseline_zero_scope (str): 基线校正范围
        despike_mode (str): 去噪模式

    Returns:
        np.ndarray: 预处理后的光谱数据
    """
    X_out = X.copy()
    if baseline_zero_scope == 'full_spectrum':
        X_out = baseline_zero(X_out, baseline_zero_mode, baseline_zero_scope)
        X_out = despike(X_out, despike_mode)
    if preproc_mode != 'none':
        if 'sg' in preproc_mode:
            X_out = apply_sg(X_out, sg_order, sg_window)
        if 'msc' in preproc_mode:
            X_out = apply_msc(X_out, msc_ref_mode)
        if 'snv' in preproc_mode:
            X_out = apply_snv(X_out, snv_mode)
    if baseline_zero_scope != 'full_spectrum':
        X_out = baseline_zero(X_out, baseline_zero_mode, baseline_zero_scope)
        X_out = despike(X_out, despike_mode)
    return X_out


def preprocess_pair(
    X_train: np.ndarray,
    X_test: np.ndarray,
    preproc_mode: str = 'sg+msc+snv',
    sg_order: int = 3,
    sg_window: int = 15,
    msc_ref_mode: str = 'mean',
    snv_mode: str = 'standard',
) -> tuple[np.ndarray, np.ndarray]:
    """
    对训练集和测试集光谱对执行相同的预处理

    确保训练集和测试集使用相同的预处理参数和参考值。

    Args:
        X_train (np.ndarray): 训练集光谱数据
        X_test (np.ndarray): 测试集光谱数据
        preproc_mode (str): 预处理模式组合
        sg_order (int): SG滤波的多项式阶数
        sg_window (int): SG滤波的窗口大小
        msc_ref_mode (str): MSC的参考光谱模式
        snv_mode (str): SNV变换模式

    Returns:
        tuple[np.ndarray, np.ndarray]: 预处理后的训练集和测试集光谱数据

    Raises:
        ValueError: 当msc_ref_mode不支持时抛出
    """
    mode = preproc_mode.lower().strip()
    if mode == 'none':
        return X_train, X_test
    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()
    if 'sg' in mode:
        X_train_proc = apply_sg(X_train_proc, sg_order, sg_window)
        X_test_proc = apply_sg(X_test_proc, sg_order, sg_window)
    if 'msc' in mode:
        if msc_ref_mode not in {'mean', 'median', 'first'}:
            raise ValueError(f'Unsupported MSC mode: {msc_ref_mode}')
        X_train_proc = apply_msc(X_train_proc, msc_ref_mode)
        X_test_proc = apply_msc(X_test_proc, msc_ref_mode)
    if 'snv' in mode:
        X_train_proc = apply_snv(X_train_proc, snv_mode)
        X_test_proc = apply_snv(X_test_proc, snv_mode)
    return X_train_proc, X_test_proc
