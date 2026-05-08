"""
NIR光谱特征选择模块

该模块提供了多种特征选择算法，用于从高维NIR光谱数据中选择最具信息量的特征：
- 相关性选择(Correlation): 基于特征与目标变量的相关性
- 主成分分析(PCA): 基于方差解释率选择特征
- 连续投影算法(SPA): 选择最小角度回归特征
- 竞争自适应重加权采样(CARS): 基于PLS权重选择特征
- 递归特征消除(RFE): 基于线性回归逐步消除特征

这些方法可以帮助减少数据维度，提高模型性能和解释性。
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Iterable, Tuple


def _scale_features(X: np.ndarray) -> np.ndarray:
    """Scale feature columns to comparable units before selection."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def _self_check_default_param(method_name: str, feature_count: int) -> int:
    """返回特征选择方法自检时使用的默认参数。"""
    method_name = method_name.lower().strip()
    if method_name == 'corr_topk':
        return min(5, feature_count)
    if method_name == 'pca':
        return min(5, feature_count)
    if method_name == 'spa':
        return min(5, feature_count)
    if method_name == 'cars':
        return min(5, feature_count)
    raise ValueError(f'Unsupported self-check method: {method_name}')


def quick_self_check_feature_selection(
    methods: Iterable[str] | None = None,
    sample_count: int = 24,
    feature_count: int = 12,
    random_state: int = 0,
) -> Dict[str, Any]:
    """快速自检特征选择方法是否能正常运行。

    该函数使用一份可复现的合成数据，依次调用指定的特征选择方法，
    用于排查依赖缺失、输入形状错误或算法实现异常等问题。
    """
    supported_methods = ('spa', 'pca', 'cars', 'corr_topk')
    methods_to_run = list(methods) if methods is not None else list(supported_methods)
    normalized_methods = [method.lower().strip() for method in methods_to_run]

    invalid_methods = [method for method in normalized_methods if method not in supported_methods]
    if invalid_methods:
        raise ValueError(f'Unsupported self-check methods: {invalid_methods}')

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(sample_count, feature_count))
    y = (1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] + 0.05 * rng.normal(size=sample_count)).astype(float)

    print(
        f'开始特征选择快速自检，共 {len(normalized_methods)} 个方法，'
        f'样本数={sample_count}，特征数={feature_count}'
    )

    results: Dict[str, Any] = {}
    for index, method_name in enumerate(normalized_methods, start=1):
        fs_param = _self_check_default_param(method_name, feature_count)
        print(f'[{index}/{len(normalized_methods)}] 自检方法={method_name} | 参数={fs_param}')
        result = select_features_by_method(X, y, method_name, fs_param)
        selected_idx = np.asarray(result['selected_idx'])
        results[method_name] = {
            'success': True,
            'fs_param': fs_param,
            'selected_count': int(selected_idx.size),
            'selected_idx': selected_idx.tolist(),
        }
        print(
            f'[{index}/{len(normalized_methods)}] 完成 {method_name} | '
            f'选中特征数={selected_idx.size}'
        )

    return {
        'success': True,
        'sample_count': sample_count,
        'feature_count': feature_count,
        'methods': results,
    }


def feature_select_corr_topk(X: np.ndarray, y: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于相关系数的特征选择

    计算每个特征与目标变量的相关系数绝对值，选择相关性最强的top_k个特征。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        y (np.ndarray): 目标变量，形状为(n_samples,)
        top_k (int): 选择的前k个特征数量

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - selected_idx: 选中的特征索引数组
            - scores: 每个特征的相关系数绝对值分数
    """
    y = y.reshape(-1)
    scores = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = np.isfinite(col) & np.isfinite(y)
        if np.count_nonzero(valid) < 2:
            scores[j] = 0.0
            continue
        corr = np.corrcoef(col[valid], y[valid])[0, 1]
        scores[j] = 0.0 if np.isnan(corr) else abs(corr)
    order = np.argsort(scores)[::-1]
    top_k = max(1, min(top_k, X.shape[1]))
    return order[:top_k], scores


def feature_select_pca(X: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    基于主成分分析的特征选择

    使用PCA计算特征的综合重要性权重，选择贡献最大的特征。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        top_k (int): 选择的前k个特征数量

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - selected_idx: 选中的特征索引数组
            - scores: 每个特征的综合重要性分数
            - info: 包含PCA信息的字典
                - n_pc: 主成分数量
                - explained_ratio: 方差解释率
                - coeff: 主成分系数矩阵
    """
    X_scaled = _scale_features(X)
    pca = PCA()
    scores = np.zeros(X.shape[1], dtype=float)
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)
    n_pc = int(np.searchsorted(cum_explained, 0.99) + 1)
    n_pc = max(1, min(n_pc, X.shape[1]))
    weights = explained[:n_pc].astype(float)
    weights = weights / np.sum(weights)
    coeff = pca.components_.T[:, :n_pc]
    scores = np.sum(np.abs(coeff) * weights.reshape(1, -1), axis=1)
    order = np.argsort(scores)[::-1]
    top_k = max(1, min(top_k, X.shape[1]))
    return order[:top_k], scores, {'n_pc': n_pc, 'explained_ratio': explained, 'coeff': coeff}


def feature_select_spa(X: np.ndarray, y: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    基于连续投影算法的特征选择

    SPA是一种前向选择算法，选择最小角度回归的特征组合。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        y (np.ndarray): 目标变量，形状为(n_samples,)
        top_k (int): 选择的前k个特征数量

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - selected_idx: 选中的特征索引数组
            - scores: 特征选择分数(选中为1.0，未选中为0.0)
            - info: 包含SPA信息的字典
                - initial_idx: 初始选择的特征索引
    """
    order, corr_scores = feature_select_corr_topk(X, y, X.shape[1])
    initial_idx = int(order[0])
    selected = spa(X, initial_idx, top_k)
    scores = np.zeros(X.shape[1], dtype=float)
    scores[selected] = 1.0
    return selected, scores, {'initial_idx': initial_idx}


def feature_select_cars(X: np.ndarray, y: np.ndarray, target_count: int = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    基于竞争自适应重加权采样的特征选择

    CARS使用PLS回归权重来评估特征重要性，选择最具影响力的特征。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        y (np.ndarray): 目标变量，形状为(n_samples,)
        target_count (int, optional): 目标特征数量，默认min(30, n_features)

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - selected_idx: 选中的特征索引数组
            - scores: 特征选择分数(选中为1.0，未选中为0.0)
            - info: 包含CARS信息的字典
                - target_count: 目标特征数量
                - n_components: PLS主成分数量

    Raises:
        ImportError: 当sklearn.cross_decomposition不可用时抛出
    """
    if target_count is None:
        target_count = min(30, X.shape[1])
    n_components = min(30, X.shape[1], X.shape[0] - 1)
    if n_components < 1:
        n_components = 1
    try:
        from sklearn.cross_decomposition import PLSRegression
    except ImportError:
        raise
    X_scaled = _scale_features(X)
    plsr = PLSRegression(n_components=n_components)
    plsr.fit(X_scaled, y)
    weights = np.sum(np.abs(plsr.x_weights_), axis=1)
    order = np.argsort(weights)[::-1]
    target_count = max(1, min(target_count, X.shape[1]))
    selected = order[:target_count]
    scores = np.zeros(X.shape[1], dtype=float)
    scores[selected] = 1.0
    return selected, scores, {'target_count': target_count, 'n_components': n_components}


def spa(X: np.ndarray, initial_idx: int, tot_n: int) -> np.ndarray:
    """
    连续投影算法(SPA)的核心实现

    迭代选择与已选特征投影正交的特征，实现最小角度回归。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        initial_idx (int): 初始特征索引(通常选择相关性最高的特征)
        tot_n (int): 总共要选择的特征数量

    Returns:
        np.ndarray: 选中的特征索引数组
    """
    n_samples, n_features = X.shape
    tot_n = min(tot_n, n_features)
    selected = [initial_idx]
    specj = X.copy()
    specn = X[:, initial_idx].reshape(-1, 1)
    all_idx = list(range(n_features))

    for _ in range(tot_n - 1):
        not_selected = [j for j in all_idx if j not in selected]
        aps = np.full(len(not_selected), -np.inf, dtype=float)
        denom = float((specn.T @ specn).item())
        if abs(denom) < 1e-12:
            denom = 1.0
        for idx, j in enumerate(not_selected):
            proj = specj[:, j] - (specj[:, j].T @ specn).item() / denom * specn.ravel()
            aps[idx] = np.linalg.norm(proj)
        best = int(np.argmax(aps))
        selected.append(not_selected[best])
        specn = X[:, selected[-1]].reshape(-1, 1)
        specj = X.copy()
    return np.array(selected, dtype=int)


def select_features_by_method(X: np.ndarray, y: np.ndarray, fs_method: str, fs_param: Any = None) -> Dict[str, Any]:
    """
    统一的特征选择接口

    根据指定的方法和参数执行特征选择，返回统一格式的结果。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        y (np.ndarray): 目标变量，形状为(n_samples,)
        fs_method (str): 特征选择方法
            - 'corr_topk': 相关性选择
            - 'pca': PCA特征选择
            - 'spa': SPA特征选择
            - 'cars': CARS特征选择
            - 'rfe': 递归特征消除
        fs_param (Any, optional): 方法特定的参数

    Returns:
        Dict[str, Any]: 包含特征选择结果的字典
            - selected_idx: 选中的特征索引数组
            - score: 特征重要性分数数组
            - info: 方法特定的附加信息字典

    Raises:
        ValueError: 当fs_method不支持时抛出
    """
    method = fs_method.lower().strip()
    if method == 'corr_topk':
        top_k = int(fs_param) if fs_param is not None else min(100, X.shape[1])
        selected_idx, score = feature_select_corr_topk(X, y, top_k)
        return {'selected_idx': selected_idx, 'score': score, 'info': {'top_k': top_k}}
    if method == 'pca':
        top_k = int(fs_param) if fs_param is not None else min(30, X.shape[1])
        selected_idx, score, info = feature_select_pca(X, top_k)
        return {'selected_idx': selected_idx, 'score': score, 'info': {**info, 'top_k': top_k}}
    if method == 'spa':
        top_k = int(fs_param) if fs_param is not None else min(30, X.shape[1])
        selected_idx, score, info = feature_select_spa(X, y, top_k)
        return {'selected_idx': selected_idx, 'score': score, 'info': {**info, 'top_k': top_k}}
    if method == 'cars':
        target_count = int(fs_param) if fs_param is not None else min(30, X.shape[1])
        selected_idx, score, info = feature_select_cars(X, y, target_count)
        return {'selected_idx': selected_idx, 'score': score, 'info': info}
    if method == 'rfe':
        top_k = int(fs_param) if fs_param is not None else min(40, X.shape[1])
        X_scaled = _scale_features(X)
        selector = RFE(LinearRegression(), n_features_to_select=max(1, min(top_k, X.shape[1])), step=0.1)
        selector.fit(X_scaled, y)
        selected_idx = np.where(selector.support_)[0]
        score = np.full(X.shape[1], np.nan, dtype=float)
        score[selected_idx] = 1.0
        return {'selected_idx': selected_idx, 'score': score, 'info': {'top_k': top_k}}
    raise ValueError(f'Unsupported feature selection method: {fs_method}')
