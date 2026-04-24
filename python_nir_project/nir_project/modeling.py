"""
NIR光谱建模模块

该模块提供了多种机器学习算法用于NIR光谱定量分析，包括：
- 传统方法：PLS回归、主成分回归(PCR)、支持向量机回归(SVR)
- 集成方法：随机森林(RF)、XGBoost
- 神经网络：多层感知机(CNN)、高斯过程回归(GPR)
- 距离方法：K近邻回归(KNN)

模块还包含了数据分割、模型训练、参数优化和性能评估的功能。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C

from .data import load_dataset
from .feature_selection import select_features_by_method
from .preprocessing import preprocess_pair

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _kennard_stone_split(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Kennard-Stone算法进行训练集和测试集分割

    Kennard-Stone算法通过最大化样本间的欧几里得距离来选择最具代表性的训练样本。

    Args:
        X (np.ndarray): 特征矩阵，形状为(n_samples, n_features)
        k (int): 训练集样本数量

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 训练集索引数组
            - 测试集索引数组

    Raises:
        ValueError: 当k无效时抛出
    """
    n_samples = X.shape[0]
    if k <= 0 or k >= n_samples:
        raise ValueError('Invalid KS training size')
    remaining = list(range(n_samples))
    selected: List[int] = []
    mean_sample = np.nanmean(X, axis=0)
    dists = np.linalg.norm(X - mean_sample, axis=1)
    first = int(np.argmax(dists))
    selected.append(first)
    remaining.remove(first)
    for _ in range(1, k):
        rem_X = X[remaining]
        dists = np.full(len(remaining), np.inf)
        for idx_sel in selected:
            dist_matrix = np.linalg.norm(X[remaining] - X[idx_sel], axis=1)
            dists = np.minimum(dists, dist_matrix)
        best_index = int(np.argmax(dists))
        selected.append(remaining[best_index])
        remaining.pop(best_index)
    return np.array(selected, dtype=int), np.array(remaining, dtype=int)


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    评估回归模型的性能

    计算决定系数(R²)和均方根误差(RMSE)。

    Args:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值

    Returns:
        Dict[str, Any]: 包含评估指标的字典
            - r2: 决定系数
            - rmse: 均方根误差
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {'r2': float(r2), 'rmse': float(rmse)}


def _train_pls(X: np.ndarray, y: np.ndarray, max_lv: int, cv_fold: int) -> Tuple[PLSRegression, int]:
    """
    训练PLS回归模型并选择最优主成分数量

    通过交叉验证选择最佳的主成分数量。

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        max_lv (int): 最大主成分数量
        cv_fold (int): 交叉验证折数

    Returns:
        Tuple[PLSRegression, int]:
            - 训练好的PLS模型
            - 最优主成分数量
    """
    max_lv = min(max_lv, X.shape[1], X.shape[0] - 1)
    best = 1
    best_rmse = float('inf')
    for n in range(1, max_lv + 1):
        model = PLSRegression(n_components=n)
        cv = KFold(n_splits=min(cv_fold, X.shape[0] - 1), shuffle=True, random_state=1)
        rmses = []
        for train_idx, valid_idx in cv.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[valid_idx])
            rmses.append(mean_squared_error(y[valid_idx], pred, squared=False))
        avg_rmse = np.mean(rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best = n
    model = PLSRegression(n_components=best)
    model.fit(X, y)
    return model, best


def _train_pcr(X: np.ndarray, y: np.ndarray, max_pc: int) -> Tuple[LinearRegression, PCA, int]:
    """
    训练主成分回归(PCR)模型

    先进行PCA降维，然后在主成分上进行线性回归。

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        max_pc (int): 最大主成分数量

    Returns:
        Tuple[LinearRegression, PCA, int]:
            - 训练好的线性回归模型
            - 训练好的PCA模型
            - 最优主成分数量
    """
    max_pc = min(max_pc, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=max_pc)
    score = pca.fit_transform(X)
    best = 1
    best_rmse = float('inf')
    for n in range(1, max_pc + 1):
        reg = LinearRegression()
        cv = KFold(n_splits=min(5, X.shape[0] - 1), shuffle=True, random_state=1)
        rmses = []
        for train_idx, valid_idx in cv.split(score):
            reg.fit(score[train_idx, :n], y[train_idx])
            pred = reg.predict(score[valid_idx, :n])
            rmses.append(mean_squared_error(y[valid_idx], pred, squared=False))
        avg_rmse = np.mean(rmses)
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best = n
    reg = LinearRegression()
    reg.fit(score[:, :best], y)
    return reg, pca, best


def _train_svr(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[SVR, Dict[str, Any]]:
    """
    训练支持向量机回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[SVR, Dict[str, Any]]:
            - 训练好的SVR模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
        }
    model = GridSearchCV(SVR(), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_, model.best_params_


def _train_rf(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    训练随机森林回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[RandomForestRegressor, Dict[str, Any]]:
            - 训练好的随机森林模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {'n_estimators': [100, 200], 'min_samples_leaf': [1, 5, 10]}
    model = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_, model.best_params_


def _train_knn(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[KNeighborsRegressor, Dict[str, Any]]:
    """
    训练K近邻回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[KNeighborsRegressor, Dict[str, Any]]:
            - 训练好的KNN模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    model = GridSearchCV(KNeighborsRegressor(), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_, model.best_params_


def _train_cnn(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[MLPRegressor, Dict[str, Any]]:
    """
    训练多层感知机(神经网络)回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[MLPRegressor, Dict[str, Any]]:
            - 训练好的神经网络模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [1000]
        }
    model = GridSearchCV(MLPRegressor(random_state=1), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_, model.best_params_


def _train_xgb(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[XGBRegressor, Dict[str, Any]]:
    """
    训练XGBoost回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[XGBRegressor, Dict[str, Any]]:
            - 训练好的XGBoost模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    model = GridSearchCV(XGBRegressor(random_state=1), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_, model.best_params_


def _train_gpr(X: np.ndarray, y: np.ndarray, param_grid: Optional[Dict[str, Any]] = None) -> Tuple[GaussianProcessRegressor, Dict[str, Any]]:
    """
    训练高斯过程回归模型并进行参数优化

    Args:
        X (np.ndarray): 训练特征矩阵
        y (np.ndarray): 训练目标变量
        param_grid (Optional[Dict[str, Any]]): 参数网格，默认为预设网格

    Returns:
        Tuple[GaussianProcessRegressor, Dict[str, Any]]:
            - 训练好的高斯过程模型
            - 最优参数字典
    """
    if param_grid is None:
        param_grid = {'kernel': ['squaredexponential', 'matern32', 'matern52']}
    kernel_map = {
        'squaredexponential': C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
        'matern32': C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5),
        'matern52': C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5),
    }
    class GPRWrapper:
        def __init__(self, kernel='squaredexponential'):
            self.kernel = kernel
        def fit(self, X, y):
            self.gpr_ = GaussianProcessRegressor(kernel=kernel_map[self.kernel], random_state=1)
            self.gpr_.fit(X, y)
            return self
        def predict(self, X):
            return self.gpr_.predict(X, return_std=False)
        def score(self, X, y):
            return self.gpr_.score(X, y)
    model = GridSearchCV(GPRWrapper(), param_grid, scoring='neg_root_mean_squared_error', cv=min(5, X.shape[0] - 1), n_jobs=-1)
    model.fit(X, y)
    return model.best_estimator_.gpr_, model.best_params_


def _save_regression_plot(
    path: Path,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    title: str,
) -> None:
    """
    保存回归模型的预测结果散点图

    生成训练集和测试集预测结果的散点图，并保存为PNG文件。

    Args:
        path (Path): 保存路径
        y_train (np.ndarray): 训练集真实值
        y_pred_train (np.ndarray): 训练集预测值
        y_test (np.ndarray): 测试集真实值
        y_pred_test (np.ndarray): 测试集预测值
        title (str): 图表标题
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_test, label='Test', alpha=0.6)
    plt.scatter(y_train, y_pred_train, label='Train', alpha=0.6)
    min_val = min(np.min(y_test), np.min(y_pred_test), np.min(y_train), np.min(y_pred_train))
    max_val = max(np.max(y_test), np.max(y_pred_test), np.max(y_train), np.max(y_pred_train))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_model_from_dataset(dataset_path: Path, method_name: str, method_param: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    从数据集文件训练机器学习模型

    完整的建模流程：加载数据、预处理、特征选择、模型训练、评估和结果保存。

    Args:
        dataset_path (Path): 数据集文件路径
        method_name (str): 模型方法名称
            - 'pls': 偏最小二乘回归
            - 'pcr': 主成分回归
            - 'svr': 支持向量机回归
            - 'rf': 随机森林回归
            - 'gpr': 高斯过程回归
            - 'knn': K近邻回归
            - 'cnn': 神经网络回归
            - 'xgb': XGBoost回归
        method_param (Optional[Dict[str, Any]]): 模型特定参数

    Returns:
        Dict[str, Any]: 包含训练结果的字典
            - method: 模型方法
            - dataset_tag: 数据集标签
            - metadata: 元数据
            - selected_idx: 选中的特征索引
            - best_param: 最优参数
            - ytrain/ypred_train: 训练集真实值和预测值
            - ytest/ypred: 测试集真实值和预测值
            - metrics: 训练和测试集的评估指标
            - plot_path: 预测结果图路径

    Raises:
        ValueError: 当方法不支持或数据格式错误时抛出
    """
    dataset = load_dataset(dataset_path)
    X = dataset['X']
    y = dataset['y'].reshape(-1)
    metadata = dict(dataset.get('metadata', {}))
    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError('X and y sample counts do not match')

    preproc_mode = metadata.get('preproc_mode', 'sg+msc')
    sg_order = int(metadata.get('sg_order', 3))
    sg_window = int(metadata.get('sg_window', 15))
    msc_ref_mode = metadata.get('msc_ref_mode', 'mean')
    snv_mode = metadata.get('snv_mode', 'standard')
    baseline_zero_mode = metadata.get('baseline_zero_mode', 'none')
    baseline_zero_scope = metadata.get('baseline_zero_scope', 'cropped_spectrum')
    despike_mode = metadata.get('despike_mode', 'none')
    data_stage = metadata.get('data_stage', 'raw')

    if data_stage.lower().strip() == 'raw':
        X_train_full = X.copy()
        X_test_full = X.copy()
    else:
        X_train_full, X_test_full = _preprocess_split(
            X,
            preproc_mode=preproc_mode,
            sg_order=sg_order,
            sg_window=sg_window,
            msc_ref_mode=msc_ref_mode,
            snv_mode=snv_mode,
            baseline_zero_mode=baseline_zero_mode,
            baseline_zero_scope=baseline_zero_scope,
            despike_mode=despike_mode,
        )

    train_size = max(2, round(n_samples * 0.75))
    sel_idx, rem_idx = _kennard_stone_split(X_train_full, train_size)
    Xtrain = X_train_full[sel_idx, :]
    Xtest = X_train_full[rem_idx, :]
    ytrain = y[sel_idx]
    ytest = y[rem_idx]

    if data_stage.lower() == 'selected':
        fs_result = select_features_by_method(X_train_full, y, metadata.get('fs_method', 'corr_topk'), metadata.get('fs_param'))
        selected_idx = fs_result['selected_idx']
        Xtrain = Xtrain[:, selected_idx]
        Xtest = Xtest[:, selected_idx]
        metadata['selected_idx'] = selected_idx.tolist()
        metadata['feature_score'] = fs_result['score'].tolist()

    method = method_name.lower().strip()
    result: Dict[str, Any] = {
        'method': method,
        'dataset_tag': metadata.get('dataset_tag', ''),
        'metadata': metadata,
        'selected_idx': metadata.get('selected_idx', []),
    }

    if method == 'pls':
        max_lv = int(method_param.get('max_lv', 300)) if isinstance(method_param, dict) else 300
        cv_fold = int(method_param.get('cv_fold', 10)) if isinstance(method_param, dict) else 10
        model, best_lv = _train_pls(Xtrain, ytrain, max_lv, cv_fold)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = {'n_components': best_lv, 'cv_fold': cv_fold}
    elif method == 'pcr':
        max_pc = int(method_param.get('max_pc', 300)) if isinstance(method_param, dict) else 300
        reg, pca, best_k = _train_pcr(Xtrain, ytrain, max_pc)
        ypred_train = reg.predict(pca.transform(Xtrain)[:, :best_k]).reshape(-1)
        ypred = reg.predict(pca.transform(Xtest)[:, :best_k]).reshape(-1)
        result['best_param'] = {'n_components': best_k}
    elif method == 'svr':
        model, best = _train_svr(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    elif method == 'rf':
        model, best = _train_rf(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    elif method == 'gpr':
        model, best = _train_gpr(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    elif method == 'knn':
        model, best = _train_knn(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    elif method == 'cnn':
        model, best = _train_cnn(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    elif method == 'xgb':
        model, best = _train_xgb(Xtrain, ytrain, method_param)
        ypred_train = model.predict(Xtrain).reshape(-1)
        ypred = model.predict(Xtest).reshape(-1)
        result['best_param'] = best
    else:
        raise ValueError(f'Unsupported method_name: {method_name}')

    train_eval = _evaluate_regression(ytrain, ypred_train)
    test_eval = _evaluate_regression(ytest, ypred)
    result.update(
        {
            'ytrain': ytrain.tolist(),
            'ypred_train': ypred_train.tolist(),
            'ytest': ytest.tolist(),
            'ypred': ypred.tolist(),
            'metrics': {
                'train': train_eval,
                'test': test_eval,
            },
        }
    )

    result_dir = PROJECT_ROOT / 'result' / 'model' / method / Path(metadata.get('dataset_tag', 'dataset')).name
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_path = result_dir / f'{metadata.get("dataset_tag", "dataset")}_{method}.png'
    _save_regression_plot(plot_path, ytrain, ypred_train, ytest, ypred, f'{method.upper()} Prediction')
    result['plot_path'] = str(plot_path)
    return result


def _preprocess_split(
    X: np.ndarray,
    preproc_mode: str = 'sg+msc+snv',
    sg_order: int = 3,
    sg_window: int = 15,
    msc_ref_mode: str = 'mean',
    snv_mode: str = 'standard',
    baseline_zero_mode: str = 'none',
    baseline_zero_scope: str = 'cropped_spectrum',
    despike_mode: str = 'none',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对训练集和测试集执行相同的预处理

    Args:
        X (np.ndarray): 输入特征矩阵
        preproc_mode (str): 预处理模式
        sg_order (int): SG滤波多项式阶数
        sg_window (int): SG滤波窗口大小
        msc_ref_mode (str): MSC参考模式
        snv_mode (str): SNV模式
        baseline_zero_mode (str): 基线校正模式
        baseline_zero_scope (str): 基线校正范围
        despike_mode (str): 去噪模式

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - 预处理后的训练集特征矩阵
            - 预处理后的测试集特征矩阵
    """
    X_train = X.copy()
    X_test = X.copy()
    if preproc_mode.lower().strip() == 'none':
        return X_train, X_test
    return preprocess_pair(
        X_train,
        X_test,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        msc_ref_mode=msc_ref_mode,
        snv_mode=snv_mode,
    )
