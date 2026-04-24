"""
NIR光谱分析管道模块

该模块提供了完整的NIR光谱分析工作流程，包括：
- 数据集准备和预处理
- 特征选择参数优化
- 多模型比较和评估
- 自动化管道执行

模块整合了数据加载、预处理、特征选择和建模的所有步骤。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .data import (
    PROJECT_ROOT,
    build_property_vector_from_all_csv,
    black_white_processing,
    save_dataset,
)
from .modeling import train_model_from_dataset
from .pinn import train, evaluate


def _safe_tag(name: str) -> str:
    """
    将字符串转换为安全的文件名标签

    将非字母数字字符替换为下划线，确保文件名有效性。

    Args:
        name (str): 原始字符串

    Returns:
        str: 安全的文件名字符串
    """
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)


def list_all_csv_headers() -> List[str]:
    """
    列出所有CSV数据文件的表头

    读取物理属性数据的CSV文件并返回列名列表。

    Returns:
        List[str]: CSV文件的表头列表
    """
    all_csv_path = PROJECT_ROOT / 'data' / 'physical' / 'all_csv_data.csv'
    with open(all_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers


def prepare_property_dataset(
    property_name: str,
    data_stage: str = 'raw',
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'corr_topk',
    fs_param: Any = None,
    msc_ref_mode: str = 'mean',
    snv_mode: str = 'standard',
    keep_exports: bool = False,
    baseline_zero_mode: str = 'none',
    despike_mode: str = 'none',
    baseline_zero_scope: str = 'cropped_spectrum',
) -> Dict[str, Any]:
    """
    准备特定属性的数据集

    从原始CSV数据和NIR光谱数据构建完整的分析数据集，
    包括黑白校正和元数据记录。

    Args:
        property_name (str): 要预测的属性名称
        data_stage (str): 数据处理阶段 ('raw', 'preprocessed', 'selected')
        preproc_mode (str): 预处理模式组合
        sg_order (int): SG滤波多项式阶数
        sg_window (int): SG滤波窗口大小
        fs_method (str): 特征选择方法
        fs_param (Any): 特征选择参数
        msc_ref_mode (str): MSC参考模式
        snv_mode (str): SNV变换模式
        keep_exports (bool): 是否保留导出文件
        baseline_zero_mode (str): 基线校正模式
        despike_mode (str): 去噪模式
        baseline_zero_scope (str): 基线校正范围

    Returns:
        Dict[str, Any]: 包含数据集信息和路径的字典
    """
    y, _ = build_property_vector_from_all_csv(
        'data/physical/all_csv_data.csv', property_name, 'data/NIR', average_num=1
    )
    X_full = black_white_processing(
        PROJECT_ROOT / 'data' / 'NIR',
        black_dir=PROJECT_ROOT / 'data' / 'Black white',
    )
    X_full = X_full[: y.shape[0], :]

    metadata = {
        'property_name': property_name,
        'data_stage': data_stage,
        'preproc_mode': preproc_mode,
        'sg_order': sg_order,
        'sg_window': sg_window,
        'fs_method': fs_method,
        'fs_param': fs_param,
        'msc_ref_mode': msc_ref_mode,
        'snv_mode': snv_mode,
        'baseline_zero_mode': baseline_zero_mode,
        'baseline_zero_scope': baseline_zero_scope,
        'despike_mode': despike_mode,
        'keep_exports': keep_exports,
        'preprocess_after_split': True,
    }
    if data_stage.lower().strip() == 'selected':
        metadata['selection_applied_in_training'] = True

    safe_name = _safe_tag(property_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_tag = f'{safe_name}/{data_stage.upper()}_{timestamp}'
    dataset = save_dataset(dataset_tag, X_full, y, metadata, keep_exports=keep_exports)
    return dataset


def compare_property_prediction_pipeline(
    property_name: str,
    filter_method: str = 'sg+msc+snv',
    feature_selection_method: str = 'pca',
    sg_order: int = 3,
    sg_window: int = 35,
    include_preprocessed_group: bool = False,
) -> List[Dict[str, Any]]:
    """
    比较不同模型和参数组合的属性预测性能

    对指定的属性运行完整的比较实验，包括多种特征选择参数
    和多种回归模型的组合测试。

    Args:
        property_name (str): 要预测的属性名称
        filter_method (str): 预处理方法组合
        feature_selection_method (str): 特征选择方法 ('cars', 'pca', 'corr_topk', 'spa')
        sg_order (int): SG滤波多项式阶数
        sg_window (int): SG滤波窗口大小
        include_preprocessed_group (bool): 是否包含预处理组的比较

    Returns:
        List[Dict[str, Any]]: 按测试集R²排序的模型结果列表

    Raises:
        ValueError: 当特征选择方法不支持时抛出
    """
    feature_selection_method = feature_selection_method.lower().strip()
    if feature_selection_method in {'cars', 'pca', 'corr_topk', 'spa'}:
        pass
    else:
        raise ValueError(f'Unsupported feature selection method: {feature_selection_method}')

    fs_param_grid = {
        'cars': [120, 150],
        'pca': [40, 80, 120, 160, 200],
        'corr_topk': [20, 40, 80, 120],
        'spa': [8, 12, 16, 20],
    }
    regressors = ['pls', 'pcr', 'svr', 'rf', 'gpr', 'knn', 'cnn', 'xgb']
    regressor_params = {
        'pls': {'max_lv': 300, 'cv_fold': 10},
        'pcr': {'max_pc': 300},
        'svr': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
        'rf': {'n_estimators': [100, 200], 'min_samples_leaf': [1, 5, 10]},
        'gpr': {'kernel': ['squaredexponential', 'matern32', 'matern52']},
        'knn': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'cnn': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu'], 'max_iter': [1000]},
        'xgb': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.3]},
    }

    results = []
    if include_preprocessed_group:
        dataset_pre = prepare_property_dataset(
            property_name,
            data_stage='preprocessed',
            preproc_mode=filter_method,
            sg_order=sg_order,
            sg_window=sg_window,
            fs_method='corr_topk',
            fs_param=None,
            msc_ref_mode='mean',
            snv_mode='standard',
            keep_exports=False,
        )
        for method in regressors:
            result = train_model_from_dataset(Path(dataset_pre['paths']['npz']), method, regressor_params[method])
            result['group'] = 'preprocessed'
            result['feature_selection'] = 'none'
            results.append(result)

    for fs_param in fs_param_grid[feature_selection_method]:
        dataset_sel = prepare_property_dataset(
            property_name,
            data_stage='selected',
            preproc_mode=filter_method,
            sg_order=sg_order,
            sg_window=sg_window,
            fs_method=feature_selection_method,
            fs_param=fs_param,
            msc_ref_mode='mean',
            snv_mode='standard',
            keep_exports=False,
        )
        for method in regressors:
            result = train_model_from_dataset(Path(dataset_sel['paths']['npz']), method, regressor_params[method])
            result['group'] = 'selected'
            result['feature_selection'] = feature_selection_method
            result['fs_param'] = fs_param
            results.append(result)

    results.sort(key=lambda info: (info['group'], -info['metrics']['test']['r2'], info['metrics']['test']['rmse']))
    return results


def process_by_all_csv_feature_selection_only(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'pca',
    fs_param: Any = None,
) -> Dict[str, Any]:
    """
    仅执行特征选择的数据处理流程

    准备经过特征选择的数据集，用于后续建模。

    Args:
        property_name (str): 属性名称
        preproc_mode (str): 预处理模式
        sg_order (int): SG滤波阶数
        sg_window (int): SG滤波窗口
        fs_method (str): 特征选择方法
        fs_param (Any): 特征选择参数

    Returns:
        Dict[str, Any]: 数据集信息字典
    """
    return prepare_property_dataset(
        property_name,
        data_stage='selected',
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        fs_method=fs_method,
        fs_param=fs_param,
    )


def process_by_all_csv_with_feature_selection(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'cars',
    fs_param: Any = None,
) -> Dict[str, Any]:
    """
    执行完整的数据处理流程（包括特征选择）

    准备经过预处理和特征选择的数据集。

    Args:
        property_name (str): 属性名称
        preproc_mode (str): 预处理模式
        sg_order (int): SG滤波阶数
        sg_window (int): SG滤波窗口
        fs_method (str): 特征选择方法
        fs_param (Any): 特征选择参数

    Returns:
        Dict[str, Any]: 数据集信息字典
    """
    return prepare_property_dataset(
        property_name,
        data_stage='selected',
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        fs_method=fs_method,
        fs_param=fs_param,
    )


def _default_fs_param(method_name: str) -> Any:
    """
    获取特征选择方法的默认参数

    Args:
        method_name (str): 特征选择方法名称

    Returns:
        Any: 默认参数值
    """
    method_name = method_name.lower().strip()
    if method_name == 'rfe':
        return 40
    if method_name == 'spa':
        return 20
    if method_name == 'pca':
        return 40
    if method_name == 'cars':
        return 120
    return None


def process_by_all_csv_header(
    property_name: str,
    method_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
) -> List[Dict[str, Any]]:
    fs_param = _default_fs_param(method_name)
    dataset = prepare_property_dataset(
        property_name,
        data_stage='selected',
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        fs_method=method_name,
        fs_param=fs_param,
    )
    result = train_model_from_dataset(Path(dataset['paths']['npz']), 'pls', None)
    result['feature_selection'] = method_name
    result['fs_param'] = fs_param
    return [result]


def run_property_prediction(
    property_name: str,
    method_name: str = 'spa',
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: Optional[str] = None,
    fs_param: Any = None,
) -> Any:
    if property_name.strip().lower() == 'list':
        return list_all_csv_headers()
    if method_name.lower().strip() == 'select':
        return process_by_all_csv_feature_selection_only(property_name, preproc_mode, sg_order, sg_window, fs_method or 'pca', fs_param)
    if method_name.lower().strip() == 'fs':
        return process_by_all_csv_with_feature_selection(property_name, preproc_mode, sg_order, sg_window, fs_method or 'cars', fs_param)
    return process_by_all_csv_header(property_name, method_name, preproc_mode, sg_order, sg_window)


def run_pinn_prediction(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
) -> Dict[str, Any]:
    """
    运行PINN（物理信息神经网络）属性预测管道

    PINN结合了数据驱动的学习和物理知识约束，能够：
    - 利用Arrhenius方程建模化学反应动力学
    - 处理多模态数据（NIR + E-nose）
    - 提供更好的泛化能力和物理可解释性
    - 特别适用于时间相关的降解过程预测

    Args:
        property_name (str): 目标属性名称，如'a*'或'b*'
        preproc_mode (str): 预处理模式，默认'sg+msc'
        sg_order (int): Savitzky-Golay多项式阶数，默认3
        sg_window (int): Savitzky-Golay窗口长度，默认15

    Returns:
        Dict[str, Any]: 包含训练历史、评估指标和模型性能的字典
    """
    print(f"开始PINN预测管道 - 属性: {property_name}")

    # 数据准备
    print("准备PINN数据集...")
    from .pinn.dataset import prepare_pinn_dataset

    train_data, val_data, test_data = prepare_pinn_dataset(
        property_name=property_name,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
    )

    # 模型训练
    print("训练PINN模型...")
    from .pinn.model import PINNNetwork
    from .pinn.loss import PINNLoss

    model = PINNNetwork()
    loss_fn = PINNLoss()

    history = train.train_pinn_two_stage(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        epochs=3000,  # PINN通常需要更多训练轮数
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
    )

    # 模型评估
    print("评估PINN模型...")
    metrics = evaluate.evaluate_pinn(
        model=model,
        X_nir=test_data['X_nir'],
        X_enose=test_data['X_enose'],
        times=test_data['times'],
        temperatures=test_data['temperatures'],
        y_true=test_data['y'].numpy(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # 可视化结果
    print("生成可视化结果...")
    evaluate.plot_kinetics(
        model=model,
        X_nir=test_data['X_nir'][:3],  # 只显示前3个样本
        X_enose=test_data['X_enose'][:3],
        times_seq=test_data['times'].numpy()[:3],
        temperatures=test_data['temperatures'][:3],
        y_true=test_data['y'].numpy()[:3],
        save_path=f"pinn_kinetics_{property_name}.png",
    )

    result = {
        'property': property_name,
        'model_type': 'PINN',
        'training_history': history,
        'test_metrics': metrics,
        'preprocessing': {
            'mode': preproc_mode,
            'sg_order': sg_order,
            'sg_window': sg_window,
        }
    }

    print(f"PINN预测完成 - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    return result
