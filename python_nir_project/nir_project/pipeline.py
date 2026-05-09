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
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import (
    PROJECT_ROOT,
    build_property_vector_from_all_csv,
    black_white_processing,
    save_dataset,
)
from .modeling import train_model_from_dataset
from .pinn import train, evaluate


def _merge_dict(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two optional dictionaries while keeping the originals unchanged."""
    merged = dict(base or {})
    merged.update(override or {})
    return merged


def _json_safe(value: Any) -> Any:
    """Convert nested values into JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _project_data_sources() -> Dict[str, str]:
    """Describe the project data sources used to build datasets."""
    return {
        'property_table_csv': str(PROJECT_ROOT / 'data' / 'physical' / 'all_csv_data.csv'),
        'nir_data_dir': str(PROJECT_ROOT / 'data' / 'NIR'),
        'black_white_dir': str(PROJECT_ROOT / 'data' / 'Black white'),
    }


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


def _parse_property_names(value: str) -> List[str]:
    """将逗号分隔的属性名字符串解析为列表。"""
    return [item.strip() for item in value.split(',') if item.strip()]


def _result_to_summary_row(property_name: str, result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """将单个模型结果展平为摘要表的一行。"""
    metrics = result['metrics']
    return {
        'property_name': property_name,
        'rank': rank,
        'group': result.get('group', ''),
        'feature_selection': result.get('feature_selection', ''),
        'fs_param': result.get('fs_param'),
        'model': result.get('method', ''),
        'train_r2': metrics['train']['r2'],
        'train_rmse': metrics['train']['rmse'],
        'test_r2': metrics['test']['r2'],
        'test_rmse': metrics['test']['rmse'],
        'dataset_tag': result.get('dataset_tag', ''),
        'plot_path': result.get('plot_path', ''),
        'best_param_json': json.dumps(result.get('best_param', {}), ensure_ascii=False, sort_keys=True),
    }


def _append_prediction_rows(rows: List[Dict[str, Any]], property_name: str, result: Dict[str, Any]) -> None:
    """将训练集和测试集预测结果追加到逐样本预测表。"""
    split_config = [
        ('train', result.get('ytrain', []), result.get('ypred_train', [])),
        ('test', result.get('ytest', []), result.get('ypred', [])),
    ]
    for split_name, truths, preds in split_config:
        for sample_index, (truth, pred) in enumerate(zip(truths, preds), start=1):
            rows.append(
                {
                    'property_name': property_name,
                    'group': result.get('group', ''),
                    'feature_selection': result.get('feature_selection', ''),
                    'fs_param': result.get('fs_param'),
                    'model': result.get('method', ''),
                    'split': split_name,
                    'sample_index': sample_index,
                    'y_true': truth,
                    'y_pred': pred,
                }
            )


def _save_property_comparison_plot(property_name: str, rows: List[Dict[str, Any]], output_dir: Path) -> str:
    """保存单个属性下不同算法的对比总览图。"""
    df = pd.DataFrame(rows).sort_values(by=['test_r2', 'test_rmse'], ascending=[False, True])
    labels = [
        f"{row['model']}|{row['feature_selection']}|{row['fs_param']}"
        for _, row in df.iterrows()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 0.55), 6))
    axes[0].bar(range(len(labels)), df['test_r2'], color='#2C7FB8')
    axes[0].set_title(f'{property_name} 测试集 R2 对比')
    axes[0].set_ylabel('R2')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(labels)), df['test_rmse'], color='#D95F0E')
    axes[1].set_title(f'{property_name} 测试集 RMSE 对比')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f'{_safe_tag(property_name)}_algorithm_comparison.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def _persist_batch_compare_outputs(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    property_names: List[str],
    preproc_mode: str,
    sg_order: int,
    sg_window: int,
    feature_selection_method: str,
    include_preprocessed_group: bool,
    traditional_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """将批量对比的中间结果立即落盘，避免中途中断后丢失数据。"""
    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    if summary_df.empty:
        parameter_df = pd.DataFrame(
            columns=[
                'property_name',
                'rank',
                'group',
                'feature_selection',
                'fs_param',
                'model',
                'dataset_tag',
                'best_param_json',
            ]
        )
    else:
        parameter_df = summary_df[
            [
                'property_name',
                'rank',
                'group',
                'feature_selection',
                'fs_param',
                'model',
                'dataset_tag',
                'best_param_json',
            ]
        ].copy()

    summary_path = output_dir / 'metrics_summary.csv'
    prediction_path = output_dir / 'prediction_table.csv'
    parameter_path = output_dir / 'parameter_table.csv'
    config_path = output_dir / 'run_config.json'

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    prediction_df.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    parameter_df.to_csv(parameter_path, index=False, encoding='utf-8-sig')
    config_path.write_text(
        json.dumps(
            {
                'model_type': 'traditional',
                'property_names': property_names,
                'preproc_mode': preproc_mode,
                'sg_order': sg_order,
                'sg_window': sg_window,
                'feature_selection_method': feature_selection_method,
                'include_preprocessed_group': include_preprocessed_group,
                'data_sources': _project_data_sources(),
                'traditional_config': _json_safe(traditional_config or {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    return {
        'metrics_summary_path': str(summary_path),
        'prediction_table_path': str(prediction_path),
        'parameter_table_path': str(parameter_path),
        'config_path': str(config_path),
    }


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
    result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    fs_param_grid: Optional[Dict[str, List[Any]]] = None,
    regressors: Optional[List[str]] = None,
    regressor_params: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_options: Optional[Dict[str, Any]] = None,
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

    resolved_fs_param_grid = _merge_dict({
        'cars': [120, 150],
        'pca': [40, 80, 120, 160, 200],
        'corr_topk': [20, 40, 80, 120],
        'spa': [8, 12, 16, 20],
    }, fs_param_grid)
    resolved_regressors = list(regressors or ['pls', 'pcr', 'svr', 'rf', 'gpr', 'knn', 'cnn', 'xgb'])
    resolved_regressor_params = _merge_dict({
        'pls': {'max_lv': 300, 'cv_fold': 10},
        'pcr': {'max_pc': 300},
        'svr': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
        'rf': {'n_estimators': [100, 200], 'min_samples_leaf': [1, 5, 10]},
        'gpr': {'kernel': ['squaredexponential', 'matern32', 'matern52']},
        'knn': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'cnn': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu'], 'max_iter': [1000]},
        'xgb': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.3]},
    }, regressor_params)
    dataset_options = dict(dataset_options or {})

    results = []
    total_jobs = len(resolved_regressors) * len(resolved_fs_param_grid[feature_selection_method])
    if include_preprocessed_group:
        total_jobs += len(resolved_regressors)
    completed_jobs = 0

    print(
        f"[{property_name}] 开始算法对比，共 {total_jobs} 个训练任务，"
        f"特征选择方法: {feature_selection_method}，预处理: {filter_method}"
    )

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
            **dataset_options,
        )
        for method in resolved_regressors:
            completed_jobs += 1
            print(
                f"[{property_name}] 进度 {completed_jobs}/{total_jobs} | "
                f"数据组=preprocessed | 模型={method}"
            )
            result = train_model_from_dataset(Path(dataset_pre['paths']['npz']), method, resolved_regressor_params[method])
            result['group'] = 'preprocessed'
            result['feature_selection'] = 'none'
            results.append(result)
            if result_callback is not None:
                result_callback(result)
            print(
                f"[{property_name}] 完成 {method} | "
                f"Test R2={result['metrics']['test']['r2']:.4f} | "
                f"RMSE={result['metrics']['test']['rmse']:.4f}"
            )

    for fs_param in resolved_fs_param_grid[feature_selection_method]:
        print(f"[{property_name}] 准备特征选择参数 fs_param={fs_param}")
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
            **dataset_options,
        )
        for method in resolved_regressors:
            completed_jobs += 1
            print(
                f"[{property_name}] 进度 {completed_jobs}/{total_jobs} | "
                f"数据组=selected | fs_param={fs_param} | 模型={method}"
            )
            result = train_model_from_dataset(Path(dataset_sel['paths']['npz']), method, resolved_regressor_params[method])
            result['group'] = 'selected'
            result['feature_selection'] = feature_selection_method
            result['fs_param'] = fs_param
            results.append(result)
            if result_callback is not None:
                result_callback(result)
            print(
                f"[{property_name}] 完成 {method} | fs_param={fs_param} | "
                f"Test R2={result['metrics']['test']['r2']:.4f} | "
                f"RMSE={result['metrics']['test']['rmse']:.4f}"
            )

    results.sort(key=lambda info: (info['group'], -info['metrics']['test']['r2'], info['metrics']['test']['rmse']))
    if results:
        best_result = max(results, key=lambda info: info['metrics']['test']['r2'])
        print(
            f"[{property_name}] 对比完成，最佳模型={best_result['method']} | "
            f"Test R2={best_result['metrics']['test']['r2']:.4f} | "
            f"RMSE={best_result['metrics']['test']['rmse']:.4f}"
        )
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


def compare_all_feature_selection_pipeline(
    property_name: str,
    filter_method: str = 'sg+msc+snv',
    sg_order: int = 3,
    sg_window: int = 35,
    include_preprocessed_group: bool = False,
    result_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    fs_param_grid: Optional[Dict[str, List[Any]]] = None,
    regressors: Optional[List[str]] = None,
    regressor_params: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """依次运行全部特征选择方法，并汇总模型对比结果。"""
    all_results: List[Dict[str, Any]] = []
    supported_fs_methods = ['cars', 'pca', 'corr_topk', 'spa']

    print(f"[{property_name}] 开始全部特征选择对比，共 {len(supported_fs_methods)} 类特征选择方法")
    for index, current_fs_method in enumerate(supported_fs_methods, start=1):
        print(f"[{property_name}] 特征选择方法进度 {index}/{len(supported_fs_methods)} | fs={current_fs_method}")
        method_results = compare_property_prediction_pipeline(
            property_name=property_name,
            filter_method=filter_method,
            feature_selection_method=current_fs_method,
            sg_order=sg_order,
            sg_window=sg_window,
            include_preprocessed_group=include_preprocessed_group and index == 1,
            result_callback=result_callback,
            fs_param_grid=fs_param_grid,
            regressors=regressors,
            regressor_params=regressor_params,
            dataset_options=dataset_options,
        )
        all_results.extend(method_results)

    all_results.sort(key=lambda info: (info['group'], -info['metrics']['test']['r2'], info['metrics']['test']['rmse']))
    if all_results:
        best_result = max(all_results, key=lambda info: info['metrics']['test']['r2'])
        print(
            f"[{property_name}] 全部特征选择对比完成，最佳组合="
            f"{best_result.get('feature_selection', 'unknown')} + {best_result['method']} | "
            f"Test R2={best_result['metrics']['test']['r2']:.4f} | "
            f"RMSE={best_result['metrics']['test']['rmse']:.4f}"
        )
    return all_results


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


def train_single_property_prediction(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'corr_topk',
    fs_param: Any = None,
    regressor: str = 'pls',
    regressor_param: Optional[Dict[str, Any]] = None,
    dataset_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """执行单次传统建模训练。"""
    resolved_fs_param = fs_param if fs_param is not None else _default_fs_param(fs_method)
    dataset = prepare_property_dataset(
        property_name,
        data_stage='selected',
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        fs_method=fs_method,
        fs_param=resolved_fs_param,
        **dict(dataset_options or {}),
    )
    result = train_model_from_dataset(Path(dataset['paths']['npz']), regressor, regressor_param)
    result['group'] = 'selected'
    result['feature_selection'] = fs_method
    result['fs_param'] = resolved_fs_param
    result['dataset_metadata'] = dataset.get('metadata', {})
    return result


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
    if method_name == 'corr_topk':
        return 20
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


def batch_compare_property_prediction(
    property_names: List[str],
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    feature_selection_method: str = 'pca',
    include_preprocessed_group: bool = False,
    fs_param_grid: Optional[Dict[str, List[Any]]] = None,
    regressors: Optional[List[str]] = None,
    regressor_params: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_options: Optional[Dict[str, Any]] = None,
    traditional_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """批量处理多个属性，并输出算法对比结果、预测图和参数表。"""
    property_names = [name.strip() for name in property_names if name.strip()]
    if not property_names:
        raise ValueError('property_names 不能为空')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_batch_name = '_'.join(_safe_tag(name) for name in property_names[:3])
    if len(property_names) > 3:
        safe_batch_name += f'_plus_{len(property_names) - 3}'
    output_dir = PROJECT_ROOT / 'result' / 'batch_compare' / f'{timestamp}_{safe_batch_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    property_plot_paths: Dict[str, str] = {}
    raw_results: Dict[str, List[Dict[str, Any]]] = {}
    _persist_batch_compare_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        property_names=property_names,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        feature_selection_method=feature_selection_method,
        include_preprocessed_group=include_preprocessed_group,
        traditional_config=traditional_config,
    )

    print(
        f"开始批量算法对比，共 {len(property_names)} 个属性，"
        f"预处理={preproc_mode}，特征选择={feature_selection_method}"
    )

    for property_index, property_name in enumerate(property_names, start=1):
        print(f"========== 属性进度 {property_index}/{len(property_names)}：{property_name} ==========")
        property_rows: List[Dict[str, Any]] = []

        def persist_single_result(result: Dict[str, Any]) -> None:
            rank = len(property_rows) + 1
            row = _result_to_summary_row(property_name, result, rank)
            summary_rows.append(row)
            property_rows.append(row)
            _append_prediction_rows(prediction_rows, property_name, result)
            persisted_paths = _persist_batch_compare_outputs(
                output_dir=output_dir,
                summary_rows=summary_rows,
                prediction_rows=prediction_rows,
                property_names=property_names,
                preproc_mode=preproc_mode,
                sg_order=sg_order,
                sg_window=sg_window,
                feature_selection_method=feature_selection_method,
                include_preprocessed_group=include_preprocessed_group,
                traditional_config=traditional_config,
            )
            print(f"[{property_name}] 中间结果已保存: {persisted_paths['metrics_summary_path']}")

        if feature_selection_method.lower().strip() == 'all':
            results = compare_all_feature_selection_pipeline(
                property_name=property_name,
                filter_method=preproc_mode,
                sg_order=sg_order,
                sg_window=sg_window,
                include_preprocessed_group=include_preprocessed_group,
                result_callback=persist_single_result,
                fs_param_grid=fs_param_grid,
                regressors=regressors,
                regressor_params=regressor_params,
                dataset_options=dataset_options,
            )
        else:
            results = compare_property_prediction_pipeline(
                property_name=property_name,
                filter_method=preproc_mode,
                feature_selection_method=feature_selection_method,
                sg_order=sg_order,
                sg_window=sg_window,
                include_preprocessed_group=include_preprocessed_group,
                result_callback=persist_single_result,
                fs_param_grid=fs_param_grid,
                regressors=regressors,
                regressor_params=regressor_params,
                dataset_options=dataset_options,
            )
        raw_results[property_name] = results
        if not property_rows:
            for rank, result in enumerate(results, start=1):
                row = _result_to_summary_row(property_name, result, rank)
                summary_rows.append(row)
                property_rows.append(row)
                _append_prediction_rows(prediction_rows, property_name, result)
        property_plot_paths[property_name] = _save_property_comparison_plot(property_name, property_rows, output_dir)
        persisted_paths = _persist_batch_compare_outputs(
            output_dir=output_dir,
            summary_rows=summary_rows,
            prediction_rows=prediction_rows,
            property_names=property_names,
            preproc_mode=preproc_mode,
            sg_order=sg_order,
            sg_window=sg_window,
            feature_selection_method=feature_selection_method,
            include_preprocessed_group=include_preprocessed_group,
            traditional_config=traditional_config,
        )
        print(f"[{property_name}] 中间结果已保存: {persisted_paths['metrics_summary_path']}")
        print(f"[{property_name}] 总览图已保存: {property_plot_paths[property_name]}")

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    parameter_df = summary_df[
        [
            'property_name',
            'rank',
            'group',
            'feature_selection',
            'fs_param',
            'model',
            'dataset_tag',
            'best_param_json',
        ]
    ].copy()

    summary_path = output_dir / 'metrics_summary.csv'
    prediction_path = output_dir / 'prediction_table.csv'
    parameter_path = output_dir / 'parameter_table.csv'
    config_path = output_dir / 'run_config.json'

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    prediction_df.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    parameter_df.to_csv(parameter_path, index=False, encoding='utf-8-sig')
    config_path.write_text(
        json.dumps(
            {
                'property_names': property_names,
                'model_type': 'traditional',
                'preproc_mode': preproc_mode,
                'sg_order': sg_order,
                'sg_window': sg_window,
                'feature_selection_method': feature_selection_method,
                'include_preprocessed_group': include_preprocessed_group,
                'data_sources': _project_data_sources(),
                'traditional_config': _json_safe(traditional_config or {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    print(f"批量对比完成，结果目录: {output_dir}")
    print(f"指标汇总表: {summary_path}")
    print(f"参数汇总表: {parameter_path}")
    print(f"预测结果表: {prediction_path}")

    return {
        'output_dir': str(output_dir),
        'metrics_summary_path': str(summary_path),
        'prediction_table_path': str(prediction_path),
        'parameter_table_path': str(parameter_path),
        'config_path': str(config_path),
        'property_plot_paths': property_plot_paths,
        'results': raw_results,
    }


def _pinn_fs_param_grid(feature_selection_method: str, explicit_fs_param: Any = None) -> Dict[str, List[Any]]:
    """返回 PINN 特征选择对比计划。"""
    grids: Dict[str, List[Any]] = {
        'cars': [120, 150],
        'pca': [40, 80, 120, 160, 200],
        'corr_topk': [20, 40, 80, 120],
        'spa': [8, 12, 16, 20],
    }
    resolved_method = (feature_selection_method or 'corr_topk').lower().strip()
    if resolved_method == 'all':
        if explicit_fs_param is not None:
            return {name: [explicit_fs_param] for name in grids}
        return grids
    if resolved_method not in grids:
        raise ValueError(f'Unsupported PINN feature selection method: {feature_selection_method}')
    if explicit_fs_param is not None:
        return {resolved_method: [explicit_fs_param]}
    return {resolved_method: grids[resolved_method]}


def _resolve_pinn_fs_param_grid(
    feature_selection_method: str,
    explicit_fs_param: Any = None,
    custom_grid: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, List[Any]]:
    """Resolve PINN feature-selection plans with optional user overrides."""
    default_plan = _pinn_fs_param_grid(feature_selection_method, explicit_fs_param)
    if not custom_grid:
        return default_plan

    resolved_method = (feature_selection_method or 'corr_topk').lower().strip()
    normalized_custom = {str(name).lower().strip(): list(values) for name, values in custom_grid.items()}

    if resolved_method == 'all':
        if explicit_fs_param is not None:
            return default_plan
        merged: Dict[str, List[Any]] = {}
        for method_name, default_values in default_plan.items():
            merged[method_name] = normalized_custom.get(method_name, default_values)
        return merged

    if explicit_fs_param is not None:
        return default_plan
    if resolved_method in normalized_custom:
        return {resolved_method: normalized_custom[resolved_method]}
    return default_plan


def _pinn_result_to_summary_row(property_name: str, result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """将 PINN 单次运行结果整理为汇总表一行。"""
    metrics = result['test_metrics']
    metadata = result.get('dataset_metadata', {}) or {}
    return {
        'property_name': property_name,
        'rank': rank,
        'feature_selection': metadata.get('fs_method'),
        'fs_param': metadata.get('fs_param'),
        'original_feature_count': metadata.get('original_feature_count'),
        'selected_feature_count': metadata.get('selected_feature_count'),
        'r2': metrics.get('r2'),
        'rmse': metrics.get('rmse'),
        'mae': metrics.get('mae'),
        'rpd': metrics.get('rpd'),
        'result_dir': result.get('result_dir', ''),
        'prediction_table_path': result.get('prediction_table_path', ''),
        'metrics_path': result.get('metrics_path', ''),
        'training_history_path': result.get('training_history_path', ''),
        'run_config_path': result.get('run_config_path', ''),
        'selected_feature_idx_json': json.dumps(metadata.get('selected_feature_idx'), ensure_ascii=False),
    }


def _append_pinn_prediction_rows(rows: List[Dict[str, Any]], property_name: str, result: Dict[str, Any]) -> None:
    """读取 PINN 预测结果表并追加到批量汇总。"""
    prediction_path = result.get('prediction_table_path')
    if not prediction_path:
        return
    df = pd.read_csv(prediction_path)
    metadata = result.get('dataset_metadata', {}) or {}
    for record in df.to_dict(orient='records'):
        record['property_name'] = property_name
        record['feature_selection'] = metadata.get('fs_method')
        record['fs_param'] = metadata.get('fs_param')
        rows.append(record)


def _persist_pinn_compare_outputs(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    prediction_rows: List[Dict[str, Any]],
    property_names: List[str],
    preproc_mode: str,
    sg_order: int,
    sg_window: int,
    feature_selection_method: str,
    fs_param: Any,
    pinn_config: Optional[Dict[str, Any]] = None,
    compare_plan: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, str]:
    """将 PINN compare 的中间结果持续写盘。"""
    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    parameter_df = summary_df[
        [
            'property_name',
            'rank',
            'feature_selection',
            'fs_param',
            'original_feature_count',
            'selected_feature_count',
            'selected_feature_idx_json',
            'result_dir',
            'run_config_path',
        ]
    ].copy() if not summary_df.empty else pd.DataFrame(
        columns=[
            'property_name',
            'rank',
            'feature_selection',
            'fs_param',
            'original_feature_count',
            'selected_feature_count',
            'selected_feature_idx_json',
            'result_dir',
            'run_config_path',
        ]
    )

    summary_path = output_dir / 'metrics_summary.csv'
    prediction_path = output_dir / 'prediction_table.csv'
    parameter_path = output_dir / 'parameter_table.csv'
    config_path = output_dir / 'run_config.json'

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    prediction_df.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    parameter_df.to_csv(parameter_path, index=False, encoding='utf-8-sig')
    config_path.write_text(
        json.dumps(
            {
                'property_names': property_names,
                'model_type': 'PINN',
                'preproc_mode': preproc_mode,
                'sg_order': sg_order,
                'sg_window': sg_window,
                'feature_selection_method': feature_selection_method,
                'fs_param': fs_param,
                'data_sources': _project_data_sources(),
                'pinn_config': _json_safe(pinn_config or {}),
                'compare_plan': _json_safe(compare_plan or {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return {
        'metrics_summary_path': str(summary_path),
        'prediction_table_path': str(prediction_path),
        'parameter_table_path': str(parameter_path),
        'config_path': str(config_path),
    }


def batch_compare_pinn_prediction(
    property_names: List[str],
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    feature_selection_method: str = 'corr_topk',
    fs_param: Any = None,
    pinn_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """批量比较 PINN 在不同特征选择方法/特征维度下的表现。"""
    property_names = [name.strip() for name in property_names if name.strip()]
    if not property_names:
        raise ValueError('property_names 不能为空')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_batch_name = '_'.join(_safe_tag(name) for name in property_names[:3])
    if len(property_names) > 3:
        safe_batch_name += f'_plus_{len(property_names) - 3}'
    pinn_config = dict(pinn_config or {})
    input_mode = (pinn_config.get('input_mode') or 'fusion').strip().lower()
    pinn_fs_param_grid = pinn_config.get('fs_param_grid')

    output_dir = PROJECT_ROOT / 'result' / 'pinn_compare' / f'{timestamp}_{safe_batch_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_mode == 'enose':
        plan = {'none': [None]}
    else:
        plan = _resolve_pinn_fs_param_grid(feature_selection_method, fs_param, pinn_fs_param_grid)
    total_jobs = len(property_names) * sum(len(params) for params in plan.values())
    completed_jobs = 0
    summary_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    raw_results: Dict[str, List[Dict[str, Any]]] = {}

    _persist_pinn_compare_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        property_names=property_names,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        feature_selection_method=feature_selection_method,
        fs_param=fs_param,
        pinn_config=pinn_config,
        compare_plan=plan,
    )

    print(
        f"开始PINN特征选择对比，共 {len(property_names)} 个属性，"
        f"特征选择={feature_selection_method}，总任务数={total_jobs}"
    )

    for property_name in property_names:
        property_results: List[Dict[str, Any]] = []
        for current_method, param_list in plan.items():
            print(f"[{property_name}] 开始 PINN 特征选择方法对比：fs={current_method}")
            for current_param in param_list:
                completed_jobs += 1
                print(
                    f"[{property_name}] PINN 对比进度 {completed_jobs}/{total_jobs} | "
                    f"fs_method={current_method} | fs_param={current_param}"
                )
                result = run_pinn_prediction(
                    property_name=property_name,
                    preproc_mode=preproc_mode,
                    sg_order=sg_order,
                    sg_window=sg_window,
                    fs_method=current_method,
                    fs_param=current_param,
                    pinn_config=pinn_config,
                )
                property_results.append(result)
                summary_rows.append(_pinn_result_to_summary_row(property_name, result, len(property_results)))
                _append_pinn_prediction_rows(prediction_rows, property_name, result)
                persisted_paths = _persist_pinn_compare_outputs(
                    output_dir=output_dir,
                    summary_rows=summary_rows,
                    prediction_rows=prediction_rows,
                    property_names=property_names,
                    preproc_mode=preproc_mode,
                    sg_order=sg_order,
                    sg_window=sg_window,
                    feature_selection_method=feature_selection_method,
                    fs_param=fs_param,
                    pinn_config=pinn_config,
                    compare_plan=plan,
                )
                print(f"[{property_name}] PINN 中间结果已保存: {persisted_paths['metrics_summary_path']}")
        property_results.sort(key=lambda item: item['test_metrics']['r2'], reverse=True)
        raw_results[property_name] = property_results

    persisted_paths = _persist_pinn_compare_outputs(
        output_dir=output_dir,
        summary_rows=summary_rows,
        prediction_rows=prediction_rows,
        property_names=property_names,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        feature_selection_method=feature_selection_method,
        fs_param=fs_param,
        pinn_config=pinn_config,
        compare_plan=plan,
    )
    print(f"PINN 特征选择对比完成，结果目录: {output_dir}")

    return {
        'output_dir': str(output_dir),
        'metrics_summary_path': persisted_paths['metrics_summary_path'],
        'prediction_table_path': persisted_paths['prediction_table_path'],
        'parameter_table_path': persisted_paths['parameter_table_path'],
        'config_path': persisted_paths['config_path'],
        'results': raw_results,
    }


def run_property_prediction(
    property_name: str,
    mode: str = 'train',
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: Optional[str] = None,
    fs_param: Any = None,
    include_preprocessed_group: bool = False,
    traditional_config: Optional[Dict[str, Any]] = None,
) -> Any:
    traditional_config = dict(traditional_config or {})
    dataset_options = traditional_config.get('dataset_options')
    fs_param_grid = traditional_config.get('fs_param_grid')
    regressors = traditional_config.get('regressors')
    regressor_params = traditional_config.get('regressor_params')
    train_regressor = traditional_config.get('train_regressor', 'pls')
    train_regressor_param = traditional_config.get('train_regressor_param')

    resolved_mode = (mode or 'train').lower().strip()
    if property_name.strip().lower() == 'list':
        return list_all_csv_headers()
    if resolved_mode == 'compare':
        return batch_compare_property_prediction(
            property_names=_parse_property_names(property_name),
            preproc_mode=preproc_mode,
            sg_order=sg_order,
            sg_window=sg_window,
            feature_selection_method=fs_method or 'pca',
            include_preprocessed_group=include_preprocessed_group,
            fs_param_grid=fs_param_grid,
            regressors=regressors,
            regressor_params=regressor_params,
            dataset_options=dataset_options,
            traditional_config=traditional_config,
        )
    if resolved_mode == 'select':
        return process_by_all_csv_feature_selection_only(
            property_name,
            preproc_mode,
            sg_order,
            sg_window,
            fs_method or 'pca',
            fs_param,
        )
    if resolved_mode == 'train':
        return train_single_property_prediction(
            property_name=property_name,
            preproc_mode=preproc_mode,
            sg_order=sg_order,
            sg_window=sg_window,
            fs_method=fs_method or 'corr_topk',
            fs_param=fs_param,
            regressor=train_regressor,
            regressor_param=train_regressor_param,
            dataset_options=dataset_options,
        )
    raise ValueError(f'Unsupported mode: {resolved_mode}')


def run_pinn_prediction(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: Optional[str] = None,
    fs_param: Optional[int] = None,
    pinn_config: Optional[Dict[str, Any]] = None,
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
    pinn_config = dict(pinn_config or {})
    input_mode = (pinn_config.get('input_mode') or 'fusion').strip().lower()
    dataset_options = dict(pinn_config.get('dataset_options') or {})
    model_params = dict(pinn_config.get('model_params') or {})
    loss_params = dict(pinn_config.get('loss_params') or {})
    train_params = dict(pinn_config.get('train_params') or {})
    from .pinn.dataset import prepare_pinn_dataset_from_project_data

    if input_mode == 'enose':
        resolved_fs_method = 'none'
        resolved_fs_param = None
    else:
        resolved_fs_method = (fs_method or 'corr_topk').strip().lower()
        resolved_fs_param = fs_param if fs_param is not None else _default_fs_param(resolved_fs_method)

    dataset = prepare_pinn_dataset_from_project_data(
        property_name=property_name,
        preproc_mode=preproc_mode,
        sg_order=sg_order,
        sg_window=sg_window,
        fs_method=resolved_fs_method,
        fs_param=resolved_fs_param,
        input_mode=input_mode,
        **dataset_options,
    )
    dataset_metadata = dataset.get('metadata', {})
    print(
        'PINN特征选择: '
        f"fs_method={dataset_metadata.get('fs_method') or 'none'} | "
        f"fs_param={dataset_metadata.get('fs_param')} | "
        f"原始波段数={dataset_metadata.get('original_feature_count')} | "
        f"筛后波段数={dataset_metadata.get('selected_feature_count')} | "
        f"选中索引={dataset_metadata.get('selected_feature_idx')}"
    )
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    collocation_data = dataset['collocation']

    # 模型训练
    print("训练PINN模型...")
    from .pinn.model import PINNNetwork
    from .pinn.loss import PINNLoss

    model = PINNNetwork(
        nir_dim=int(train_data['X_nir'].shape[1]),
        nose_dim=int(train_data['X_enose'].shape[1]),
        **model_params,
    )
    loss_fn = PINNLoss(**loss_params)

    history = train.train_pinn_two_stage(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        collocation_data=collocation_data,
        total_epochs=3000,  # PINN通常需要更多训练轮数
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=True,
        **train_params,
    )

    # 模型评估
    print("评估PINN模型...")
    metrics, y_pred = evaluate.evaluate_pinn(
        model=model,
        X_nir=test_data['X_nir'],
        X_enose=test_data['X_enose'],
        times=test_data['times'],
        temperatures=test_data['temperatures'],
        y_true=test_data['y'].numpy(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        return_predictions=True,
    )

    # 可视化结果
    print("生成可视化结果...")
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pinn_result_dir = PROJECT_ROOT / 'result' / 'pinn' / f'{run_stamp}_{_safe_tag(property_name)}'
    pinn_result_dir.mkdir(parents=True, exist_ok=True)
    kinetics_plots = evaluate.plot_sparse_observation_kinetics(
        model=model,
        X_nir=test_data['X_nir'],
        X_enose=test_data['X_enose'],
        times_seq=test_data['times'].numpy().reshape(-1),
        temperatures=test_data['temperatures'],
        y_true=test_data['y'].numpy().reshape(-1),
        save_path=str(pinn_result_dir / f"pinn_kinetics_{_safe_tag(property_name)}.png"),
    )
    scatter_plot_path = evaluate.plot_prediction_scatter(
        y_true=test_data['y'].detach().cpu().numpy().reshape(-1),
        y_pred=y_pred.reshape(-1),
        metrics=metrics,
        save_path=str(pinn_result_dir / f"pinn_prediction_scatter_{_safe_tag(property_name)}.png"),
    )

    prediction_df = pd.DataFrame(
        {
            'sample_index': list(range(1, len(y_pred) + 1)),
            'time': test_data['times'].detach().cpu().numpy().reshape(-1),
            'temperature_kelvin': test_data['temperatures'].detach().cpu().numpy().reshape(-1),
            'y_true': test_data['y'].detach().cpu().numpy().reshape(-1),
            'y_pred': y_pred.reshape(-1),
        }
    )
    prediction_path = pinn_result_dir / 'prediction_table.csv'
    prediction_df.to_csv(prediction_path, index=False, encoding='utf-8-sig')

    metrics_path = pinn_result_dir / 'metrics.json'
    serializable_metrics = {key: float(value) for key, value in metrics.items()}
    metrics_path.write_text(json.dumps(serializable_metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    history_path = pinn_result_dir / 'training_history.csv'
    pd.DataFrame({key: pd.Series(values) for key, values in history.items()}).to_csv(
        history_path,
        index=False,
        encoding='utf-8-sig',
    )

    config_path = pinn_result_dir / 'run_config.json'
    config_path.write_text(
        json.dumps(
            {
                'property_name': property_name,
                'model_type': 'PINN',
                'data_sources': _project_data_sources(),
                'preproc_mode': preproc_mode,
                'sg_order': sg_order,
                'sg_window': sg_window,
                'fs_method': resolved_fs_method,
                'fs_param': resolved_fs_param,
                'input_mode': input_mode,
                'dataset_metadata': dataset_metadata,
                'pinn_config': pinn_config,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )

    result = {
        'property': property_name,
        'model_type': 'PINN',
        'training_history': history,
        'test_metrics': metrics,
        'prediction_table_path': str(prediction_path),
        'metrics_path': str(metrics_path),
        'training_history_path': str(history_path),
        'run_config_path': str(config_path),
        'result_dir': str(pinn_result_dir),
        'kinetics_plots': kinetics_plots,
        'prediction_scatter_path': scatter_plot_path,
        'preprocessing': {
            'mode': preproc_mode,
            'sg_order': sg_order,
            'sg_window': sg_window,
        },
        'dataset_metadata': dataset_metadata,
    }

    print(f"PINN预测完成 - R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    return result
