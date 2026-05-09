"""项目命令行入口。"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from .pipeline import batch_compare_pinn_prediction, run_pinn_prediction, run_property_prediction


def run_prediction(
    property_name: str = 'a*',
    model_type: str = 'traditional',
    mode: str = 'train',
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'corr_topk',
    fs_param: Optional[int] = None,
    include_preprocessed_group: bool = False,
    traditional_config: Optional[Dict[str, Any]] = None,
    pinn_config: Optional[Dict[str, Any]] = None,
):
    """根据模型类型运行传统模型或 PINN 流程。"""
    resolved_mode = mode.lower().strip()
    if model_type == 'pinn':
        resolved_fs_method = fs_method
        if (resolved_fs_method or '').strip().lower() == 'all':
            resolved_fs_method = 'all' if resolved_mode == 'compare' else 'corr_topk'
        if resolved_mode == 'compare':
            result = batch_compare_pinn_prediction(
                property_names=[item.strip() for item in property_name.split(',') if item.strip()],
                preproc_mode=preproc_mode,
                sg_order=sg_order,
                sg_window=sg_window,
                feature_selection_method=resolved_fs_method,
                fs_param=fs_param,
                pinn_config=pinn_config,
            )
        else:
            result = run_pinn_prediction(
                property_name,
                preproc_mode,
                sg_order,
                sg_window,
                fs_method=resolved_fs_method,
                fs_param=fs_param,
                pinn_config=pinn_config,
            )
    else:
        result = run_property_prediction(
            property_name=property_name,
            mode=resolved_mode,
            preproc_mode=preproc_mode,
            sg_order=sg_order,
            sg_window=sg_window,
            fs_method=fs_method,
            fs_param=fs_param,
            include_preprocessed_group=include_preprocessed_group,
            traditional_config=traditional_config,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description='运行 NIR 属性预测管道。')
    parser.add_argument(
        '--property_name',
        default='a*',
        help='目标属性名称，例如 a* 或 b*；默认使用 a*',
    )
    parser.add_argument(
        '--model_type',
        default='pinn',
        choices=['traditional', 'pinn'],
        help='模型类型：traditional 为传统机器学习，pinn 为物理信息神经网络',
    )
    parser.add_argument(
        '--mode',
        default='compare',
        choices=['train', 'compare', 'select'],
        help='运行模式：train 单次训练，compare 批量对比，select 仅做特征选择并导出数据集',
    )
    parser.add_argument(
        '--preproc_mode',
        default='sg+msc+snv',
        help='预处理模式：sg+msc / sg+snv / sg+msc+snv / sg / none',
    )
    parser.add_argument(
        '--sg_order',
        type=int,
        default=3,
        help='Savitzky-Golay 滤波的多项式阶数',
    )
    parser.add_argument(
        '--sg_window',
        type=int,
        default=5,
        help='Savitzky-Golay 滤波窗口长度',
    )
    parser.add_argument(
        '--fs_method',
        default='all',
        help='特征选择方法，可填 spa / pca / cars / corr_topk / all',
    )
    parser.add_argument(
        '--fs_param',
        type=int,
        default=None,
        help='特征选择方法对应的参数，例如保留维度或特征数',
    )
    parser.add_argument(
        '--include_preprocessed_group',
        action='store_true',
        help='在 compare 模式下，额外包含“仅预处理不选特征”的模型组',
    )
    return parser


def main() -> None:
    """命令行主入口。"""
    args = build_parser().parse_args()
    result = run_prediction(
        property_name=args.property_name,
        model_type=args.model_type,
        mode=args.mode,
        preproc_mode=args.preproc_mode,
        sg_order=args.sg_order,
        sg_window=args.sg_window,
        fs_method=args.fs_method,
        fs_param=args.fs_param,
        include_preprocessed_group=args.include_preprocessed_group,
    )
