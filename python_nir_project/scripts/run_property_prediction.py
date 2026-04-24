from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nir_project.pipeline import run_property_prediction, run_pinn_prediction


def run_prediction(
    property_name: str,
    model_type: str = 'traditional',
    method_name: str = 'spa',
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: str = 'corr_topk',
    fs_param: Optional[int] = None,
):
    """
    直接在Python中运行NIR属性预测

    Args:
        property_name: 目标属性名称，如"a*"或"b*"
        model_type: 模型类型，'traditional'或'pinn'
        method_name: 传统模型方法名，仅当model_type='traditional'时使用
        preproc_mode: 预处理模式
        sg_order: Savitzky-Golay多项式阶数
        sg_window: Savitzky-Golay窗口长度
        fs_method: 特征选择方法，仅当method_name='select'或'fs'时使用
        fs_param: 特征选择参数值

    Returns:
        预测结果
    """
    print(f"开始运行NIR属性预测 - 属性: {property_name}, 模型类型: {model_type}")

    if model_type == 'pinn':
        result = run_pinn_prediction(
            property_name,
            preproc_mode,
            sg_order,
            sg_window,
        )
    else:
        result = run_property_prediction(
            property_name,
            method_name,
            preproc_mode,
            sg_order,
            sg_window,
            fs_method,
            fs_param,
        )

    print("预测完成!")
    return result


def main() -> None:
    """
    主函数：自动检测是命令行调用还是Python调用
    """
    # 检查是否有命令行参数（除了脚本名）
    if len(sys.argv) > 1:
        # 命令行模式
        parser = argparse.ArgumentParser(description='Run NIR property prediction pipeline')
        parser.add_argument('--property_name', required=True, help='Target property name such as a* or b*')
        parser.add_argument('--model_type', default='traditional', choices=['traditional', 'pinn'],
                           help='Model type: traditional (machine learning) or pinn (physics-informed neural network)')
        parser.add_argument('--method_name', default='spa', help='Run mode for traditional models: spa / cars / pca / corr_topk / select / fs')
        parser.add_argument('--preproc_mode', default='sg+msc', help='Preprocessing mode: sg+msc / sg+snv / sg+msc+snv / sg / none')
        parser.add_argument('--sg_order', type=int, default=3, help='Savitzky-Golay polynomial order')
        parser.add_argument('--sg_window', type=int, default=15, help='Savitzky-Golay window length')
        parser.add_argument('--fs_method', default='corr_topk', help='Feature selection method for select/fs mode')
        parser.add_argument('--fs_param', type=int, default=None, help='Feature selection parameter value')
        args = parser.parse_args()

        result = run_prediction(
            property_name=args.property_name,
            model_type=args.model_type,
            method_name=args.method_name,
            preproc_mode=args.preproc_mode,
            sg_order=args.sg_order,
            sg_window=args.sg_window,
            fs_method=args.fs_method,
            fs_param=args.fs_param,
        )
        print(result)
    else:
        # Python调用模式 - 显示使用说明
        print("=" * 60)
        print("NIR属性预测脚本使用说明")
        print("=" * 60)
        print()
        print("命令行使用:")
        print("python run_property_prediction.py --property_name a* --model_type traditional")
        print()
        print("Python中直接调用:")
        print("from scripts.run_property_prediction import run_prediction")
        print("result = run_prediction('a*', 'traditional', 'spa')")
        print()
        print("可用参数:")
        print("- property_name: 目标属性名 (必需)")
        print("- model_type: 'traditional' 或 'pinn'")
        print("- method_name: 传统方法 (spa/cars/pca等)")
        print("- preproc_mode: 预处理模式")
        print("- sg_order: SG多项式阶数")
        print("- sg_window: SG窗口长度")
        print()


if __name__ == '__main__':
    main()
