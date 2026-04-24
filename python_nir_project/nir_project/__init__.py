"""
NIR (近红外光谱) 质量预测项目核心模块

本项目提供完整的NIR光谱分析管道，包括：
- 数据加载与预处理
- 特征选择与降维
- 传统机器学习与深度学习模型
- 物理信息神经网络 (PINN)
- 完整的预测管道

主要功能：
1. 数据处理：加载NIR光谱、E-nose传感器数据、黑白校正
2. 预处理：SNV、MSC、Savitzky-Golay滤波、基线校正
3. 特征选择：相关性、PCA、SPA、CARS、RFE等方法
4. 建模：PLS、PCR、SVR、随机森林、KNN、GPR、CNN、XGBoost
5. PINN：物理约束的神经网络，支持Arrhenius动力学
6. 评估：R²、RMSE、MAE、RPD指标，可视化分析

作者：AI Assistant
版本：2.0 (包含PINN模块)
"""

from .data import (
    build_property_vector_from_all_csv,
    black_white_processing,
    load_csv_spectrum,
    save_dataset,
    load_dataset,
)
from .preprocessing import preprocess_spectrum, preprocess_pair
from .feature_selection import select_features_by_method
from .modeling import train_model_from_dataset
from .pipeline import compare_property_prediction_pipeline, run_property_prediction

# PINN模块导入
try:
    from . import pinn
    from .pinn import (
        PINNNetwork,
        PINNLoss,
        prepare_pinn_dataset,
        train_pinn,
        train_pinn_two_stage,
        evaluate_pinn,
        plot_comparison,
        plot_kinetics,
    )
    HAS_PINN = True
except ImportError:
    HAS_PINN = False

__all__ = [
    'build_property_vector_from_all_csv',
    'black_white_processing',
    'load_csv_spectrum',
    'save_dataset',
    'load_dataset',
    'preprocess_spectrum',
    'preprocess_pair',
    'select_features_by_method',
    'train_model_from_dataset',
    'compare_property_prediction_pipeline',
    'run_property_prediction',
    'pinn',
    'PINNNetwork',
    'PINNLoss',
    'prepare_pinn_dataset',
    'train_pinn',
    'train_pinn_two_stage',
    'evaluate_pinn',
    'plot_comparison',
    'plot_kinetics',
]
