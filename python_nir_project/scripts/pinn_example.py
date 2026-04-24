"""
PINN使用示例脚本
展示如何使用PINN模块进行NIR质量预测
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from nir_project.pinn import (
    PINNNetwork,
    PINNLoss,
    prepare_pinn_dataset,
    train_pinn_two_stage,
    evaluate_pinn,
    compare_models,
    plot_comparison,
    plot_training_history,
    generate_evaluation_report,
)


def example_basic_training():
    """
    基础训练示例
    """
    print("\n" + "="*60)
    print("PINN基础训练示例")
    print("="*60 + "\n")
    
    # 生成模拟数据
    n_samples = 100
    X_nir = np.random.randn(n_samples, 50).astype(np.float32)  # 50维NIR特征
    X_enose = np.random.randn(n_samples, 10).astype(np.float32)  # 10维E-nose特征
    times = np.linspace(0, 100, n_samples).astype(np.float32)  # 时间 (小时)
    temperatures = (20 + np.random.randn(n_samples) * 5).astype(np.float32)  # 温度 (摄氏度)
    
    # 根据Arrhenius动力学生成标签 (一阶衰减)
    C0 = 90.0  # 初始质量
    E_a = 50000.0  # 激活能 (J/mol)
    k0 = 1e-3  # 频率因子
    R = 8.314  # 气体常数
    T_kelvin = temperatures + 273.15
    
    k_vals = k0 * np.exp(-E_a / (R * T_kelvin))
    y = C0 * np.exp(-k_vals * times)  # 一阶衰减模型
    y = y.astype(np.float32)
    
    print(f"✓ 已生成模拟数据:")
    print(f"  - 样本数: {n_samples}")
    print(f"  - NIR特征维度: {X_nir.shape[1]}")
    print(f"  - E-nose特征维度: {X_enose.shape[1]}")
    print(f"  - 质量范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 准备数据集 (包括Mixup增强)
    print("\n准备数据集 (含Mixup增强)...")
    dataset = prepare_pinn_dataset(
        X_nir=X_nir,
        X_enose=X_enose,
        y=y,
        times=times,
        temperatures=temperatures + 273.15,  # 转为Kelvin
        train_ratio=0.7,
        augment=True,
        mixup_alpha=0.2,
        mixup_samples=50,
    )
    print(f"✓ 训练集大小: {dataset['train']['X_nir'].shape[0]}")
    print(f"✓ 验证集大小: {dataset['val']['X_nir'].shape[0]}")
    print(f"✓ 配点集大小: {dataset['collocation']['X_nir'].shape[0]} (含增强)")
    
    # 创建模型
    print("\n创建PINN网络...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PINNNetwork(
        nir_dim=50,
        nose_dim=10,
        nir_hidden=(128, 64),
        nose_hidden=(32, 16),
        shared_hidden=(128, 64, 32),
        physics_dim=2,
        output_dim=1,
        activation='relu',
        dropout=0.1,
    )
    print(f"✓ 模型创建成功 | 设备: {device}")
    print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    loss_fn = PINNLoss(
        E_a=E_a,
        k0=k0,
        R=R,
        w_data=1.0,
        w_physics=0.001,
        kinetics_order=1,
    )
    print("✓ 损失函数已创建")
    
    # 两阶段训练
    print("\n开始两阶段训练...\n")
    
    # 权重调度 (逐步增加物理约束的权重)
    def physics_weight_schedule(epoch, total_epochs):
        return 0.001 * (1 + 9 * (epoch / max(1, total_epochs)))  # 从0.001增加到0.01
    
    history = train_pinn_two_stage(
        model=model,
        loss_fn=loss_fn,
        train_data=dataset['train'],
        val_data=dataset['val'],
        collocation_data=dataset['collocation'],
        stage1_epochs=1000,
        stage2_epochs=500,
        stage1_lr=0.001,
        stage2_lr=1.0,
        device=device,
        verbose=True,
        w_physics_schedule=physics_weight_schedule,
    )
    
    # 评估模型
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    metrics_train = evaluate_pinn(
        model=model,
        X_nir=dataset['train']['X_nir'],
        X_enose=dataset['train']['X_enose'],
        times=dataset['train']['times'],
        temperatures=dataset['train']['temperatures'],
        y_true=dataset['train']['y'].numpy(),
        device=device,
    )
    
    metrics_val = evaluate_pinn(
        model=model,
        X_nir=dataset['val']['X_nir'],
        X_enose=dataset['val']['X_enose'],
        times=dataset['val']['times'],
        temperatures=dataset['val']['temperatures'],
        y_true=dataset['val']['y'].numpy(),
        device=device,
    )
    
    print(f"\n训练集指标:")
    print(f"  R²: {metrics_train['r2']:.4f}")
    print(f"  RMSE: {metrics_train['rmse']:.4f}")
    print(f"  RPD: {metrics_train['rpd']:.2f}")
    
    print(f"\n验证集指标:")
    print(f"  R²: {metrics_val['r2']:.4f}")
    print(f"  RMSE: {metrics_val['rmse']:.4f}")
    print(f"  RPD: {metrics_val['rpd']:.2f}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(history, save_path=None)
    
    # 绘制预测对比图
    print("绘制预测对比图...")
    plot_comparison(
        models={'PINN': model},
        X_nir=dataset['val']['X_nir'],
        X_enose=dataset['val']['X_enose'],
        times=dataset['val']['times'],
        temperatures=dataset['val']['temperatures'],
        y_true=dataset['val']['y'].numpy(),
        device=device,
    )
    
    return model, dataset, history


def example_compare_models():
    """
    模型对比示例
    对比: PINN vs 纯DNN
    """
    print("\n" + "="*60)
    print("模型对比示例: PINN vs DNN")
    print("="*60 + "\n")
    
    # 生成数据
    n_samples = 80
    X_nir = np.random.randn(n_samples, 50).astype(np.float32)
    X_enose = np.random.randn(n_samples, 10).astype(np.float32)
    times = np.linspace(0, 100, n_samples).astype(np.float32)
    temperatures = (20 + np.random.randn(n_samples) * 5).astype(np.float32)
    
    # 生成目标值
    C0 = 90.0
    E_a = 50000.0
    k0 = 1e-3
    R = 8.314
    T_kelvin = temperatures + 273.15
    k_vals = k0 * np.exp(-E_a / (R * T_kelvin))
    y = C0 * np.exp(-k_vals * times).astype(np.float32)
    
    # 准备数据
    dataset = prepare_pinn_dataset(
        X_nir=X_nir,
        X_enose=X_enose,
        y=y,
        times=times,
        temperatures=temperatures + 273.15,
        train_ratio=0.7,
        augment=True,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型1: PINN (有物理约束)
    print("训练PINN模型 (with physics)...")
    model_pinn = PINNNetwork(
        nir_dim=50, nose_dim=10,
        nir_hidden=(128, 64),
        nose_hidden=(32, 16),
        shared_hidden=(128, 64, 32),
    )
    
    loss_pinn = PINNLoss(E_a=E_a, k0=k0, w_data=1.0, w_physics=0.001)
    
    history_pinn = train_pinn_two_stage(
        model=model_pinn,
        loss_fn=loss_pinn,
        train_data=dataset['train'],
        val_data=dataset['val'],
        collocation_data=dataset['collocation'],
        stage1_epochs=500,
        stage2_epochs=200,
        device=device,
        verbose=False,
    )
    
    # 模型2: 纯DNN (无物理约束)
    print("训练DNN模型 (no physics)...")
    model_dnn = PINNNetwork(
        nir_dim=50, nose_dim=10,
        nir_hidden=(128, 64),
        nose_hidden=(32, 16),
        shared_hidden=(128, 64, 32),
    )
    
    loss_dnn = PINNLoss(E_a=E_a, k0=k0, w_data=1.0, w_physics=0.0)  # 无物理损失
    
    history_dnn = train_pinn_two_stage(
        model=model_dnn,
        loss_fn=loss_dnn,
        train_data=dataset['train'],
        val_data=dataset['val'],
        collocation_data=dataset['collocation'],
        stage1_epochs=500,
        stage2_epochs=200,
        device=device,
        verbose=False,
    )
    
    # 对比评估
    print("\n" + "="*60)
    results = compare_models(
        models={'PINN': model_pinn, 'DNN': model_dnn},
        X_nir=dataset['val']['X_nir'],
        X_enose=dataset['val']['X_enose'],
        times=dataset['val']['times'],
        temperatures=dataset['val']['temperatures'],
        y_true=dataset['val']['y'].numpy(),
        device=device,
    )
    
    print(generate_evaluation_report(results))


if __name__ == '__main__':
    # 运行示例
    example_basic_training()
    example_compare_models()
