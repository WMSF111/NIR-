"""
🎯 PINN模块使用指南

Physics-Informed Neural Network (PINN) 用于NIR光谱质量预测
结合深度学习和Arrhenius动力学约束
"""

## 📋 模块结构

```
nir_project/pinn/
├── __init__.py          # 模块导出
├── dataset.py           # 数据预处理与差值增强
├── model.py             # PINN网络架构
├── loss.py              # 物理损失函数
├── train.py             # 训练策略
└── evaluate.py          # 评估与可视化
```

## 🔧 快速开始

### 1. 数据准备

```python
from nir_project.pinn import prepare_pinn_dataset

# 准备数据集（自动处理预处理和Mixup增强）
dataset = prepare_pinn_dataset(
    X_nir=nir_spectra,           # [n_samples, n_wavelengths]
    X_enose=enose_data,          # [n_samples, n_sensors, n_timesteps]
    y=quality_labels,             # [n_samples]
    times=sampling_times,         # [n_samples]
    temperatures=temperature_data, # [n_samples]
    train_ratio=0.7,
    augment=True,                 # 启用Mixup增强
    mixup_alpha=0.2,
    mixup_samples=100,
)

# 访问数据
train_data = dataset['train']      # 训练集
val_data = dataset['val']          # 验证集
coll_data = dataset['collocation'] # 配点集（用于物理损失）
```

### 2. 构建模型

```python
from nir_project.pinn import PINNNetwork

# 创建多输入融合网络
model = PINNNetwork(
    nir_dim=50,                    # NIR特征维度
    nose_dim=10,                   # E-nose特征维度
    nir_hidden=(128, 64),          # NIR分支隐层
    nose_hidden=(32, 16),          # E-nose分支隐层
    shared_hidden=(128, 64, 32),   # 融合层
    physics_dim=2,                 # 时间+温度
    output_dim=1,                  # 单一输出（质量值）
    activation='relu',
    dropout=0.1,
)
```

### 3. 定义物理损失

```python
from nir_project.pinn import PINNLoss

loss_fn = PINNLoss(
    E_a=100000.0,        # 激活能 (J/mol)
    k0=1e-3,             # 频率因子
    R=8.314,             # 气体常数
    w_data=1.0,          # 数据损失权重
    w_physics=0.001,     # 物理损失权重
    kinetics_order=1,    # 一阶反应
)
```

### 4. 两阶段训练

```python
from nir_project.pinn import train_pinn_two_stage

history = train_pinn_two_stage(
    model=model,
    loss_fn=loss_fn,
    train_data=dataset['train'],
    val_data=dataset['val'],
    collocation_data=dataset['collocation'],
    stage1_epochs=5000,  # Adam阶段
    stage2_epochs=2000,  # L-BFGS阶段
    stage1_lr=0.001,
    stage2_lr=1.0,
    device='cuda',
    verbose=True,
)
```

### 5. 模型评估

```python
from nir_project.pinn import evaluate_pinn, plot_comparison, plot_training_history

# 计算指标
metrics = evaluate_pinn(
    model=model,
    X_nir=val_data['X_nir'],
    X_enose=val_data['X_enose'],
    times=val_data['times'],
    temperatures=val_data['temperatures'],
    y_true=val_data['y'].numpy(),
    device='cuda',
)

print(f"R²: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"RPD: {metrics['rpd']:.2f}")

# 可视化
plot_comparison(
    models={'PINN': model},
    X_nir=val_data['X_nir'],
    X_enose=val_data['X_enose'],
    times=val_data['times'],
    temperatures=val_data['temperatures'],
    y_true=val_data['y'].numpy(),
    save_path='prediction_comparison.png'
)

plot_training_history(
    history=history,
    save_path='training_history.png'
)
```

## 📊 关键概念

### 物理损失函数

PINN的核心是物理约束。采用Arrhenius动力学模型：

1. **速率常数**: k = k₀ × exp(-Eₐ / (R·T))
2. **一阶反应**: dC/dt = -k·C
3. **物理残差**: f = dC/dt + k·C
4. **物理损失**: L_physics = Σ(f²)

### 差值增强 (Delta Augmentation)

通过构造变化量样本帮助模型学习动力学趋势：
- ΔX = X₂ - X₁
- ΔY = Y₂ - Y₁
- 目的：强化对"变化过程"的学习

### Mixup增强

在特征空间生成中间样本：
- X_new = λ·X₁ + (1-λ)·X₂
- Y_new = λ·Y₁ + (1-λ)·Y₂
- λ ~ Beta(α, α)

## 🎓 完整工作流程

```python
import torch
import numpy as np
from pathlib import Path
from nir_project.pinn import *

# 1. 加载数据
X_nir = np.load('nir_features.npy')
X_enose = np.load('enose_data.npy')
y = np.load('quality_labels.npy')
times = np.load('sampling_times.npy')
temperatures = np.load('temperatures.npy')

# 2. 准备数据集
dataset = prepare_pinn_dataset(
    X_nir, X_enose, y, times, 
    temperatures + 273.15,  # 转为Kelvin
    augment=True
)

# 3. 建立模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PINNNetwork(50, 10)

# 4. 定义损失
loss_fn = PINNLoss(
    E_a=100000,    # 根据实际物质调整
    k0=1e-3,
    w_data=1.0,
    w_physics=0.001,
)

# 5. 两阶段训练
history = train_pinn_two_stage(
    model, loss_fn,
    dataset['train'], dataset['val'],
    dataset['collocation'],
    stage1_epochs=5000,
    stage2_epochs=2000,
    device=device,
)

# 6. 对比评估
models = {
    'PINN (with physics)': model,
}

results = compare_models(
    models, dataset['val']['X_nir'],
    dataset['val']['X_enose'],
    dataset['val']['times'],
    dataset['val']['temperatures'],
    dataset['val']['y'].numpy(),
)

print(generate_evaluation_report(results))

# 7. 可视化
plot_comparison(models, ...)
plot_training_history(history)
```

## 📈 性能指标

| 指标 | 说明 | 目标 |
|-----|-----|------|
| R² | 决定系数 | > 0.9 |
| RMSE | 均方根误差 | 越小越好 |
| RPD | 相对分析误差 | > 3 优秀，> 2 良好 |
| MAE | 平均绝对误差 | 越小越好 |

## 🔍 参数调整建议

### 网络架构
```python
# 对于小数据集
nir_hidden=(64, 32)
nose_hidden=(16, 8)
shared_hidden=(64, 32)
dropout=0.2

# 对于大数据集
nir_hidden=(256, 128, 64)
nose_hidden=(64, 32)
shared_hidden=(256, 128, 64, 32)
dropout=0.1
```

### 损失权重
```python
# 初始阶段 (强化数据拟合)
w_data=1.0, w_physics=0.001

# 中期 (逐步加入物理约束)
w_data=1.0, w_physics=0.01

# 后期 (平衡双约束)
w_data=0.5, w_physics=0.05
```

### 动力学参数
根据实际物质的性质调整：
- **Eₐ (激活能)**: 通常 50-500 kJ/mol (50000-500000 J/mol)
- **k₀ (频率因子)**: 通常 1e-5 到 1e2
- **n (反应阶数)**: 1（一阶）或 2（二阶）

## 🚀 高级特性

### 1. 动态权重调整

```python
def physics_schedule(epoch, total_epochs):
    # 线性增加物理权重
    return 0.001 * (1 + 99 * epoch / total_epochs)

train_pinn_two_stage(
    ...,
    w_physics_schedule=physics_schedule,
)
```

### 2. 网络预测动力学参数

```python
from nir_project.pinn import PINNNetworkWithKinetics

model = PINNNetworkWithKinetics(
    nir_dim=50, nose_dim=10,
    predict_kinetics=True,  # 让网络学习E_a和k0
)

# 输出: (C_pred, E_a, k0)
C_pred, E_a, k0 = model(X_nir, X_enose, t, T)
```

### 3. 物理残差可视化

```python
# 计算物理残差（检查是否满足动力学方程）
dC_dt = model.get_gradient_wrt_time(X_nir, X_enose, times, temps)
k = arrhenius_rate_constant(temps, E_a, k0)
residual = dC_dt + k * C_pred
residual_norm = (residual ** 2).mean()
```

## ⚠️ 常见问题

### 1. 物理损失不下降？
- 检查Eₐ和k₀是否合理（是否与实际物质匹配）
- 尝试降低w_physics的初始值
- 确保时间和温度的量纲正确

### 2. 模型过拟合？
- 增加dropout比例
- 使用更多的Mixup增强（mixup_samples）
- 减少隐层大小

### 3. 收敛缓慢？
- 尝试更大的学习率
- 增加第一阶段的epochs
- 检查数据预处理是否正确（是否归一化）

## 📚 参考文献

- Raissi et al. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs"
- Han et al. "Solving many-electron Schrödinger equation using deep neural networks"

## 📝 许可证

同项目许可
"""

