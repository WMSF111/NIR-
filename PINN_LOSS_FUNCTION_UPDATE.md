# PINN 损失函数更新指南

## 概述

当前PINN物理损失函数基于**一阶Arrhenius衰减动力学**。为与新的**挥发平衡模型**对齐，需要更新 [loss.py](nir_project/pinn/loss.py) 中的物理损失计算。

## 新旧模型对比

### 原模型（Arrhenius衰减）
$$C(t) = C_0 \cdot \exp(-k \cdot t), \quad k = k_0 \cdot \exp\left(-\frac{E_a}{R \cdot T}\right)$$

物理约束：固定衰减率，不考虑气相饱和效应

### 新模型（挥发平衡）
$$\frac{dG}{dt} = k_{vol}(T) \cdot \left(1 - \frac{G}{G_{sat}(T)}\right)$$

物理约束：动态挥发速率，接近饱和时自动趋零

## PINN损失函数修改方案

### 第1步：更新配置参数

在 `run_property_prediction.py` 的 `PINN_CONFIG['loss_params']` 中替换为：

```python
'loss_params': {
    # 旧参数（删除）
    # 'E_a': 100000.0,
    # 'k0': 1e-3,
    
    # 新参数（添加）
    'E_a_vol': 10000.0,         # 挥发活化能
    'k_vol_0': 0.1,             # 基础挥发速率常数
    'beta': 0.01,               # 饱和浓度温度系数
    'G_sat_ref': 2.0,           # 参考温度饱和浓度
    'T_ref': 298.15,            # 参考温度
    
    # 权重参数（保留）
    'R': 8.314,
    'w_data': 1.0,
    'w_physics': 0.001,
},
```

### 第2步：修改 `loss.py` 中的 `compute_physics_loss()` 方法

**位置**：[nir_project/pinn/loss.py](nir_project/pinn/loss.py) 的 `PINNLoss.compute_physics_loss()` 方法（约130-160行）

**修改内容**：

```python
def compute_physics_loss(self, G, T, dGdt):
    """计算物理约束损失：挥发平衡模型
    
    物理方程：dG/dt = k_vol(T) * (1 - G/G_sat(T))
    损失函数：L_physics = MSE(dG/dt_pred - dG/dt_theory)
    
    Args:
        G: 气相浓度预测值 [batch_size, 1]
        T: 温度值 [batch_size, 1]
        dGdt: 网络预测的浓度变化率 [batch_size, 1]
    
    Returns:
        物理损失标量
    """
    # 提取参数
    E_a_vol = self.loss_config.get('E_a_vol', 10000.0)
    k_vol_0 = self.loss_config.get('k_vol_0', 0.1)
    beta = self.loss_config.get('beta', 0.01)
    G_sat_ref = self.loss_config.get('G_sat_ref', 2.0)
    T_ref = self.loss_config.get('T_ref', 298.15)
    R = self.loss_config.get('R', 8.314)
    
    # 计算温度相关的挥发速率常数
    # k_vol(T) = k_vol_0 * exp(-E_a_vol / (R*T))
    k_vol = k_vol_0 * torch.exp(-E_a_vol / (R * (T + 1e-12)))
    
    # 计算饱和浓度
    # G_sat(T) = G_sat_ref * exp(beta * (T - T_ref))
    G_sat = G_sat_ref * torch.exp(beta * (T - T_ref))
    
    # 理论挥发速率（右侧项）
    # dG/dt_theory = k_vol(T) * (1 - G/G_sat(T))
    dGdt_theory = k_vol * (1.0 - G / (G_sat + 1e-12))
    
    # 物理损失：预测速率与理论速率的偏差
    physics_loss = torch.mean((dGdt - dGdt_theory) ** 2)
    
    # 可选：添加约束条件（气相浓度不超过饱和值）
    # penalty = torch.mean(torch.clamp(G - G_sat, min=0) ** 2)
    # physics_loss = physics_loss + 0.1 * penalty
    
    return physics_loss
```

### 第3步：在 `model.py` 中添加速率计算

确保神经网络能够计算浓度的时间导数 $\frac{dG}{dt}$

**位置**：[nir_project/pinn/model.py](nir_project/pinn/model.py) 的 `PINNNetwork` 类

**需要的方法**：

```python
def compute_dGdt(self, G, times, requires_grad=True):
    """计算气相浓度对时间的导数 dG/dt
    
    使用PyTorch自动微分实现
    """
    if not times.requires_grad:
        times = times.clone().detach().requires_grad_(True)
    
    G_pred = self.forward(...)  # 网络前向传播
    
    # 创建计算图并求导
    dGdt = torch.autograd.grad(
        outputs=G_pred,
        inputs=times,
        grad_outputs=torch.ones_like(G_pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return dGdt
```

## 数据准备修改

### 在 `dataset.py` 中的考虑

在 [nir_project/pinn/dataset.py](nir_project/pinn/dataset.py) 的数据预处理中，需要确保：

1. **时间间隔的一致性**
   - 确保相邻样品间的时间差均匀（或在`_build_sliding_forecast_windows()`中处理）
   - 用于计算数值导数 $\frac{dG}{dt} \approx \frac{\Delta G}{\Delta t}$

2. **气相饱和度的标准化**
   - 如果缩放了NIR信号（如SNV、MSC），气相浓度$G$也应按相同方式处理
   - 或者在计算物理损失前反标准化

示例代码（在dataset.py中）：

```python
def prepare_time_derivatives(self, G_values, times):
    """从浓度时间序列计算导数"""
    dGdt = np.gradient(G_values, times)  # 数值导数
    return dGdt

# 在创建training batch时
batch_G = ...
batch_times = ...
batch_dGdt = prepare_time_derivatives(batch_G, batch_times)  # 数值导数
```

## 训练策略调整

### 两阶段训练

保持现有的两阶段训练但调整权重：

**阶段1（Adam优化，0-67%）**：
- 数据损失权重高：`w_data = 1.0`
- 物理约束权重较低：`w_physics = 0.001`
- 让网络先拟合观测数据

**阶段2（L-BFGS精化，67-100%）**：
- 逐步增加物理权重：`w_physics = 0.01` 或更高
- 迫使网络遵守物理约束

```python
# 在train.py中修改
if epoch / total_epochs < stage1_ratio:
    w_physics = 0.001  # 阶段1
else:
    w_physics = 0.01 * (epoch / total_epochs - stage1_ratio) / (1 - stage1_ratio)  # 线性增长
```

## 验证检查清单

完成修改后，检查以下项目：

- [ ] `PRIOR_MODEL_CONFIG` 参数已更新
- [ ] `PINN_CONFIG['loss_params']` 包含新参数（E_a_vol、k_vol_0等）
- [ ] `loss.py` 中 `compute_physics_loss()` 实现了挥发平衡方程
- [ ] `model.py` 中能够计算 $\frac{dG}{dt}$（自动微分或数值导数）
- [ ] `dataset.py` 中处理了时间导数计算
- [ ] 运行 `python scripts/run_property_prediction.py` 生成 `prior_model_volatilization.png`
- [ ] 训练一个小的test batch验证损失计算正常

## 参数调优指南

根据实验数据调整以下参数：

| 参数 | 影响 | 调整建议 |
|------|------|--------|
| `k_vol_0` | 初期挥发速率 | 数据不增长→增大；增长过快→减小 |
| `E_a_vol` | 温度敏感性 | 温度变化影响小→增大；影响大→减小 |
| `beta` | 饱和浓度温度敏感性 | 通常保持较小（0.001-0.05） |
| `G_sat_ref` | 饱和平衡点 | 调整使网络预测不超过此值 |
| `w_physics` | 物理约束强度 | 过小→忽视物理；过大→无法拟合数据 |

## 后续步骤

1. **立即**：修改 `loss.py` 中的 `compute_physics_loss()` 方法
2. **验证**：在小数据集上训练并检查损失曲线
3. **微调**：根据训练效果调整 `E_a_vol`, `k_vol_0`, `G_sat_ref`
4. **比较**：与传统ML模型的结果对比
5. **文档**：记录最优参数配置到项目README
