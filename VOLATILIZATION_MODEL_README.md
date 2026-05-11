# 先验模型更新：挥发平衡模型

## 模型物理背景

替代之前的单向Arrhenius衰减模型，新的**挥发平衡模型** (Volatilization Equilibrium Model) 考虑了：

1. **温度的影响**：通过Arrhenius方程控制挥发速率
2. **气相饱和度的影响**：当气相浓度接近饱和值时，挥发速率趋近于0

### 数学方程

$$\frac{dG}{dt} = k_{vol}(T) \cdot \left(1 - \frac{G}{G_{sat}(T)}\right)$$

其中：

- **$G(t)$**：当前时刻的气相浓度（可测量值，如NIR信号）
- **$k_{vol}(T)$**：温度相关的基础挥发速率常数

$$k_{vol}(T) = k_{vol,0} \cdot \exp\left(-\frac{E_a^{vol}}{R \cdot T}\right)$$

- **$G_{sat}(T)$**：当前温度下的饱和浓度（Henry常数相关）

$$G_{sat}(T) = G_{sat,ref} \cdot \exp\left(\beta \cdot (T - T_{ref})\right)$$

- **$(1 - G/G_{sat})$**：修正因子，表示距离饱和所需的"驱动力"

## 物理意义

### 初期（$G \ll G_{sat}$）
挥发速率最快，约为：
$$\frac{dG}{dt} \approx k_{vol}(T)$$

### 中期（$G \sim 0.5 G_{sat}$）
挥发速率随气相浓度增加而逐渐降低

### 平衡期（$G \to G_{sat}$）
挥发速率趋近于0，系统达到动态平衡：
$$\frac{dG}{dt} \to 0$$

## 参数配置

在 `run_property_prediction.py` 中的 `PRIOR_MODEL_CONFIG`：

| 参数 | 物理含义 | 默认值 | 单位 |
|------|--------|-------|------|
| `G0` | 初始气相浓度 | 0.1 | a.u. |
| `k_vol_0` | 基础挥发速率常数 | 1e-3 | 1/time |
| `E_a_vol` | 挥发活化能 | 50000.0 | J/mol |
| `G_sat_ref` | 参考温度下的饱和浓度 | 10.0 | a.u. |
| `T_ref` | 参考温度 | 298.15 | K |
| `beta` | 饱和浓度温度系数 | 0.05 | 1/K |
| `R` | 气体常数 | 8.314 | J/(mol·K) |
| `temperature_K` | 实验温度 | 298.15 | K |

## 数值求解

使用 `scipy.integrate.odeint` 对微分方程进行数值求解：

```python
from scipy.integrate import odeint

# 求解ODE: dG/dt = k_vol(T) * (1 - G/G_sat(T))
G_solution = odeint(volatilization_ode, G0, times)
```

## 可视化

运行脚本时会自动生成 `result/pinn/prior_model.png`，包含两个子图：

1. **左图**：气相浓度随时间的演化曲线
   - 蓝色曲线：$G(t)$
   - 红色虚线：$G_{sat}(T)$表示饱和极限

2. **右图**：挥发速率随时间的变化
   - 绿色曲线：$\frac{dG}{dt}(t)$
   - 显示从最大速率向0的渐进过程

## 参数优化建议

### 若实验所有样品间相同温度
保持 `beta` 较小（0.01-0.1）

### 若有大的温度范围
调整 `E_a_vol` 以匹配温度对挥发速率的影响程度：
- 高活化能 → 温度敏感性强
- 低活化能 → 温度敏感性弱

### 若气相达到平衡较快
增加 `k_vol_0` 加快挥发速率

### 若气相饱和值不清楚
通过实验数据或文献值调整 `G_sat_ref` 和 `beta`

## 与PINN的集成

PINN的物理损失函数现在应计算：

$$L_{physics} = \text{MSE}\left(\frac{dG}{dt} - k_{vol}(T) \cdot \left(1 - \frac{G}{G_{sat}(T)}\right)\right)$$

使用自动微分 (PyTorch) 计算 $\frac{dG}{dt}$，约束PINN的输出遵循挥发平衡模型。
