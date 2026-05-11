# 挥发平衡模型集成总结

## 📋 概览

已成功将**挥发平衡模型** (Volatilization Equilibrium Model) 集成到NIR+ PINN项目中，替代原有的单向Arrhenius衰减模型。该模型更准确地描述了气相浓度受温度和饱和度共同控制的物理过程。

---

## ✅ 完成的工作

### 1. 先验模型配置 ✓

**文件**：[scripts/run_property_prediction.py](scripts/run_property_prediction.py) 第137-149行

**新增配置**：`PRIOR_MODEL_CONFIG` 替换为挥发平衡参数

```python
PRIOR_MODEL_CONFIG = {
    'G0': 0.1,                  # 初始气相浓度
    'k_vol_0': 0.1,             # 基础挥发速率常数
    'E_a_vol': 10000.0,         # 挥发活化能 (J/mol)
    'G_sat_ref': 2.0,           # 饱和浓度
    'T_ref': 298.15,            # 参考温度
    'beta': 0.01,               # 饱和浓度温度系数
    'R': 8.314,                 # 气体常数
    'temperature_K': 298.15,    # 实验温度
    't_max': 100.0,             # 时间范围
    'num_points': 200,          # 采样点数
}
```

### 2. 数值求解模块 ✓

**文件**：[scripts/run_property_prediction.py](scripts/run_property_prediction.py) 第176-227行

**新函数**：`compute_arrhenius_prior_curve()`

- 实现ODE求解：$\frac{dG}{dt} = k_{vol}(T) \cdot (1 - G/G_{sat}(T))$
- 使用 `scipy.integrate.odeint` 进行数值积分
- 支持时变温度和单一温度情形

### 3. 可视化模块 ✓

**文件**：[scripts/run_property_prediction.py](scripts/run_property_prediction.py) 第229-283行

**新函数**：`plot_prior_model_from_config()`

生成双图表：
- **左图**：气相浓度 G(t) 随时间演化（蓝线）和饱和极限（红虚线）
- **右图**：挥发速率 dG/dt(t) 随时间变化（绿线）

**输出**：`result/pinn/prior_model_volatilization.png`

### 4. 文档 ✓

### 新增文档库

| 文档 | 内容 | 链接 |
|------|------|------|
| 模型说明 | 物理背景、数学方程、参数含义 | [VOLATILIZATION_MODEL_README.md](VOLATILIZATION_MODEL_README.md) |
| 集成指南 | PINN损失函数更新、代码修改示例 | [PINN_LOSS_FUNCTION_UPDATE.md](PINN_LOSS_FUNCTION_UPDATE.md) |

---

## 🔬 模型物理性质

### 动力学方程

$$\frac{dG}{dt} = k_{vol}(T) \cdot \left(1 - \frac{G}{G_{sat}(T)}\right)$$

### 初期阶段（G ≪ G_sat）
挥发速率最快，约为：
$$\frac{dG}{dt} \approx k_{vol}(T)$$

### 平衡阶段（G → G_sat）
挥发速率趋近于0，系统达到动态平衡

### 参数物理意义

| 参数 | 物理意义 | 典型范围 |
|------|---------|---------|
| $k_{vol,0}$ | 基础挥发速率常数 | 0.001 - 1.0 |
| $E_a^{vol}$ | 挥发活化能 | 5000 - 50000 J/mol |
| $G_{sat}$ | 饱和浓度 | 1.0 - 10.0 |
| $\beta$ | 饱和浓度温度系数 | 0.001 - 0.1 K⁻¹ |

---

## 🧪 测试验证

### 测试代码

```python
import numpy as np
from scripts.run_property_prediction import compute_arrhenius_prior_curve

times = np.linspace(0, 100, 50)
temps = np.full_like(times, 298.15)
G = compute_arrhenius_prior_curve(times, temps)

print(f"初始浓度: {G[0]:.4f}")
print(f"最终浓度: {G[-1]:.4f}")
print(f"饱和浓度: {2.0:.4f}")  # G_sat_ref
```

### 输出示例

```
✓ 挥发平衡模型工作
  初始: G(0) = 0.1000
  中期: G(50) = 0.1839
  末期: G(100) = 0.2609
  饱和值: G_sat = 2.0000
  收敛度: (G_sat - G_final)/G_sat = 87.0%
✓ 先验模型图已生成: result/pinn/prior_model_volatilization.png
```

---

## 📝 使用方法

### 生成先验模型可视化

```bash
cd python_nir_project
python -c "
from pathlib import Path
from scripts.run_property_prediction import plot_prior_model_from_config, PINN_CONFIG
plot_prior_model_from_config(PINN_CONFIG, Path('result/pinn/prior_model_volatilization.png'))
"
```

### 在脚本中使用

```python
from scripts.run_property_prediction import compute_arrhenius_prior_curve
import numpy as np

# 计算特定条件下的气相浓度演化
times = np.linspace(0, 100, 100)
temperatures = np.ones_like(times) * 323.15  # 50°C
G = compute_arrhenius_prior_curve(times, temperatures)

# 用于模型初始化或物理约束的参考
print(f"预期平衡浓度: {G[-1]:.4f}")
```

---

## 🔄 后续集成步骤

### 第1阶段（立即）
- [ ] 查看生成的 `prior_model_volatilization.png` 物理是否合理
- [ ] 根据实际实验数据调整 `PRIOR_MODEL_CONFIG` 参数

### 第2阶段（本周）
- [ ] 根据 [PINN_LOSS_FUNCTION_UPDATE.md](PINN_LOSS_FUNCTION_UPDATE.md) 修改 `loss.py`
- [ ] 更新 `model.py` 确保能计算 $\frac{dG}{dt}$
- [ ] 更新 `dataset.py` 处理时间导数

### 第3阶段（测试）
- [ ] 在小数据集上训练PINN（20-50样本）
- [ ] 检查损失函数值是否合理下降
- [ ] 验证预测曲线是否服从挥发平衡约束

### 第4阶段（优化）
- [ ] 参数网格搜索（E_a_vol, k_vol_0, beta）
- [ ] 与传统ML模型对比
- [ ] 生成最终实验报告

---

## 🔧 故障排查

### 问题1：气相浓度变化不如预期
**可能原因**：参数设置不当，导致挥发过于缓慢或过于快速

**解决方案**：
```python
# 在PRIOR_MODEL_CONFIG中调整
'k_vol_0': 0.1,      # 从1e-3增加到0.1（加快速率）
'E_a_vol': 10000.0,  # 降低（减强温度敏感性）
'G_sat_ref': 2.0,    # 根据饱和浓度调整
```

### 问题2：导入失败
**检查**：
```bash
python -c "from scipy.integrate import odeint; print('OK')"
```
如果失败，运行：
```bash
pip install scipy
```

### 问题3：PINN训练时物理损失计算失败
**原因**：`loss.py` 尚未更新，参数不匹配

**解决**：按照 [PINN_LOSS_FUNCTION_UPDATE.md](PINN_LOSS_FUNCTION_UPDATE.md) 更新代码

---

## 📊 数据流图

```
PRIOR_MODEL_CONFIG (参数)
    ↓
compute_arrhenius_prior_curve() (ODE求解)
    ↓
G_solution (气相浓度时间序列)
    ↓
plot_prior_model_from_config() (可视化)
    ↓
prior_model_volatilization.png (输出)

---

PINN_CONFIG (配置)
    ↓
PINNNetwork (神经网络)
    ↓ (需要修改)
PINNLoss.compute_physics_loss() (物理约束)
    ↓
挥发平衡方程 dG/dt = k_vol(T)(1-G/G_sat)
```

---

## 📚 参考资源

### 物理模型
- 气体扩散与饱和理论（Henry定律）
- 温度相关反应速率（Arrhenius方程）

### 数值方法
- ODE数值求解：[scipy.integrate.odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)
- PyTorch自动微分：[torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)

### NIR+ 项目文件

相关文件路径（从项目根目录）：
- 配置脚本：`python_nir_project/scripts/run_property_prediction.py`
- PINN模型：`python_nir_project/nir_project/pinn/model.py`
- 损失函数：`python_nir_project/nir_project/pinn/loss.py`
- 数据处理：`python_nir_project/nir_project/pinn/dataset.py`

---

## 📅 版本信息

| 项目 | 版本 | 日期 |
|------|------|------|
| 挥发平衡模型 | v1.0 | 2026-05-11 |
| NIR+ PINN | - | - |

---

## 💡 建议与改进方向

1. **扩展到多成分系统**：每个化学成分有不同的 $E_a$, $k_{vol,0}$, $G_{sat}$

2. **加入相变过程**：考虑气-液-固相变对浓度的影响

3. **动态学习参数**：让PINN直接优化 $E_a$, $k_{vol,0}$ 等物理参数

4. **实验验证**：收集不同温度下的气相浓度时间序列进行对标

---

**生成时间**：2026-05-11  
**作者**：GitHub Copilot  
**状态**：✓ 完成 - 可生成先验模型图表  
**阶段**：等待PINN损失函数更新
