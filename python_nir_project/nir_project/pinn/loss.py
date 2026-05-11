"""PINN 训练中的损失函数，包含数据损失与 Arrhenius 物理约束损失。

本模块提供PINN训练中常用的损失结构，包括监督数据损失、
物理约束损失以及可选的动力学参数光滑性正则项。
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable


def arrhenius_rate_constant(
    temperatures: torch.Tensor,
    E_a: float,
    k0: float,
    R: float = 8.314,
) -> torch.Tensor:
    """
    根据Arrhenius公式计算化学反应速率常数

    Arrhenius方程：k = k0 * exp(-E_a / (R * T))

    其中：
    - k: 速率常数
    - k0: 频率因子（前因子）
    - E_a: 激活能（J/mol）
    - R: 气体常数（8.314 J/(mol·K)）
    - T: 绝对温度（Kelvin）

    Args:
        temperatures (torch.Tensor): 温度数组，形状为[batch_size, 1]，单位为Kelvin
        E_a (float): 激活能，单位为J/mol
        k0 (float): 频率因子
        R (float): 气体常数，默认8.314 J/(mol·K)

    Returns:
        torch.Tensor: 速率常数数组，形状为[batch_size, 1]
    """
    # 避免数值溢出
    exponent = -E_a / (R * temperatures + 1e-10)
    exponent = torch.clamp(exponent, min=-100, max=20)

    k = k0 * torch.exp(exponent)
    return k


class PINNLoss(nn.Module):
    """
    PINN基础损失函数

    总损失 = w_data * L_data + w_physics * L_physics

    数据损失L_data：监督学习损失，衡量预测值与真实值的差异
    物理损失L_physics：基于Arrhenius动力学的物理约束损失

    支持一阶和二阶化学动力学：
    - 一阶：dC/dt = -k*C
    - 二阶：dC/dt = -k*C²

    Args:
        E_a (float): Arrhenius激活能（J/mol）
        k0 (float): Arrhenius频率因子
        R (float): 气体常数（8.314 J/(mol·K)）
        w_data (float): 数据损失权重
        w_physics (float): 物理损失权重
        kinetics_order (int): 反应阶数（1或2）
        data_loss_fn (Optional[Callable]): 自定义数据损失函数，默认MSE
        physics_loss_fn (Optional[Callable]): 自定义物理残差函数，默认MSE
    """

    def __init__(
        self,
        E_a: float = 100000.0,  # 激活能 (J/mol)
        k0: float = 1e-3,  # 频率因子
        R: float = 8.314,  # 气体常数
        w_data: float = 1.0,  # 数据损失权重
        w_physics: float = 0.001,  # 物理损失权重
        kinetics_order: int = 1,  # 动力学阶数 (1=一阶, 2=二阶)
        data_loss_fn: Optional[Callable] = None,
        physics_loss_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.E_a = E_a
        self.k0 = k0
        self.R = R
        self.w_data = w_data
        self.w_physics = w_physics
        self.kinetics_order = kinetics_order

        self.data_loss_fn = data_loss_fn or nn.MSELoss()
        self.physics_loss_fn = physics_loss_fn or nn.MSELoss()

    def set_weights(self, w_data: float, w_physics: float):
        """动态调整数据损失和物理损失的权重"""
        self.w_data = w_data
        self.w_physics = w_physics

    def compute_data_loss(
        self,
        C_pred: torch.Tensor,
        C_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算数据损失（监督学习损失）

        L_data = (1/N) * Σ(C_pred - C_true)²

        Args:
            C_pred (torch.Tensor): 预测的质量值，形状为[batch_size, 1]或[batch_size]
            C_true (torch.Tensor): 真实的质量值，形状为[batch_size, 1]或[batch_size]

        Returns:
            torch.Tensor: 标量数据损失值
        """
        return self.data_loss_fn(C_pred.squeeze(), C_true.squeeze())

    def compute_physics_loss(
        self,
        C_pred: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
        dC_dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算物理损失（基于Arrhenius动力学的约束）

        对于一阶动力学：dC/dt = -k*C
        物理残差：f = dC/dt + k*C = 0
        L_physics = (1/N) * Σ(f²)

        对于二阶动力学：dC/dt = -k*C²
        物理残差：f = dC/dt + k*C² = 0

        Args:
            C_pred (torch.Tensor): 预测的质量值，形状为[batch_size, 1]
            times (torch.Tensor): 时间，形状为[batch_size, 1]
            temperatures (torch.Tensor): 温度（Kelvin），形状为[batch_size, 1]
            dC_dt (torch.Tensor): 预测值的时间导数，形状为[batch_size, 1]

        Returns:
            torch.Tensor: 标量物理损失值
        """
        # 计算速率常数
        k = arrhenius_rate_constant(temperatures, self.E_a, self.k0, self.R)

        if self.kinetics_order == 1:
            # 一阶动力学: dC/dt = -k*C
            # 物理方程残差: f = dC/dt + k*C = 0
            residual = dC_dt + k * C_pred
        elif self.kinetics_order == 2:
            # 二阶动力学: dC/dt = -k*C^2
            # 物理方程残差: f = dC/dt + k*C^2 = 0
            residual = dC_dt + k * (C_pred ** 2)
        else:
            residual = dC_dt + k * C_pred

        # 物理损失
        physics_loss = (residual ** 2).mean()

        return physics_loss

    def forward(
        self,
        C_pred: torch.Tensor,
        C_true: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
        dC_dt: torch.Tensor,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算PINN总损失

        Args:
            C_pred (torch.Tensor): 预测的质量值，形状为[batch_size, 1]
            C_true (torch.Tensor): 真实的质量值，形状为[batch_size, 1]
            times (torch.Tensor): 时间，形状为[batch_size, 1]
            temperatures (torch.Tensor): 温度（Kelvin），形状为[batch_size, 1]
            dC_dt (torch.Tensor): 时间导数，形状为[batch_size, 1]
            labeled_mask (Optional[torch.Tensor]): 有标签样本的掩码，形状为[batch_size]

        Returns:
            Dict[str, torch.Tensor]: 损失字典
                - 'total': 总损失
                - 'data': 数据损失
                - 'physics': 物理损失
        """
        # 数据损失 (仅在有标签的数据上计算)
        if labeled_mask is not None:
            C_pred_labeled = C_pred[labeled_mask]
            C_true_labeled = C_true[labeled_mask]
        else:
            C_pred_labeled = C_pred
            C_true_labeled = C_true

        loss_data = self.compute_data_loss(C_pred_labeled, C_true_labeled)

        # 物理损失 (在所有配点上计算)
        loss_physics = self.compute_physics_loss(
            C_pred, times, temperatures, dC_dt
        )

        # 总损失
        total_loss = self.w_data * loss_data + self.w_physics * loss_physics

        return {
            'total': total_loss,
            'data': loss_data,
            'physics': loss_physics,
        }


class PINNLossWithKinetics(PINNLoss):
    """
    扩展的PINN损失函数：支持网络预测动力学参数

    在基础PINN损失基础上，允许网络同时学习Arrhenius参数（E_a和k0），
    并可添加动力学参数的光滑性约束。

    Args:
        w_kinetics_smooth (float): 动力学参数光滑性损失权重
        其他参数继承自PINNLoss
    """

    def __init__(
        self,
        R: float = 8.314,
        w_data: float = 1.0,
        w_physics: float = 0.001,
        w_kinetics_smooth: float = 0.0,
        kinetics_order: int = 1,
        **kwargs
    ):
        super().__init__(R=R, w_data=w_data, w_physics=w_physics,
                        kinetics_order=kinetics_order, **kwargs)

        self.w_kinetics_smooth = w_kinetics_smooth

    def compute_kinetics_smoothness_loss(
        self,
        E_a_batch: torch.Tensor,
        k0_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算动力学参数的光滑性损失

        通过最小化参数的方差来促进E_a和k0的稳定性，
        避免在不同样本间出现不合理的参数波动。

        Args:
            E_a_batch (torch.Tensor): 预测的激活能，形状为[batch_size, 1]
            k0_batch (torch.Tensor): 预测的频率因子，形状为[batch_size, 1]

        Returns:
            torch.Tensor: 光滑性损失（标量）
        """
        # 方差正则化
        E_a_loss = E_a_batch.var()
        k0_loss = k0_batch.var()

        return E_a_loss + k0_loss

    def forward(
        self,
        C_pred: torch.Tensor,
        C_true: torch.Tensor,
        E_a_pred: Optional[torch.Tensor],
        k0_pred: Optional[torch.Tensor],
        times: torch.Tensor,
        temperatures: torch.Tensor,
        dC_dt: torch.Tensor,
        labeled_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算扩展的PINN损失（支持网络预测动力学参数）

        Args:
            C_pred (torch.Tensor): 预测的质量值，形状为[batch_size, 1]
            C_true (torch.Tensor): 真实的质量值，形状为[batch_size, 1]
            E_a_pred (Optional[torch.Tensor]): 预测的激活能，形状为[batch_size, 1]
            k0_pred (Optional[torch.Tensor]): 预测的频率因子，形状为[batch_size, 1]
            times (torch.Tensor): 时间，形状为[batch_size, 1]
            temperatures (torch.Tensor): 温度（Kelvin），形状为[batch_size, 1]
            dC_dt (torch.Tensor): 时间导数，形状为[batch_size, 1]
            labeled_mask (Optional[torch.Tensor]): 有标签样本的掩码

        Returns:
            Dict[str, torch.Tensor]: 扩展的损失字典
                - 'total': 总损失
                - 'data': 数据损失
                - 'physics': 物理损失
                - 'kinetics': 动力学参数光滑性损失（如果启用）
        """
        # 使用预测的动力学参数
        if E_a_pred is not None and k0_pred is not None:
            E_a = E_a_pred.mean().item()
            k0 = k0_pred.mean().item()

        # 基础损失
        losses = super().forward(C_pred, C_true, times, temperatures, dC_dt, labeled_mask)

        # 动力学光滑性损失
        if self.w_kinetics_smooth > 0 and E_a_pred is not None and k0_pred is not None:
            kinetics_loss = self.compute_kinetics_smoothness_loss(E_a_pred, k0_pred)
            losses['kinetics'] = kinetics_loss
            losses['total'] += self.w_kinetics_smooth * kinetics_loss

        return losses
