"""
PINN网络架构模块

该模块定义了基于物理信息的神经网络(PINN)架构，用于多模态数据融合和物质降解预测。

主要组件：
- PINNNetwork: 基础PINN网络，支持NIR和E-nose特征融合
- PINNNetworkWithKinetics: 扩展网络，同时预测质量和动力学参数

网络特点：
- 多输入分支结构（NIR分支、E-nose分支）
- 物理参数直接输入（时间、温度）
- 支持自动求导计算物理约束
- 可选的动力学参数预测
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class PINNNetwork(nn.Module):
    """
    物理信息神经网络(PINN)架构

    该网络采用多分支结构融合NIR光谱和E-nose传感器数据，
    并直接将物理参数（时间、温度）作为网络输入，实现数据驱动与物理约束的结合。

    网络结构：
    - NIR分支：处理光谱特征提取
    - E-nose分支：处理传感器特征提取
    - 融合层：整合所有特征和物理参数
    - 输出层：预测质量指标

    Args:
        nir_dim (int): NIR光谱输入维度
        nose_dim (int): E-nose传感器输入维度
        nir_hidden (Tuple[int, ...]): NIR分支各隐层神经元数量
        nose_hidden (Tuple[int, ...]): E-nose分支各隐层神经元数量
        shared_hidden (Tuple[int, ...]): 融合层各隐层神经元数量
        physics_dim (int): 物理参数维度（时间+温度=2）
        output_dim (int): 输出维度（通常为1，表示质量预测）
        activation (str): 激活函数类型（'relu'或'tanh'）
        dropout (float): Dropout比例（0-1之间）
        output_activaton (Optional[str]): 输出层激活函数，可选'sigmoid'或'relu'
    """

    def __init__(
        self,
        nir_dim: int = 50,
        nose_dim: int = 10,
        nir_hidden: Tuple[int, ...] = (128, 64),
        nose_hidden: Tuple[int, ...] = (32, 16),
        shared_hidden: Tuple[int, ...] = (128, 64, 32),
        physics_dim: int = 2,
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.0,
        output_activaton: Optional[str] = None,
    ):
        super(PINNNetwork, self).__init__()

        self.nir_dim = nir_dim
        self.nose_dim = nose_dim
        self.physics_dim = physics_dim
        self.activation_name = activation

        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        output_act = None
        if output_activaton == 'sigmoid':
            output_act = nn.Sigmoid()
        elif output_activaton == 'relu':
            output_act = nn.ReLU()

        # ========== NIR特征提取分支 ==========
        nir_layers = []
        in_dim = nir_dim
        for hidden_dim in nir_hidden:
            nir_layers.append(nn.Linear(in_dim, hidden_dim))
            nir_layers.append(self.activation)
            if dropout > 0:
                nir_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.nir_branch = nn.Sequential(*nir_layers)
        self.nir_output_dim = in_dim

        # ========== E-nose特征提取分支 ==========
        nose_layers = []
        in_dim = nose_dim
        for hidden_dim in nose_hidden:
            nose_layers.append(nn.Linear(in_dim, hidden_dim))
            nose_layers.append(self.activation)
            if dropout > 0:
                nose_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.nose_branch = nn.Sequential(*nose_layers)
        self.nose_output_dim = in_dim

        # ========== 融合层 ==========
        # 拼接: [F_nir, F_nose, t, T]
        fusion_input_dim = self.nir_output_dim + self.nose_output_dim + physics_dim

        shared_layers = []
        in_dim = fusion_input_dim
        for hidden_dim in shared_hidden:
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            # 融合层使用Tanh (对求导更友好)
            shared_layers.append(nn.Tanh())
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.shared_layers = nn.Sequential(*shared_layers)
        shared_output_dim = in_dim

        # ========== 输出层 ==========
        self.output_layer = nn.Linear(shared_output_dim, output_dim)
        if output_act is not None:
            self.output_layer = nn.Sequential(
                self.output_layer,
                output_act
            )

    def forward(
        self,
        X_nir: torch.Tensor,
        X_nose: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播计算质量预测值

        Args:
            X_nir (torch.Tensor): NIR光谱特征，形状为[batch_size, nir_dim]
            X_nose (torch.Tensor): E-nose传感器特征，形状为[batch_size, nose_dim]
            times (torch.Tensor): 采样时间，形状为[batch_size, 1]
            temperatures (torch.Tensor): 温度条件，形状为[batch_size, 1]

        Returns:
            torch.Tensor: 质量预测值，形状为[batch_size, output_dim]
        """
        # 特征提取
        f_nir = self.nir_branch(X_nir)  # [Batch, nir_hidden[-1]]
        f_nose = self.nose_branch(X_nose)  # [Batch, nose_hidden[-1]]

        # 拼接所有输入
        fused = torch.cat([f_nir, f_nose, times, temperatures], dim=1)

        # 共享层
        fused = self.shared_layers(fused)

        # 输出
        C_pred = self.output_layer(fused)

        return C_pred

    def get_gradient_wrt_time(
        self,
        X_nir: torch.Tensor,
        X_nose: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算预测质量对时间的导数 dC/dt

        该导数用于计算基于Arrhenius方程的物理损失函数。

        Args:
            X_nir (torch.Tensor): NIR光谱特征，形状为[batch_size, nir_dim]
            X_nose (torch.Tensor): E-nose传感器特征，形状为[batch_size, nose_dim]
            times (torch.Tensor): 采样时间，形状为[batch_size, 1]，需要requires_grad=True
            temperatures (torch.Tensor): 温度条件，形状为[batch_size, 1]

        Returns:
            torch.Tensor: 时间导数 dC/dt，形状为[batch_size, 1]
        """
        C_pred = self.forward(X_nir, X_nose, times, temperatures)

        # 自动求导
        dC_dt = torch.autograd.grad(
            outputs=C_pred,
            inputs=times,
            grad_outputs=torch.ones_like(C_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        return dC_dt

    def get_gradient_wrt_temperature(
        self,
        X_nir: torch.Tensor,
        X_nose: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算预测质量对温度的导数 dC/dT

        Args:
            X_nir (torch.Tensor): NIR光谱特征，形状为[batch_size, nir_dim]
            X_nose (torch.Tensor): E-nose传感器特征，形状为[batch_size, nose_dim]
            times (torch.Tensor): 采样时间，形状为[batch_size, 1]
            temperatures (torch.Tensor): 温度条件，形状为[batch_size, 1]，需要requires_grad=True

        Returns:
            torch.Tensor: 温度导数 dC/dT，形状为[batch_size, 1]
        """
        C_pred = self.forward(X_nir, X_nose, times, temperatures)

        dC_dT = torch.autograd.grad(
            outputs=C_pred,
            inputs=temperatures,
            grad_outputs=torch.ones_like(C_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        return dC_dT


class PINNNetworkWithKinetics(PINNNetwork):
    """
    扩展的PINN网络：同时预测质量和动力学参数

    在基础PINN网络基础上，增加对Arrhenius动力学参数（激活能E_a和频率因子k0）的预测。
    这使得网络能够学习物质降解的内在动力学机制，而不仅仅是表观模式。

    Args:
        predict_kinetics (bool): 是否预测动力学参数E_a和k0
        其他参数继承自PINNNetwork
    """

    def __init__(
        self,
        nir_dim: int = 50,
        nose_dim: int = 10,
        nir_hidden: Tuple[int, ...] = (128, 64),
        nose_hidden: Tuple[int, ...] = (32, 16),
        shared_hidden: Tuple[int, ...] = (128, 64, 32),
        physics_dim: int = 2,
        predict_kinetics: bool = True,
        **kwargs
    ):
        self.predict_kinetics = predict_kinetics
        super().__init__(
            nir_dim=nir_dim,
            nose_dim=nose_dim,
            nir_hidden=nir_hidden,
            nose_hidden=nose_hidden,
            shared_hidden=shared_hidden,
            physics_dim=physics_dim,
            output_dim=3 if predict_kinetics else 1,
            **kwargs
        )

    def forward(
        self,
        X_nir: torch.Tensor,
        X_nose: torch.Tensor,
        times: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播，同时输出质量预测和动力学参数

        Returns:
            如果predict_kinetics=True: 返回(C_pred, E_a, k0)
                - C_pred: 质量预测值，形状为[batch_size, 1]
                - E_a: 激活能(kJ/mol)，范围50-500，形状为[batch_size, 1]
                - k0: 频率因子，范围1e-5到1e2，形状为[batch_size, 1]
            否则: 返回C_pred (与基础网络相同)
        """
        output = super().forward(X_nir, X_nose, times, temperatures)

        if self.predict_kinetics:
            C_pred = output[:, :1]
            # E_a: 激活能 (50 - 500 kJ/mol, 归一化到 [0,1])
            E_a = output[:, 1:2] * 450 + 50
            # k0: 频率因子 (1e-5 - 1e2, log scale)
            k0 = torch.exp(output[:, 2:3] * 9.21 - 11.51)  # log10(k0) in [-5, 4]
            return C_pred, E_a, k0
        else:
            return output
