"""
PINN训练策略模块

该模块实现了PINN（物理信息神经网络）的训练策略，
包括单阶段和两阶段优化方法。

主要功能：
- 单阶段训练：使用Adam优化器进行快速收敛
- 两阶段训练：Adam + L-BFGS组合优化策略
- 动态权重调整：训练过程中调整数据损失和物理损失的权重
- 训练历史记录：损失曲线和性能指标跟踪

训练策略：
- 第一阶段（Adam）：快速探索解空间，平衡数据和物理约束
- 第二阶段（L-BFGS）：精细优化，提高解的精度和稳定性
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import warnings

warnings.filterwarnings('ignore')


def train_pinn(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    collocation_data: Optional[Dict[str, torch.Tensor]] = None,
    epochs: int = 5000,
    learning_rate: float = 0.001,
    optimizer_type: str = 'adam',
    device: str = 'cpu',
    verbose: bool = True,
    checkpoint_every: int = 500,
) -> Dict[str, List[float]]:
    """
    单阶段PINN训练函数

    使用单一优化器（默认Adam）进行PINN训练，平衡数据驱动损失和物理约束损失。

    训练过程：
    1. 前向传播计算预测值
    2. 自动微分计算时间导数
    3. 计算PINN总损失（数据损失+物理损失）
    4. 反向传播更新参数

    Args:
        model (torch.nn.Module): PINN网络模型
        loss_fn (torch.nn.Module): PINN损失函数（包含数据和物理损失）
        train_data (Dict[str, torch.Tensor]): 训练数据字典
            - 'X_nir': NIR光谱数据，形状为[batch_size, nir_features]
            - 'X_enose': 电子鼻数据，形状为[batch_size, enose_features]
            - 'times': 时间序列，形状为[batch_size, 1]
            - 'temperatures': 温度序列，形状为[batch_size, 1]
            - 'y': 目标质量值，形状为[batch_size, 1]
        val_data (Optional[Dict[str, torch.Tensor]]): 验证数据字典，格式同train_data
        collocation_data (Optional[Dict[str, torch.Tensor]]): 配点数据字典，用于计算物理损失
            如果为None，则使用train_data作为配点数据
        epochs (int): 训练轮数，默认5000
        learning_rate (float): 学习率，默认0.001
        optimizer_type (str): 优化器类型，支持'adam'、'sgd'等，默认'adam'
        device (str): 计算设备，'cpu'或'cuda'，默认'cpu'
        verbose (bool): 是否输出训练日志，默认True
        checkpoint_every (int): 每多少轮输出一次日志，默认500

    Returns:
        Dict[str, List[float]]: 训练历史字典
            - 'train_loss': 训练总损失历史
            - 'val_loss': 验证损失历史（如果有验证数据）
            - 'data_loss': 数据损失历史
            - 'physics_loss': 物理损失历史
    """
    model = model.to(device)

    # 选择优化器
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 使用train_data作为collocation_data的默认值
    if collocation_data is None:
        collocation_data = train_data

    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
    }

    # 移到设备
    train_X_nir = train_data['X_nir'].to(device)
    train_X_nose = train_data['X_enose'].to(device)
    train_times = train_data['times'].to(device).requires_grad_(True)
    train_temps = train_data['temperatures'].to(device)
    train_y = train_data['y'].to(device)

    coll_X_nir = collocation_data['X_nir'].to(device)
    coll_X_nose = collocation_data['X_enose'].to(device)
    coll_times = collocation_data['times'].to(device).requires_grad_(True)
    coll_temps = collocation_data['temperatures'].to(device)
    coll_y = collocation_data['y'].to(device)

    # 训练循环
    for epoch in range(epochs):
        model.train()

        # 前向传播
        C_pred = model(train_X_nir, train_X_nose, train_times, train_temps)

        # 计算物理导数 (使用collocation数据)
        coll_C_pred = model(coll_X_nir, coll_X_nose, coll_times, coll_temps)
        dC_dt = torch.autograd.grad(
            outputs=coll_C_pred.sum(),
            inputs=coll_times,
            create_graph=True,
            retain_graph=True,
        )[0]

        # 计算损失
        losses = loss_fn(
            C_pred=coll_C_pred,
            C_true=coll_y,
            times=coll_times,
            temperatures=coll_temps,
            dC_dt=dC_dt,
            labeled_mask=None
        )

        total_loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 记录损失
        history['train_loss'].append(total_loss.item())
        history['data_loss'].append(losses['data'].item())
        history['physics_loss'].append(losses['physics'].item())

        # 验证集评估
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_X_nir = val_data['X_nir'].to(device)
                val_X_nose = val_data['X_enose'].to(device)
                val_times = val_data['times'].to(device)
                val_temps = val_data['temperatures'].to(device)
                val_y = val_data['y'].to(device)

                val_C_pred = model(val_X_nir, val_X_nose, val_times, val_temps)
                val_loss = torch.nn.functional.mse_loss(val_C_pred.squeeze(), val_y.squeeze())
                history['val_loss'].append(val_loss.item())

        # 输出日志
        if verbose and (epoch + 1) % checkpoint_every == 0:
            msg = f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss.item():.6f}"
            msg += f" | Data: {losses['data'].item():.6f} | Physics: {losses['physics'].item():.6f}"
            if val_data is not None:
                msg += f" | Val: {history['val_loss'][-1]:.6f}"
            print(msg)

    return history


def train_pinn_two_stage(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_data: Dict[str, torch.Tensor],
    val_data: Optional[Dict[str, torch.Tensor]] = None,
    collocation_data: Optional[Dict[str, torch.Tensor]] = None,
    stage1_epochs: int = 5000,
    stage2_epochs: int = 2000,
    stage1_lr: float = 0.001,
    stage2_lr: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True,
    w_data_schedule: Optional[Callable] = None,
    w_physics_schedule: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """
    两阶段PINN训练策略

    采用分阶段优化策略提高训练效果：
    - 第一阶段（Adam）：快速收敛，探索解空间
    - 第二阶段（L-BFGS）：精细优化，提高精度

    优势：
    - Adam阶段：对噪声和不精确梯度有更好的鲁棒性
    - L-BFGS阶段：利用二阶导数信息，实现更精确的优化
    - 动态权重：训练过程中可调整数据和物理损失的相对重要性

    Args:
        model (torch.nn.Module): PINN网络模型
        loss_fn (torch.nn.Module): PINN损失函数
        train_data (Dict[str, torch.Tensor]): 训练数据字典
        val_data (Optional[Dict[str, torch.Tensor]]): 验证数据字典
        collocation_data (Optional[Dict[str, torch.Tensor]]): 配点数据字典
        stage1_epochs (int): 第一阶段训练轮数，默认5000
        stage2_epochs (int): 第二阶段训练轮数，默认2000
        stage1_lr (float): 第一阶段学习率，默认0.001
        stage2_lr (float): 第二阶段学习率，默认1.0
        device (str): 计算设备，默认'cpu'
        verbose (bool): 是否输出训练日志，默认True
        w_data_schedule (Optional[Callable]): 数据损失权重调度函数
            格式：f(epoch, total_epochs) -> weight
        w_physics_schedule (Optional[Callable]): 物理损失权重调度函数
            格式：f(epoch, total_epochs) -> weight

    Returns:
        Dict[str, List[float]]: 合并的训练历史字典
            - 'train_loss': 总训练损失历史（两个阶段合并）
            - 'val_loss': 验证损失历史
            - 'data_loss': 数据损失历史
            - 'physics_loss': 物理损失历史
    """

    print("=" * 60)
    print("STAGE 1: Adam Optimization")
    print("=" * 60)

    # 第一阶段: Adam
    history1 = train_pinn(
        model=model,
        loss_fn=loss_fn,
        train_data=train_data,
        val_data=val_data,
        collocation_data=collocation_data,
        epochs=stage1_epochs,
        learning_rate=stage1_lr,
        optimizer_type='adam',
        device=device,
        verbose=verbose,
        checkpoint_every=max(1, stage1_epochs // 10),
    )

    print("\n" + "=" * 60)
    print("STAGE 2: L-BFGS Optimization (Fine-tuning)")
    print("=" * 60)

    # 第二阶段: L-BFGS
    model = model.to(device)
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=stage2_lr,
        max_iter=20,  # L-BFGS每步的最大迭代数
        line_search_fn='strong_wolfe'
    )

    train_X_nir = train_data['X_nir'].to(device)
    train_X_nose = train_data['X_enose'].to(device)
    train_times = train_data['times'].to(device).requires_grad_(True)
    train_temps = train_data['temperatures'].to(device)
    train_y = train_data['y'].to(device)

    coll_X_nir = (collocation_data or train_data)['X_nir'].to(device)
    coll_X_nose = (collocation_data or train_data)['X_enose'].to(device)
    coll_times = (collocation_data or train_data)['times'].to(device).requires_grad_(True)
    coll_temps = (collocation_data or train_data)['temperatures'].to(device)
    coll_y = (collocation_data or train_data)['y'].to(device)

    history2 = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
    }

    for epoch in range(stage2_epochs):
        model.train()

        # 动态调整权重
        total_epochs = stage1_epochs + stage2_epochs
        current_epoch = stage1_epochs + epoch

        if w_data_schedule is not None:
            w_d = w_data_schedule(current_epoch, total_epochs)
            loss_fn.set_weights(w_d, loss_fn.w_physics)

        if w_physics_schedule is not None:
            w_p = w_physics_schedule(current_epoch, total_epochs)
            loss_fn.set_weights(loss_fn.w_data, w_p)

        def closure():
            optimizer_lbfgs.zero_grad()

            C_pred = model(train_X_nir, train_X_nose, train_times, train_temps)

            coll_C_pred = model(coll_X_nir, coll_X_nose, coll_times, coll_temps)
            dC_dt = torch.autograd.grad(
                outputs=coll_C_pred.sum(),
                inputs=coll_times,
                create_graph=True,
                retain_graph=True,
            )[0]

            losses = loss_fn(
                C_pred=coll_C_pred,
                C_true=coll_y,
                times=coll_times,
                temperatures=coll_temps,
                dC_dt=dC_dt,
                labeled_mask=None
            )

            total_loss = losses['total']
            total_loss.backward()

            return total_loss

        loss_value = optimizer_lbfgs.step(closure)

        # 记录损失
        with torch.no_grad():
            C_pred = model(train_X_nir, train_X_nose, train_times, train_temps)
            coll_C_pred = model(coll_X_nir, coll_X_nose, coll_times, coll_temps)
            dC_dt = torch.autograd.grad(
                outputs=coll_C_pred.sum(),
                inputs=coll_times,
                create_graph=True,
            )[0]
            losses = loss_fn(
                C_pred=coll_C_pred,
                C_true=coll_y,
                times=coll_times,
                temperatures=coll_temps,
                dC_dt=dC_dt,
            )

            history2['train_loss'].append(losses['total'].item())
            history2['data_loss'].append(losses['data'].item())
            history2['physics_loss'].append(losses['physics'].item())

        if verbose and (epoch + 1) % max(1, stage2_epochs // 5) == 0:
            print(f"L-BFGS Epoch {epoch + 1}/{stage2_epochs} | Loss: {loss_value.item():.6f}")

    # 合并历史
    history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'data_loss': history1['data_loss'] + history2['data_loss'],
        'physics_loss': history1['physics_loss'] + history2['physics_loss'],
    }

    if history1.get('val_loss'):
        history['val_loss'] = history1['val_loss'] + history2.get('val_loss', [])

    return history


def weight_schedule_linear(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """
    线性权重调度函数

    在训练过程中线性地调整损失权重，从w_start逐渐变化到w_end。

    Args:
        epoch (int): 当前训练轮数
        total_epochs (int): 总训练轮数
        w_start (float): 初始权重
        w_end (float): 最终权重

    Returns:
        float: 当前轮数的权重值

    Example:
        # 数据损失权重从1.0线性增加到2.0
        w_data_schedule = lambda e, t: weight_schedule_linear(e, t, 1.0, 2.0)
    """
    return w_start + (w_end - w_start) * (epoch / max(1, total_epochs))


def weight_schedule_exponential(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """
    指数权重调度函数

    在训练过程中指数地调整损失权重，从w_start逐渐变化到w_end。

    Args:
        epoch (int): 当前训练轮数
        total_epochs (int): 总训练轮数
        w_start (float): 初始权重
        w_end (float): 最终权重

    Returns:
        float: 当前轮数的权重值

    Example:
        # 物理损失权重从0.001指数增加到0.01
        w_physics_schedule = lambda e, t: weight_schedule_exponential(e, t, 0.001, 0.01)
    """
    progress = epoch / max(1, total_epochs)
    return w_start * (w_end / w_start) ** progress


def weight_schedule_linear(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """线性权重计划"""
    return w_start + (w_end - w_start) * (epoch / max(1, total_epochs))


def weight_schedule_exponential(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """指数权重计划"""
    progress = epoch / max(1, total_epochs)
    return w_start * (w_end / w_start) ** progress
