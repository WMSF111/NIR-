"""PINN 模型的训练工具函数。

本模块实现了PINN的单阶段和两阶段训练策略，支持Adam预热和L-BFGS精调。
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import torch
import torch.optim as optim

warnings.filterwarnings('ignore')


def _compute_pinn_losses(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_batch: Dict[str, torch.Tensor],
    collocation_batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """分别在监督批次和配点批次上计算数据损失与物理损失。

    该函数用于PINN训练的核心损失计算，
    同时覆盖模型预测和物理约束项。
    """
    C_pred = model(
        train_batch['X_nir'],
        train_batch['X_enose'],
        train_batch['times'],
        train_batch['temperatures'],
    )
    coll_C_pred = model(
        collocation_batch['X_nir'],
        collocation_batch['X_enose'],
        collocation_batch['times'],
        collocation_batch['temperatures'],
    )
    dC_dt = torch.autograd.grad(
        outputs=coll_C_pred.sum(),
        inputs=collocation_batch['times'],
        create_graph=True,
        retain_graph=True,
    )[0]

    loss_data = loss_fn.compute_data_loss(C_pred, train_batch['y'])
    loss_physics = loss_fn.compute_physics_loss(
        coll_C_pred,
        collocation_batch['times'],
        collocation_batch['temperatures'],
        dC_dt,
    )
    total_loss = loss_fn.w_data * loss_data + loss_fn.w_physics * loss_physics

    return {
        'total': total_loss,
        'data': loss_data,
        'physics': loss_physics,
    }


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
    """使用单阶段优化器训练 PINN 模型。

    该函数支持Adam和SGD优化器，
    适用于快速验证PINN模型的训练效果。
    """
    model = model.to(device)

    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if collocation_data is None:
        collocation_data = train_data

    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
    }

    train_batch = {
        'X_nir': train_data['X_nir'].to(device),
        'X_enose': train_data['X_enose'].to(device),
        'times': train_data['times'].to(device).requires_grad_(True),
        'temperatures': train_data['temperatures'].to(device),
        'y': train_data['y'].to(device),
    }
    collocation_batch = {
        'X_nir': collocation_data['X_nir'].to(device),
        'X_enose': collocation_data['X_enose'].to(device),
        'times': collocation_data['times'].to(device).requires_grad_(True),
        'temperatures': collocation_data['temperatures'].to(device),
        'y': collocation_data['y'].to(device),
    }

    for epoch in range(epochs):
        model.train()
        losses = _compute_pinn_losses(model, loss_fn, train_batch, collocation_batch)

        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        history['train_loss'].append(losses['total'].item())
        history['data_loss'].append(losses['data'].item())
        history['physics_loss'].append(losses['physics'].item())

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

        if verbose and (epoch + 1) % checkpoint_every == 0:
            msg = f"Epoch {epoch + 1}/{epochs} | Loss: {losses['total'].item():.6f}"
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
    total_epochs: int = 3000,
    stage1_ratio: float = 0.67,
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
    stage1_lr: float = 0.001,
    stage2_lr: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True,
    w_data_schedule: Optional[Callable] = None,
    w_physics_schedule: Optional[Callable] = None,
    epochs: Optional[int] = None,
) -> Dict[str, List[float]]:
    """先用 Adam 预热，再用 L-BFGS 精调的两阶段 PINN 训练。

    该函数实现了典型的PINN训练策略：先用Adam稳定训练，
    再用L-BFGS做精细最优化。
    
    默认使用 `total_epochs` 作为总训练轮数语义，并通过 `stage1_ratio`
    自动切分两阶段轮数。若显式传入 `stage1_epochs` 和 `stage2_epochs`，
    则优先使用显式配置。
    """
    if epochs is not None:
        total_epochs = epochs

    if not 0 < stage1_ratio < 1:
        raise ValueError('stage1_ratio must be between 0 and 1')

    if stage1_epochs is None or stage2_epochs is None:
        stage1_epochs = max(1, int(round(total_epochs * stage1_ratio)))
        stage2_epochs = max(1, total_epochs - stage1_epochs)

    total_epochs = stage1_epochs + stage2_epochs

    if verbose:
        print("=" * 60)
        print("STAGE 1: Adam Optimization")
        print("=" * 60)

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

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: L-BFGS Optimization (Fine-tuning)")
        print("=" * 60)

    model = model.to(device)
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(),
        lr=stage2_lr,
        max_iter=20,
        line_search_fn='strong_wolfe',
    )

    train_batch = {
        'X_nir': train_data['X_nir'].to(device),
        'X_enose': train_data['X_enose'].to(device),
        'times': train_data['times'].to(device).requires_grad_(True),
        'temperatures': train_data['temperatures'].to(device),
        'y': train_data['y'].to(device),
    }
    source_batch = collocation_data or train_data
    collocation_batch = {
        'X_nir': source_batch['X_nir'].to(device),
        'X_enose': source_batch['X_enose'].to(device),
        'times': source_batch['times'].to(device).requires_grad_(True),
        'temperatures': source_batch['temperatures'].to(device),
        'y': source_batch['y'].to(device),
    }

    history2 = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'physics_loss': [],
    }

    for epoch in range(stage2_epochs):
        model.train()

        current_epoch = stage1_epochs + epoch

        if w_data_schedule is not None:
            loss_fn.set_weights(w_data_schedule(current_epoch, total_epochs), loss_fn.w_physics)
        if w_physics_schedule is not None:
            loss_fn.set_weights(loss_fn.w_data, w_physics_schedule(current_epoch, total_epochs))

        def closure():
            optimizer_lbfgs.zero_grad()
            losses = _compute_pinn_losses(model, loss_fn, train_batch, collocation_batch)
            losses['total'].backward()
            return losses['total']

        loss_value = optimizer_lbfgs.step(closure)

        with torch.enable_grad():
            losses = _compute_pinn_losses(model, loss_fn, train_batch, collocation_batch)
            history2['train_loss'].append(losses['total'].item())
            history2['data_loss'].append(losses['data'].item())
            history2['physics_loss'].append(losses['physics'].item())

        if verbose and (epoch + 1) % max(1, stage2_epochs // 5) == 0:
            display_loss = loss_value.item() if hasattr(loss_value, 'item') else float(loss_value)
            if verbose:
                print(f"L-BFGS Epoch {epoch + 1}/{stage2_epochs} | Loss: {display_loss:.6f}")

    history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'data_loss': history1['data_loss'] + history2['data_loss'],
        'physics_loss': history1['physics_loss'] + history2['physics_loss'],
    }
    if history1.get('val_loss'):
        history['val_loss'] = history1['val_loss'] + history2.get('val_loss', [])

    return history


def weight_schedule_linear(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """在线性训练进度上插值损失权重。"""
    return w_start + (w_end - w_start) * (epoch / max(1, total_epochs))


def weight_schedule_exponential(epoch: int, total_epochs: int, w_start: float, w_end: float) -> float:
    """按指数方式随训练进度插值损失权重。"""
    progress = epoch / max(1, total_epochs)
    return w_start * (w_end / w_start) ** progress
