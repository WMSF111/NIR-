"""PINN 模型的评估与可视化工具。"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


def evaluate_scalar_observations(
    predictions: np.ndarray,
    observations: np.ndarray,
    time_points: np.ndarray,
    sample_ids: List[int],
) -> Dict[str, float]:
    """评估 PINN 在稀疏标量观测点上的预测精度。"""
    predictions = np.asarray(predictions)
    observations = np.asarray(observations)
    time_points = np.asarray(time_points)

    if predictions.ndim != 1:
        raise ValueError(f'期望 1 维 predictions，实际为 {predictions.ndim} 维')
    if observations.ndim != 1:
        raise ValueError(f'期望 1 维 observations，实际为 {observations.ndim} 维')
    if time_points.ndim != 1:
        raise ValueError(f'期望 1 维 time_points，实际为 {time_points.ndim} 维')
    if len(predictions) != len(observations) or len(predictions) != len(time_points) or len(predictions) != len(sample_ids):
        raise ValueError('predictions、observations、time_points 与 sample_ids 的长度必须一致')

    r2 = r2_score(observations, predictions)
    rmse = np.sqrt(mean_squared_error(observations, predictions))
    mae = np.mean(np.abs(observations - predictions))
    std_true = np.std(observations)
    rpd = std_true / (rmse + 1e-10)
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'rpd': rpd,
    }


def evaluate_trajectory_predictions(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    time_grid: np.ndarray,
) -> Dict[str, float]:
    """评估完整时序轨迹预测。"""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    time_grid = np.asarray(time_grid)

    if predictions.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError('完整轨迹评估要求 predictions 和 ground_truth 都是 2 维数组')
    if predictions.shape != ground_truth.shape:
        raise ValueError('predictions 和 ground_truth 的形状必须一致')
    if time_grid.ndim != 1 or time_grid.shape[0] != predictions.shape[1]:
        raise ValueError('time_grid 必须是一维，且长度等于轨迹时间步数')

    return {
        'r2': float(r2_score(ground_truth.reshape(-1), predictions.reshape(-1))),
        'rmse': float(np.sqrt(mean_squared_error(ground_truth.reshape(-1), predictions.reshape(-1)))),
        'mae': float(np.mean(np.abs(predictions - ground_truth))),
    }


def evaluate_pinn(
    model: torch.nn.Module,
    X_nir: torch.Tensor,
    X_enose: torch.Tensor,
    times: torch.Tensor,
    temperatures: torch.Tensor,
    y_true: np.ndarray,
    device: str = 'cpu',
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    评估PINN模型性能

    计算常用的回归评估指标，包括决定系数、均方根误差等。
    这些指标用于衡量模型的预测准确性和泛化能力。

    Args:
        model (torch.nn.Module): 已训练的PINN网络模型
        X_nir (torch.Tensor): NIR光谱特征数据，形状为[batch_size, nir_features]
        X_enose (torch.Tensor): 电子鼻特征数据，形状为[batch_size, enose_features]
        times (torch.Tensor): 采样时间序列，形状为[batch_size, 1]
        temperatures (torch.Tensor): 温度序列，形状为[batch_size, 1]，单位为Kelvin
        y_true (np.ndarray): 真实质量值标签，形状为[batch_size]或[batch_size, 1]
        device (str): 计算设备，'cpu'或'cuda'，默认'cpu'
        return_predictions (bool): 是否同时返回预测值，默认False

    Returns:
        Dict[str, float]: 评估指标字典
            - 'r2': 决定系数R²（0-1，越高越好）
            - 'rmse': 均方根误差RMSE（越小越好）
            - 'mae': 平均绝对误差MAE（越小越好）
            - 'rpd': 相对分析误差RPD（>3优秀，>2良好）

        如果return_predictions=True，则返回(metrics, predictions)的元组
    """
    model.eval()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        X_nir = X_nir.to(device)
        X_enose = X_enose.to(device)
        times = times.to(device)
        temperatures = temperatures.to(device)
        
        y_pred = model(X_nir, X_enose, times, temperatures).cpu().numpy().flatten()
    
    y_true = y_true.flatten() if isinstance(y_true, torch.Tensor) else np.array(y_true).flatten()
    time_points = times.detach().cpu().numpy().reshape(-1)
    sample_ids = list(range(len(y_true)))
    metrics = evaluate_scalar_observations(y_pred, y_true, time_points, sample_ids)
    
    if return_predictions:
        return metrics, y_pred
    else:
        return metrics


def compare_models(
    models: Dict[str, torch.nn.Module],
    X_nir: torch.Tensor,
    X_enose: torch.Tensor,
    times: torch.Tensor,
    temperatures: torch.Tensor,
    y_true: np.ndarray,
    device: str = 'cpu',
) -> Dict[str, Dict[str, float]]:
    """
    对比多个模型的性能

    同时评估多个PINN模型或传统机器学习模型的性能，
    为模型选择和性能比较提供依据。

    Args:
        models (Dict[str, torch.nn.Module]): 模型字典，格式为{'模型名称': model}
        X_nir (torch.Tensor): NIR光谱特征数据
        X_enose (torch.Tensor): 电子鼻特征数据
        times (torch.Tensor): 时间序列
        temperatures (torch.Tensor): 温度序列
        y_true (np.ndarray): 真实标签
        device (str): 计算设备

    Returns:
        Dict[str, Dict[str, float]]: 评估结果字典
            格式：{'模型名称': {'r2': 值, 'rmse': 值, 'mae': 值, 'rpd': 值}}
    """
    results = {}
    
    for name, model in models.items():
        print(f"评估 {name}...")
        metrics = evaluate_pinn(
            model=model,
            X_nir=X_nir,
            X_enose=X_enose,
            times=times,
            temperatures=temperatures,
            y_true=y_true,
            device=device,
        )
        results[name] = metrics
        print(f"  R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}, RPD: {metrics['rpd']:.2f}")
    
    return results


def plot_comparison(
    models: Dict[str, torch.nn.Module],
    X_nir: torch.Tensor,
    X_enose: torch.Tensor,
    times: torch.Tensor,
    temperatures: torch.Tensor,
    y_true: np.ndarray,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    绘制预测值vs真实值对比散点图

    为每个模型生成散点图，展示预测值与真实值的相关性。
    对角线表示完美预测，数据点越接近对角线，模型性能越好。

    Args:
        models (Dict[str, torch.nn.Module]): 模型字典
        X_nir (torch.Tensor): NIR特征数据
        X_enose (torch.Tensor): 电子鼻特征数据
        times (torch.Tensor): 时间序列
        temperatures (torch.Tensor): 温度序列
        y_true (np.ndarray): 真实标签
        save_path (Optional[str]): 图表保存路径，如果为None则显示在屏幕上
        device (str): 计算设备
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    y_true_np = y_true.flatten() if isinstance(y_true, torch.Tensor) else np.array(y_true).flatten()
    min_val = min(y_true_np.min(), y_true_np.min())
    max_val = max(y_true_np.max(), y_true_np.max())
    
    for idx, (name, model) in enumerate(models.items()):
        metrics, y_pred = evaluate_pinn(
            model=model,
            X_nir=X_nir,
            X_enose=X_enose,
            times=times,
            temperatures=temperatures,
            y_true=y_true,
            device=device,
            return_predictions=True,
        )
        
        ax = axes[idx]
        ax.scatter(y_true_np, y_pred, alpha=0.6, s=30)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('True Value', fontsize=11)
        ax.set_ylabel('Predicted Value', fontsize=11)
        ax.set_title(f'{name}\nR²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表保存到: {save_path}")
    plt.show()


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
) -> Optional[str]:
    """绘制真实值与预测值散点图。"""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError('真实值和预测值的长度必须一致')

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=45, alpha=0.75, color='#1f77b4', label='测试样本')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想预测线')
    ax.set_xlabel('真实值', fontsize=11)
    ax.set_ylabel('预测值', fontsize=11)
    ax.set_title(
        f"PINN 预测效果散点图\nR2={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    plt.show()
    return None


def plot_sparse_observation_kinetics(
    model: torch.nn.Module,
    X_nir: torch.Tensor,
    X_enose: torch.Tensor,
    times_seq: np.ndarray,
    temperatures: torch.Tensor,
    y_true: np.ndarray,
    time_range: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
    verbose: bool = False,
):
    """绘制稀疏标量观测场景下的条件动力学曲线与全样本总览图。"""
    model.eval()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if X_nir.ndim == 1:
        X_nir = X_nir.unsqueeze(0)
    if X_enose.ndim == 1:
        X_enose = X_enose.unsqueeze(0)
    if temperatures.ndim == 1:
        temperatures = temperatures.unsqueeze(1)

    times_seq = np.asarray(times_seq, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    batch_size = X_nir.shape[0]
    if not (len(times_seq) == len(y_true) == batch_size == temperatures.shape[0]):
        raise ValueError('稀疏观测可视化要求每个样本恰好对应一个观测时间点和一个标量真值')

    if time_range is None:
        time_range = (float(times_seq.min()), float(times_seq.max()))
    if time_range[0] == time_range[1]:
        time_range = (time_range[0], time_range[0] + 1.0)

    t_continuous = np.linspace(time_range[0], time_range[1], 100)
    sample_paths: Dict[str, str] = {}
    if save_path:
        from pathlib import Path

        base = Path(save_path)
        sample_paths['samples'] = str(base.with_name(f'{base.stem}_samples{base.suffix}'))
        sample_paths['overview'] = str(base.with_name(f'{base.stem}_overview{base.suffix}'))

    fig, axes = plt.subplots(batch_size, 1, figsize=(7, max(4, 3.5 * batch_size)))
    if batch_size == 1:
        axes = [axes]

    aggregate_curves = []
    with torch.no_grad():
        for sample_idx in range(batch_size):
            ax = axes[sample_idx]

            x_nir_i = X_nir[sample_idx:sample_idx+1].to(device)
            x_enose_i = X_enose[sample_idx:sample_idx+1].to(device)
            temp_i = temperatures[sample_idx:sample_idx+1].to(device)

            y_continuous = []
            for t in t_continuous:
                t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)
                y_pred = model(x_nir_i, x_enose_i, t_tensor, temp_i)
                y_continuous.append(y_pred.item())
            aggregate_curves.append(np.asarray(y_continuous, dtype=np.float32))

            ax.plot(t_continuous, y_continuous, 'r-', lw=2, label='PINN 预测曲线')
            ax.scatter(
                [times_seq[sample_idx]],
                [y_true[sample_idx]],
                s=60,
                alpha=0.8,
                label='真实观测点',
                color='blue',
                zorder=3,
            )
            ax.set_xlabel('时间', fontsize=11)
            ax.set_ylabel('属性值', fontsize=11)
            ax.set_title(f'样本 {sample_idx + 1} 的条件响应曲线', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if 'samples' in sample_paths:
        fig.savefig(sample_paths['samples'], dpi=150)
        if verbose:
            print(f"Per-sample kinetics figure saved to: {sample_paths['samples']}")
    plt.close(fig)

    overview_fig, overview_ax = plt.subplots(1, 1, figsize=(8, 5))
    overview_ax.scatter(times_seq, y_true, s=50, alpha=0.8, color='black', label='全部真实观测点')
    for sample_idx, y_curve in enumerate(aggregate_curves):
        overview_ax.plot(
            t_continuous,
            y_curve,
            lw=1.5,
            alpha=0.7,
            label=f'样本 {sample_idx + 1} 的预测曲线',
        )
    overview_ax.set_xlabel('时间', fontsize=11)
    overview_ax.set_ylabel('属性值', fontsize=11)
    overview_ax.set_title('全部样本的条件响应总览', fontsize=11)
    overview_ax.grid(True, alpha=0.3)
    overview_ax.legend(ncol=2 if batch_size > 3 else 1, fontsize=9)
    plt.tight_layout()
    if 'overview' in sample_paths:
        overview_fig.savefig(sample_paths['overview'], dpi=150)
        if verbose:
            print(f"Aggregate kinetics figure saved to: {sample_paths['overview']}")
    plt.close(overview_fig)

    return sample_paths


def plot_kinetics(
    model: torch.nn.Module,
    X_nir: torch.Tensor,
    X_enose: torch.Tensor,
    times_seq: np.ndarray,
    temperatures: torch.Tensor,
    y_true: np.ndarray,
    time_range: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """兼容旧名称，内部转调稀疏观测场景的动力学可视化函数。"""
    return plot_sparse_observation_kinetics(
        model=model,
        X_nir=X_nir,
        X_enose=X_enose,
        times_seq=times_seq,
        temperatures=temperatures,
        y_true=y_true,
        time_range=time_range,
        save_path=save_path,
        device=device,
    )


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    绘制训练历史曲线

    可视化训练过程中的损失变化，包括：
    - 总损失（训练+验证）
    - 数据损失
    - 物理损失

    Args:
        history (Dict[str, List[float]]): 训练历史字典
            - 'train_loss': 训练总损失历史
            - 'val_loss': 验证损失历史（可选）
            - 'data_loss': 数据损失历史
            - 'physics_loss': 物理损失历史
        save_path (Optional[str]): 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 总损失
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
    
    # 数据损失
    if 'data_loss' in history:
        axes[0, 1].plot(history['data_loss'], label='Data Loss')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # 物理损失
    if 'physics_loss' in history:
        axes[1, 0].plot(history['physics_loss'], label='Physics Loss')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线保存到: {save_path}")
    plt.show()


def generate_evaluation_report(
    results: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
) -> str:
    """
    生成格式化的模型评估报告

    创建包含所有评估指标的文本报告，便于结果分析和比较。

    Args:
        results (Dict[str, Dict[str, float]]): 评估结果字典
        model_names (Optional[List[str]]): 模型名称列表（用于排序）

    Returns:
        str: 格式化的评估报告字符串

    Example:
        results = {'PINN': {'r2': 0.95, 'rmse': 0.12, ...}, 'PLS': {...}}
        report = generate_evaluation_report(results)
        print(report)
    """
    report = "\n" + "=" * 70 + "\n"
    report += "PINN 模型评估报告\n"
    report += "=" * 70 + "\n\n"
    
    # 表头
    report += f"{'Model':<20} {'R²':<12} {'RMSE':<12} {'MAE':<12} {'RPD':<10}\n"
    report += "-" * 70 + "\n"
    
    for name, metrics in results.items():
        report += f"{name:<20} "
        report += f"{metrics['r2']:<12.4f} "
        report += f"{metrics['rmse']:<12.4f} "
        report += f"{metrics['mae']:<12.4f} "
        report += f"{metrics['rpd']:<10.2f}\n"
    
    report += "=" * 70 + "\n"
    report += "\n指标说明:\n"
    report += "  R² (决定系数): 越接近1越好 (目标: > 0.9)\n"
    report += "  RMSE (均方根误差): 越小越好\n"
    report += "  RPD (相对分析误差): > 3 优秀, > 2 良好\n"
    report += "\n"
    
    return report
