"""
PINN评估与可视化模块

该模块提供了PINN（物理信息神经网络）模型的全面评估和可视化功能，
包括性能指标计算、模型对比、动力学曲线绘制和训练历史分析。

主要功能：
- 模型性能评估：R²、RMSE、MAE、RPD等指标计算
- 多模型对比：同时评估多个模型的性能
- 可视化分析：预测vs真实值散点图、动力学曲线、训练损失曲线
- 评估报告生成：格式化的性能报告输出

评估指标：
- R²（决定系数）：衡量模型拟合优度，越接近1越好
- RMSE（均方根误差）：预测误差的平方根，越小越好
- MAE（平均绝对误差）：预测误差的绝对值平均，越小越好
- RPD（相对分析误差）：标准差与RMSE的比值，>3为优秀
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


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
    
    # 计算指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 相对分析误差 RPD
    std_true = np.std(y_true)
    rpd = std_true / (rmse + 1e-10)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'rpd': rpd,
    }
    
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
    """
    绘制化学动力学曲线

    展示质量随时间的变化曲线，包括：
    - 真实数据点（散点）
    - PINN预测的连续曲线
    - 验证物理约束的合理性

    Args:
        model (torch.nn.Module): PINN模型
        X_nir (torch.Tensor): NIR特征数据，形状为[batch_size, nir_features]
        X_enose (torch.Tensor): 电子鼻特征数据，形状为[batch_size, enose_features]
        times_seq (np.ndarray): 时间序列数据点，用于绘制真实数据
        temperatures (torch.Tensor): 温度序列，形状为[batch_size, 1]
        y_true (np.ndarray): 真实质量值
        time_range (Optional[Tuple[float, float]]): 绘制的时间范围，默认使用数据范围
        save_path (Optional[str]): 保存路径
        device (str): 计算设备
    """
    model.eval()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保输入是2D
    if X_nir.ndim == 1:
        X_nir = X_nir.unsqueeze(0)
    if X_enose.ndim == 1:
        X_enose = X_enose.unsqueeze(0)
    
    batch_size = X_nir.shape[0]
    
    # 生成连续时间序列
    if time_range is None:
        time_range = (times_seq.min(), times_seq.max())
    
    t_continuous = np.linspace(time_range[0], time_range[1], 100)
    
    fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 4))
    if batch_size == 1:
        axes = [axes]
    
    with torch.no_grad():
        for sample_idx in range(batch_size):
            ax = axes[sample_idx]
            
            # 单个样本
            x_nir_i = X_nir[sample_idx:sample_idx+1].to(device)
            x_enose_i = X_enose[sample_idx:sample_idx+1].to(device)
            t_i = temperatures[sample_idx:sample_idx+1].to(device)
            
            # 真实数据点
            ax.scatter(times_seq, y_true, s=50, alpha=0.7, label='True Data', color='blue')
            
            # 连续预测曲线
            y_continuous = []
            for t in t_continuous:
                t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)
                y_pred = model(x_nir_i, x_enose_i, t_tensor, t_i)
                y_continuous.append(y_pred.item())
            
            ax.plot(t_continuous, y_continuous, 'r-', lw=2, label='PINN Prediction')
            
            ax.set_xlabel('Time (h)', fontsize=11)
            ax.set_ylabel('Quality', fontsize=11)
            ax.set_title(f'Sample {sample_idx + 1} - Kinetics Curve', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"动力学曲线保存到: {save_path}")
    plt.show()


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
