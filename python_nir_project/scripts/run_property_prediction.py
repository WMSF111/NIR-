from __future__ import annotations

"""统一入口脚本，集中配置项目中影响结果的参数。

该脚本用于集中管理传统机器学习与PINN模型的参数，
方便在一个位置进行调整和复现实验。
"""

import json
import subprocess
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from nir_project.cli import main as cli_main, run_prediction
except ModuleNotFoundError as exc:
    if exc.name == 'nir_project':
        project_package_root = Path(__file__).resolve().parents[1]
        repo_root = project_package_root.parent
        if str(project_package_root) not in sys.path:
            sys.path.insert(0, str(project_package_root))
        try:
            from nir_project.cli import main as cli_main, run_prediction
        except ModuleNotFoundError:
            venv_python = repo_root / '.venv' / 'Scripts' / 'python.exe'
            current_python = Path(sys.executable).resolve() if sys.executable else None
            if venv_python.exists() and current_python != venv_python.resolve():
                raise SystemExit(
                    subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
                ) from exc
            raise SystemExit(
                'Current interpreter cannot import nir_project. '
                'Use the project virtual environment first, or run '
                '`python -m pip install -e .` from the repo root.'
            ) from exc
    raise


RUN_CONFIG = {
    'property_name': 'a*',
    'model_type': 'pinn',      # traditional / pinn
    'mode': 'compare',         # train / compare / select
    'preproc_mode': 'sg+msc+snv',
    'sg_order': 3,
    'sg_window': 5,
    'fs_method': 'all',
    'fs_param': None,
    'include_preprocessed_group': False,
}


TRADITIONAL_CONFIG = {
    'train_regressor': 'pls',
    'train_regressor_param': {'max_lv': 300, 'cv_fold': 10},
    'regressors': ['pls', 'pcr', 'svr', 'rf', 'gpr', 'knn', 'cnn', 'xgb'],
    'regressor_params': {
        'pls': {'max_lv': 300, 'cv_fold': 10},
        'pcr': {'max_pc': 300},
        'svr': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
        'rf': {'n_estimators': [100, 200], 'min_samples_leaf': [1, 5, 10]},
        'gpr': {'kernel': ['squaredexponential', 'matern32', 'matern52']},
        'knn': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'cnn': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'activation': ['relu'], 'max_iter': [1000]},
        'xgb': {'n_estimators': [100, 200], 'max_depth': [3, 6], 'learning_rate': [0.1, 0.3]},
    },
    'fs_param_grid': {
        'cars': [120, 150],
        'pca': [40, 80, 120, 160, 200],
        'corr_topk': [20, 40, 80, 120],
        'spa': [8, 12, 16, 20],
    },
    'dataset_options': {
        'msc_ref_mode': 'mean',
        'snv_mode': 'standard',
        'baseline_zero_mode': 'none',
        'baseline_zero_scope': 'cropped_spectrum',
        'despike_mode': 'none',
        'keep_exports': False,
    },
}


PINN_CONFIG = {
    'input_mode': 'nir',   # nir / enose / fusion
    'fs_param_grid': {
        'cars': [10, 20, 30, 40],
        'pca': [10, 20, 30, 40],
        'corr_topk': [10, 20, 30, 40],
        'spa': [8, 12, 16, 20],
    },
    'dataset_options': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'augment': True,
        'mixup_alpha': 0.2,
        'mixup_samples': 100,
        'random_state': 1,
        'require_times': True,
        'require_temperatures': True,
        'require_enose': False,
        'context_size': 2,
        'horizon': 1,
    },
    'model_params': {
        'nir_hidden': (128, 64),
        'nose_hidden': (32, 16),
        'shared_hidden': (128, 64, 32),
        'physics_dim': 2,
        'output_dim': 1,
        'activation': 'relu',
        'dropout': 0.0,
        'output_activaton': None,
    },
    'loss_params': {
        'E_a': 100000.0,
        'k0': 1e-3,
        'R': 8.314,
        'w_data': 1.0,
        'w_physics': 0.001,
        'kinetics_order': 1,
    },
    'train_params': {
        'stage1_ratio': 0.67,
        'stage1_epochs': 2000,
        'stage2_epochs': 1000,
        'stage1_lr': 0.001,
        'stage2_lr': 1.0,
    },
}

PRIOR_MODEL_CONFIG = {
    # 挥发平衡模型参数（已调整以显示合理的动力学行为）
    'G0': 0.1,                  # 初始气相浓度
    'k_vol_0': 0.1,             # 基础挥发速率常数（温度基准）降低E_a可获得更快速率
    'E_a_vol': 10000.0,         # 挥发活化能 (J/mol) 调低使温度敏感性合理
    'G_sat_ref': 2.0,           # 参考温度下的饱和浓度（调整以显示平衡效果）
    'T_ref': 298.15,            # 参考温度 (K)
    'beta': 0.01,               # 饱和浓度温度系数 (1/K)
    'R': 8.314,                 # 气体常数 (J/(mol·K))
    'temperature_K': 298.15,    # 实验温度
    't_max': 100.0,             # 时间上限
    'num_points': 200,          # 绘图采样点数
}


def compute_arrhenius_prior_curve(
    times: np.ndarray,
    temperatures: np.ndarray,
    G0: float = PRIOR_MODEL_CONFIG['G0'],
    k_vol_0: float = PRIOR_MODEL_CONFIG['k_vol_0'],
    E_a_vol: float = PRIOR_MODEL_CONFIG['E_a_vol'],
    G_sat_ref: float = PRIOR_MODEL_CONFIG['G_sat_ref'],
    T_ref: float = PRIOR_MODEL_CONFIG['T_ref'],
    beta: float = PRIOR_MODEL_CONFIG['beta'],
    R: float = PRIOR_MODEL_CONFIG['R'],
) -> np.ndarray:
    """基于挥发平衡模型计算气相浓度演化。
    
    模型方程：dG/dt = k_vol(T) * (1 - G/G_sat(T))
    
    其中：
    - k_vol(T) = k_vol_0 * exp(-E_a_vol/(R*T))
    - G_sat(T) = G_sat_ref * exp(beta * (T - T_ref))
    """
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    temperatures = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    
    if temperatures.shape[0] == 1:
        temperatures = np.full_like(times, temperatures.item())
    elif temperatures.shape[0] != times.shape[0]:
        raise ValueError(f"Temperature array length {len(temperatures)} != time array length {len(times)}")
    
    # 生成插值函数用于获取任意时刻的温度
    from scipy.interpolate import interp1d
    T_interp = interp1d(times, temperatures, kind='linear', fill_value='extrapolate')
    
    # 定义ODE
    def volatilization_ode(G, t):
        """气相浓度微分方程"""
        try:
            T = float(T_interp(t))
        except:
            T = temperatures[-1]
        k_vol = k_vol_0 * np.exp(-E_a_vol / (R * (T + 1e-12)))
        G_sat = G_sat_ref * np.exp(beta * (T - T_ref))
        dGdt = k_vol * (1.0 - G / (G_sat + 1e-12))
        return dGdt
    
    # 数值求解ODE
    G_solution = odeint(volatilization_ode, G0, times, full_output=False)
    
    return G_solution.flatten()


def plot_prior_model(
    times: np.ndarray,
    temperatures: np.ndarray,
    y_prior: np.ndarray,
    save_path: Path,
    y_true: np.ndarray | None = None,
    title: str = 'Prior Model Comparison',
) -> Path:
    """绘制先验模型曲线，并可选地与真实标签对比。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, y_prior, label='Prior model', color='#1f77b4', linewidth=2)
    if y_true is not None:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        ax.scatter(times, y_true, label='True value', color='#ff7f0e', s=30, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Prediction')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_prior_model_from_config(pinn_config: dict, save_path: Path, temperature_K: float | None = None, data_dir: Path | None = None, top_n: int = 3) -> Path:
    """根据配置绘制先验挥发平衡模型，并与实验数据对比。
    
    从PINN_CONFIG中的loss_params提取E_a_vol, k_vol_0等参数，
    计算并绘制气相浓度随时间的演化曲线，同时加载并对比相同温度下的样本数据。
    
    参数：
        pinn_config: PINN配置字典
        save_path: 输出图片路径
        temperature_K: 指定对比温度（默认使用配置中的temperature_K）
        data_dir: 数据目录路径（默认搜索 matlab/data/NIR/ 或 python_nir_project/data/NIR/）
        top_n: 显示最接近先验模型的N个样本
    """
    prior = dict(PRIOR_MODEL_CONFIG)
    if pinn_config is not None:
        loss_params = pinn_config.get('loss_params', {})
        prior.update({
            'E_a_vol': loss_params.get('E_a', prior.get('E_a_vol', 50000.0)),
            'k_vol_0': loss_params.get('k0', prior.get('k_vol_0', 1e-3)),
            'R': loss_params.get('R', prior.get('R', 8.314)),
        })
    
    # 确定对比温度
    if temperature_K is None:
        temperature_K = prior['temperature_K']
    prior['temperature_K'] = temperature_K
    
    # 计算先验模型曲线
    times = np.linspace(0, prior['t_max'], int(prior['num_points']), dtype=np.float64)
    temperatures = np.full_like(times, temperature_K)
    
    y_prior = compute_arrhenius_prior_curve(
        times,
        temperatures,
        G0=prior['G0'],
        k_vol_0=prior['k_vol_0'],
        E_a_vol=prior['E_a_vol'],
        G_sat_ref=prior['G_sat_ref'],
        T_ref=prior['T_ref'],
        beta=prior['beta'],
        R=prior['R'],
    )
    
    G_sat = prior['G_sat_ref'] * np.exp(prior['beta'] * (temperature_K - prior['T_ref']))
    
    # 尝试加载实验数据
    sample_data = []
    sample_times = []
    if data_dir is None:
        # 搜索默认数据目录
        possible_dirs = [
            Path(__file__).parent.parent / 'data' / 'NIR',
            Path(__file__).parent.parent.parent / 'matlab' / 'data' / 'NIR',
        ]
        for d in possible_dirs:
            if d.exists():
                data_dir = d
                break
    
    if data_dir and data_dir.exists():
        # 提取指定温度的所有样本
        temp_int = int(temperature_K)
        pattern = f"{temp_int}_*.csv"
        sample_files = sorted(data_dir.glob(pattern))
        
        for file in sample_files[:50]:  # 限制最多50个样本避免图表过拥挤
            try:
                # 从文件名提取信息: T_sample_time.csv
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    sample_id = int(parts[1])
                    time_idx = int(parts[2])
                    
                    # 读取CSV数据（假设最后一列是标签或强度）
                    df = pd.read_csv(file, header=None)
                    if df.shape[1] > 0:
                        # 获取最后一列作为气相浓度代理
                        value = df.iloc[-1, -1]
                        sample_data.append(float(value))
                        sample_times.append(float(time_idx))
            except:
                pass
        
        # 按时间排序
        if sample_times:
            sorted_idx = np.argsort(sample_times)
            sample_times = np.array(sample_times)[sorted_idx]
            sample_data = np.array(sample_data)[sorted_idx]
            
            # 标准化到0-1范围以便对比
            if sample_data.max() > sample_data.min():
                sample_data_norm = (sample_data - sample_data.min()) / (sample_data.max() - sample_data.min()) * y_prior.max()
            else:
                sample_data_norm = sample_data
    else:
        sample_times = None
        sample_data_norm = None
    
    # 计算样本与模型的相似度（欧式距离）
    similar_samples = []
    if sample_times is not None and len(sample_times) > 0:
        # 对齐时间轴进行差值插值
        y_prior_interp_func = np.interp(sample_times, times, y_prior, left=y_prior[0], right=y_prior[-1])
        
        for i, (t, y) in enumerate(zip(sample_times, sample_data_norm)):
            if not np.isnan(y):
                dist = abs(y - y_prior_interp_func[i])
                similar_samples.append((dist, i, t, y))
        
        # 按相似度排序（距离小的在前）
        similar_samples.sort(key=lambda x: x[0])
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ========== 左图：气相浓度演化 ==========
    # 先验模型
    ax1.plot(times, y_prior, label='Prior model (Volatilization Eq.)', color='#1f77b4', linewidth=2.5, zorder=10)
    ax1.axhline(y=G_sat, color='#d62728', linestyle='--', linewidth=1.5, label=f'G_sat(T={temperature_K:.1f}K)={G_sat:.4f}', zorder=5)
    ax1.fill_between(times, 0, y_prior, alpha=0.15, color='#1f77b4', zorder=1)
    
    # 叠加实验样本
    if sample_times is not None and len(sample_times) > 0:
        # 显示所有样本（灰色）
        ax1.scatter(sample_times, sample_data_norm, alpha=0.3, s=50, color='gray', label='Sample data (all)', zorder=3)
        
        # 高亮显示最接近的top_n个样本
        colors = ['#ff7f0e', '#2ca02c', '#d62728']  # 橙、绿、红
        for rank, (dist, idx, t, y) in enumerate(similar_samples[:top_n]):
            color = colors[min(rank, len(colors)-1)]
            ax1.scatter([t], [y], s=120, color=color, marker='*', edgecolors='black', linewidth=1.5, 
                       label=f'Rank {rank+1} (dist={dist:.3f})', zorder=8-rank)
    
    ax1.set_xlabel('Time (a.u.)', fontsize=11)
    ax1.set_ylabel('Gas concentration (a.u.)', fontsize=11)
    ax1.set_title(f'Prior Model vs Sample Data (T={temperature_K:.1f}K)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim([0, max(y_prior.max(), G_sat * 1.1) if sample_data_norm is None else max(y_prior.max(), sample_data_norm.max() * 1.1)])
    
    # ========== 右图：挥发速率 ==========
    k_vol = prior['k_vol_0'] * np.exp(-prior['E_a_vol'] / (prior['R'] * (temperature_K + 1e-12)))
    volatilization_rate = k_vol * (1.0 - y_prior / (G_sat + 1e-12))
    ax2.plot(times, volatilization_rate, label='Volatilization rate dG/dt', color='#2ca02c', linewidth=2.5)
    ax2.fill_between(times, 0, volatilization_rate, alpha=0.2, color='#2ca02c')
    
    # 在样本位置标注速率
    if sample_times is not None and len(sample_times) > 0:
        rate_at_samples = k_vol * (1.0 - np.interp(sample_times, times, y_prior, left=y_prior[0], right=y_prior[-1]) / (G_sat + 1e-12))
        ax2.scatter(sample_times, rate_at_samples, alpha=0.4, s=40, color='gray', label='Sample time points', zorder=3)
        
        # 高亮最接近的样本
        for rank, (dist, idx, t, y) in enumerate(similar_samples[:top_n]):
            color = colors[min(rank, len(colors)-1)]
            rate_at_t = k_vol * (1.0 - y / (G_sat + 1e-12))
            ax2.scatter([t], [rate_at_t], s=100, color=color, marker='*', edgecolors='black', linewidth=1.5, zorder=8-rank)
    
    ax2.set_xlabel('Time (a.u.)', fontsize=11)
    ax2.set_ylabel('Rate dG/dt', fontsize=11)
    ax2.set_title(f'Volatilization Rate Evolution (T={temperature_K:.1f}K)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(loc='upper right', fontsize=9)
    
    # 添加统计信息文本框
    if sample_times is not None and len(similar_samples) > 0:
        stats_text = f"Total samples: {len(sample_times)}\nTop {top_n} closest samples shown"
        ax2.text(0.02, 0.97, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 打印最接近的样本信息
    print(f"\n📊 先验模型对比结果 (T={temperature_K:.1f}K)")
    print(f"   数据目录: {data_dir}")
    print(f"   加载样本数: {len(sample_times) if sample_times is not None else 0}")
    if similar_samples:
        print(f"\n   最接近的{min(top_n, len(similar_samples))}个样本:")
        for rank, (dist, idx, t, y) in enumerate(similar_samples[:top_n]):
            print(f"   #{rank+1}: 时间={t:.1f}, 浓度={y:.4f}, 距离={dist:.4f}")
    
    return save_path



def run_from_config():
    """从当前文件中的配置直接运行。

    该函数读取顶层的RUN_CONFIG、TRADITIONAL_CONFIG和PINN_CONFIG，
    并将其传递给顶层运行入口以执行模型训练或对比。
    """
    prior_plot_path = Path('result') / 'pinn' / 'prior_model.png'
    plot_prior_model_from_config(PINN_CONFIG, prior_plot_path)
    print(f"先验模型图已保存: {prior_plot_path}")

    # result = run_prediction(
    #     property_name=RUN_CONFIG['property_name'],
    #     model_type=RUN_CONFIG['model_type'],
    #     mode=RUN_CONFIG['mode'],
    #     preproc_mode=RUN_CONFIG['preproc_mode'],
    #     sg_order=RUN_CONFIG['sg_order'],
    #     sg_window=RUN_CONFIG['sg_window'],
    #     fs_method=RUN_CONFIG['fs_method'],
    #     fs_param=RUN_CONFIG['fs_param'],
    #     include_preprocessed_group=RUN_CONFIG['include_preprocessed_group'],
    #     traditional_config=TRADITIONAL_CONFIG,
    #     pinn_config=PINN_CONFIG,
    # )
    # pprint(result)
    # return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main()
    else:
        run_from_config()
