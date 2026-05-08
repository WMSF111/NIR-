from __future__ import annotations

"""PINN 数据准备模块，负责预处理、增强、切分与项目数据适配。"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..data import PROJECT_ROOT, black_white_processing, build_property_vector_from_all_csv
from ..feature_selection import select_features_by_method
from ..preprocessing import parse_preproc_mode


def preprocess_nir(
    nir_spectrum: np.ndarray,
    snv: bool = True,
    sg_window: int = 15,
    sg_order: int = 3,
    normalize: bool = True,
    apply_sg: bool = True,
) -> np.ndarray:
    """对 NIR 光谱执行 PINN 训练所需的预处理。"""
    spectrum = nir_spectrum.copy().astype(np.float32)

    if snv:
        mean = spectrum.mean(axis=1, keepdims=True)
        std = spectrum.std(axis=1, keepdims=True)
        spectrum = (spectrum - mean) / (std + 1e-8)

    if apply_sg:
        from scipy.signal import savgol_filter

        feature_count = spectrum.shape[1]
        valid_window = min(sg_window, feature_count)
        if valid_window % 2 == 0:
            valid_window -= 1
        min_window = sg_order + 2 if (sg_order + 2) % 2 == 1 else sg_order + 3
        if valid_window >= min_window:
            spectrum = savgol_filter(spectrum, valid_window, sg_order, axis=1)

    if normalize:
        spec_min = spectrum.min(axis=1, keepdims=True)
        spec_max = spectrum.max(axis=1, keepdims=True)
        spectrum = (spectrum - spec_min) / (spec_max - spec_min + 1e-8)

    return spectrum.astype(np.float32)


def preprocess_enose(sensor_data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """对电子鼻特征执行 PINN 训练所需的预处理。"""
    if sensor_data.ndim == 3:
        r_max = sensor_data.max(axis=2)
        r_mean = sensor_data.mean(axis=2)
        r_std = sensor_data.std(axis=2)
        r_cv = r_std / (r_mean + 1e-8)
        features = np.concatenate([r_max, r_mean, r_cv], axis=1).astype(np.float32)
    else:
        features = sensor_data.astype(np.float32)

    if normalize:
        feat_min = features.min(axis=0, keepdims=True)
        feat_max = features.max(axis=0, keepdims=True)
        features = (features - feat_min) / (feat_max - feat_min + 1e-8)

    return features.astype(np.float32)


def mixup_augmentation(
    X1: np.ndarray,
    X2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    alpha: float = 0.2,
    n_samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 Mixup 增强样本，并返回对应的插值权重。"""
    lambdas = np.random.beta(alpha, alpha, n_samples)

    X_new = []
    y_new = []
    for lam in lambdas:
        X_new.append(lam * X1 + (1 - lam) * X2)
        y_new.append(lam * y1 + (1 - lam) * y2)

    return np.array(X_new), np.array(y_new), lambdas


def delta_augmentation(
    X1: np.ndarray,
    X2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造基于差分的增强样本。"""
    delta_X = X2 - X1
    delta_y = y2 - y1
    delta_t = times[1] - times[0]
    rate_y = delta_y / delta_t if delta_t > 0 else delta_y
    avg_X = (X1 + X2) / 2.0
    return delta_X, rate_y, delta_t, avg_X


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """将 NumPy 数组转换为 `float32` 的 Torch 张量。"""
    return torch.from_numpy(arr).float()


def _extract_aligned_metadata_rows(file_names: list[str]) -> pd.DataFrame:
    """按文件名顺序从 `all_csv_data.csv` 中对齐并提取元数据行。"""
    all_csv_path = PROJECT_ROOT / 'data' / 'physical' / 'all_csv_data.csv'
    df = pd.read_csv(all_csv_path)
    if 'csv_name' not in df.columns:
        raise ValueError('all_csv_data.csv must contain column csv_name')
    return df.set_index('csv_name').loc[file_names].reset_index()


def _parse_filename_experiment_tokens(file_names: list[str]) -> pd.DataFrame:
    """从文件名如 `70_0_1.csv` 解析温度、样本号和时间。"""
    parsed_rows = []
    for file_name in file_names:
        stem = Path(file_name).stem
        parts = stem.split('_')
        if len(parts) < 3:
            raise ValueError(
                'PINN dataset could not infer experiment metadata from file names. '
                'Expected pattern like 70_0_1.csv.'
            )
        try:
            temperature = float(parts[0])
            sample_id = float(parts[1])
            time_value = float(parts[2])
        except ValueError as exc:
            raise ValueError(
                f'PINN dataset could not parse temperature/sample/time from file name: {file_name}'
            ) from exc
        parsed_rows.append(
            {
                'csv_name': file_name,
                'filename_temperature': temperature,
                'filename_sample_id': sample_id,
                'filename_time': time_value,
                'filename_trajectory_id': f'{int(temperature)}_{int(sample_id)}',
            }
        )
    return pd.DataFrame(parsed_rows)


def _extract_numeric_series(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    field_name: str,
    required: bool,
) -> np.ndarray:
    """按候选列名提取数值序列，并在需要时强制要求字段存在。"""
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            series = pd.to_numeric(df[lower_to_original[candidate]], errors='coerce').to_numpy(dtype=np.float32)
            finite = np.isfinite(series)
            if finite.any():
                fill_value = float(np.nanmean(series[finite]))
                return np.nan_to_num(series, nan=fill_value).astype(np.float32)
            if required:
                raise ValueError(f'PINN dataset requires at least one numeric value for {field_name}')
    if required:
        raise ValueError(f'PINN dataset requires column(s) for {field_name}: {", ".join(candidates)}')
    return np.zeros(len(df), dtype=np.float32)


def _extract_enose_features(df: pd.DataFrame, sample_count: int, *, required: bool) -> np.ndarray:
    """从元数据表中提取电子鼻相关特征列。"""
    ignore = {'csv_name', 'time', 'times', 'hour', 'hours', 'day', 'days', 'temperature', 'temp', 't'}
    sensor_columns = [
        col
        for col in df.columns
        if any(token in col.lower() for token in ('enose', 'nose', 'sensor', 'gas'))
        and col.lower() not in ignore
    ]
    if not sensor_columns:
        if required:
            raise ValueError('PINN dataset requires at least one e-nose feature column in all_csv_data.csv')
        return np.zeros((sample_count, 1), dtype=np.float32)
    return (
        df[sensor_columns]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _split_indices_by_group(
    group_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    random_state: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按轨迹分组切分，避免同一轨迹同时落入训练/验证/测试集。"""
    unique_groups = np.unique(group_ids.astype(str))
    if unique_groups.size < 3:
        raise ValueError('PINN grouped splitting requires at least 3 trajectories')

    rng = np.random.default_rng(random_state)
    shuffled_groups = rng.permutation(unique_groups)

    n_groups = shuffled_groups.size
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))
    n_test = n_groups - n_train - n_val
    if n_test < 1:
        deficit = 1 - n_test
        reducible_val = max(0, n_val - 1)
        reduce_from_val = min(deficit, reducible_val)
        n_val -= reduce_from_val
        deficit -= reduce_from_val
        if deficit > 0:
            n_train = max(1, n_train - deficit)
        n_test = n_groups - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError('Unable to create non-empty grouped train/validation/test splits')

    train_groups = set(shuffled_groups[:n_train].tolist())
    val_groups = set(shuffled_groups[n_train:n_train + n_val].tolist())
    test_groups = set(shuffled_groups[n_train + n_val:].tolist())

    train_idx = np.array([idx for idx, gid in enumerate(group_ids.astype(str)) if gid in train_groups], dtype=int)
    val_idx = np.array([idx for idx, gid in enumerate(group_ids.astype(str)) if gid in val_groups], dtype=int)
    test_idx = np.array([idx for idx, gid in enumerate(group_ids.astype(str)) if gid in test_groups], dtype=int)

    if min(train_idx.size, val_idx.size, test_idx.size) < 1:
        raise ValueError('Grouped split produced an empty subset; please provide more trajectory data')
    return train_idx, val_idx, test_idx


def _build_same_trajectory_pairs(train_idx: np.ndarray, group_ids: np.ndarray, times: np.ndarray) -> list[tuple[int, int]]:
    """构造同一轨迹内部、按时间相邻的样本对，用于 PINN 增强。"""
    train_groups = group_ids[train_idx].astype(str)
    train_times = times[train_idx]
    pairs: list[tuple[int, int]] = []

    for group_name in np.unique(train_groups):
        local_positions = np.where(train_groups == group_name)[0]
        if local_positions.size < 2:
            continue
        ordered = local_positions[np.argsort(train_times[local_positions])]
        ordered_global = train_idx[ordered]
        for pos in range(len(ordered_global) - 1):
            pairs.append((int(ordered_global[pos]), int(ordered_global[pos + 1])))

    return pairs


def _build_sliding_forecast_windows(
    X_nir: np.ndarray,
    X_enose: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    temperatures: np.ndarray,
    trajectory_ids: np.ndarray,
    *,
    context_size: int = 2,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按轨迹构造滑窗样本，用前若干点预测后一个点。"""
    if context_size < 1:
        raise ValueError('context_size must be at least 1')
    if horizon < 1:
        raise ValueError('horizon must be at least 1')

    windowed_nir = []
    windowed_enose = []
    windowed_y = []
    windowed_times = []
    windowed_temperatures = []
    windowed_groups = []

    for trajectory_name in np.unique(trajectory_ids.astype(str)):
        trajectory_idx = np.where(trajectory_ids.astype(str) == trajectory_name)[0]
        if trajectory_idx.size < context_size + horizon:
            continue
        ordered_idx = trajectory_idx[np.argsort(times[trajectory_idx])]
        max_start = ordered_idx.size - context_size - horizon + 1
        for start in range(max_start):
            context_idx = ordered_idx[start:start + context_size]
            target_idx = ordered_idx[start + context_size + horizon - 1]
            windowed_nir.append(np.concatenate([X_nir[idx] for idx in context_idx], axis=0))
            windowed_enose.append(np.concatenate([X_enose[idx] for idx in context_idx], axis=0))
            windowed_y.append(y[target_idx])
            windowed_times.append(times[target_idx])
            windowed_temperatures.append(temperatures[target_idx])
            windowed_groups.append(trajectory_name)

    if not windowed_y:
        raise ValueError('No sliding PINN windows could be constructed from the available trajectories')

    return (
        np.asarray(windowed_nir, dtype=np.float32),
        np.asarray(windowed_enose, dtype=np.float32),
        np.asarray(windowed_y, dtype=np.float32),
        np.asarray(windowed_times, dtype=np.float32),
        np.asarray(windowed_temperatures, dtype=np.float32),
        np.asarray(windowed_groups, dtype=str),
    )


def prepare_pinn_dataset(
    X_nir: np.ndarray,
    X_enose: Optional[np.ndarray],
    y: np.ndarray,
    times: np.ndarray,
    temperatures: np.ndarray,
    group_ids: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    augment: bool = True,
    mixup_alpha: float = 0.2,
    mixup_samples: int = 100,
    nir_snv: bool = True,
    nir_apply_sg: bool = True,
    nir_sg_window: int = 15,
    nir_sg_order: int = 3,
    nir_normalize: bool = True,
    enose_normalize: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """为 PINN 训练准备训练集、验证集、测试集和配点集。"""
    n_samples = len(y)
    if n_samples < 3:
        raise ValueError('PINN training requires at least 3 samples to create train/validation/test splits')
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between 0 and 1')
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    if train_ratio + val_ratio >= 1:
        raise ValueError('train_ratio + val_ratio must be less than 1 so a test split remains')

    rng = np.random.default_rng(random_state)
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.shape[0] != n_samples:
            raise ValueError('group_ids length must match the number of PINN samples')
        train_idx, val_idx, test_idx = _split_indices_by_group(
            group_ids=group_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_state=random_state,
        )
    else:
        idx = rng.permutation(n_samples)

        n_train = max(1, int(round(n_samples * train_ratio)))
        n_val = max(1, int(round(n_samples * val_ratio)))
        n_test = n_samples - n_train - n_val
        if n_test < 1:
            deficit = 1 - n_test
            reducible_val = max(0, n_val - 1)
            reduce_from_val = min(deficit, reducible_val)
            n_val -= reduce_from_val
            deficit -= reduce_from_val
            if deficit > 0:
                n_train = max(1, n_train - deficit)
            n_test = n_samples - n_train - n_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError('Unable to create non-empty train/validation/test splits from the provided samples')

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

    X_nir_proc = preprocess_nir(
        X_nir,
        snv=nir_snv,
        sg_window=nir_sg_window,
        sg_order=nir_sg_order,
        normalize=nir_normalize,
        apply_sg=nir_apply_sg,
    )
    X_enose_proc = (
        preprocess_enose(X_enose, normalize=enose_normalize)
        if X_enose is not None
        else np.zeros((n_samples, 1), dtype=np.float32)
    )

    train_data = {
        'X_nir': _to_tensor(X_nir_proc[train_idx]),
        'X_enose': _to_tensor(X_enose_proc[train_idx]),
        'times': _to_tensor(times[train_idx].reshape(-1, 1).astype(np.float32)),
        'temperatures': _to_tensor(temperatures[train_idx].reshape(-1, 1).astype(np.float32)),
        'y': _to_tensor(y[train_idx].astype(np.float32)),
    }
    val_data = {
        'X_nir': _to_tensor(X_nir_proc[val_idx]),
        'X_enose': _to_tensor(X_enose_proc[val_idx]),
        'times': _to_tensor(times[val_idx].reshape(-1, 1).astype(np.float32)),
        'temperatures': _to_tensor(temperatures[val_idx].reshape(-1, 1).astype(np.float32)),
        'y': _to_tensor(y[val_idx].astype(np.float32)),
    }
    test_data = {
        'X_nir': _to_tensor(X_nir_proc[test_idx]),
        'X_enose': _to_tensor(X_enose_proc[test_idx]),
        'times': _to_tensor(times[test_idx].reshape(-1, 1).astype(np.float32)),
        'temperatures': _to_tensor(temperatures[test_idx].reshape(-1, 1).astype(np.float32)),
        'y': _to_tensor(y[test_idx].astype(np.float32)),
    }

    collocation_data = {
        key: value.clone() for key, value in train_data.items()
    }

    trajectory_pairs = (
        _build_same_trajectory_pairs(train_idx, group_ids, times)
        if augment and group_ids is not None
        else []
    )

    if augment and len(train_idx) > 1:
        if group_ids is not None:
            candidate_pairs = trajectory_pairs
        else:
            candidate_pairs = []
            for _ in range(3):
                i, j = rng.choice(len(train_idx), 2, replace=False)
                candidate_pairs.append((int(train_idx[i]), int(train_idx[j])))

        for idx_i, idx_j in candidate_pairs:
            X_nir_aug, y_aug, lambdas = mixup_augmentation(
                X_nir_proc[idx_i],
                X_nir_proc[idx_j],
                y[idx_i],
                y[idx_j],
                alpha=mixup_alpha,
                n_samples=mixup_samples,
            )
            X_enose_aug = (
                lambdas.reshape(-1, 1) * X_enose_proc[idx_i]
                + (1 - lambdas).reshape(-1, 1) * X_enose_proc[idx_j]
            ).astype(np.float32)
            t_low, t_high = sorted((float(times[idx_i]), float(times[idx_j])))
            temp_low, temp_high = sorted(
                (float(temperatures[idx_i]), float(temperatures[idx_j]))
            )
            t_aug = rng.uniform(t_low, t_high, mixup_samples).astype(np.float32)
            T_aug = rng.uniform(temp_low, temp_high, mixup_samples).astype(np.float32)

            collocation_data['X_nir'] = torch.cat([collocation_data['X_nir'], _to_tensor(X_nir_aug)], dim=0)
            collocation_data['X_enose'] = torch.cat([collocation_data['X_enose'], _to_tensor(X_enose_aug)], dim=0)
            collocation_data['y'] = torch.cat([collocation_data['y'], _to_tensor(y_aug.astype(np.float32))], dim=0)
            collocation_data['times'] = torch.cat(
                [collocation_data['times'], _to_tensor(t_aug.reshape(-1, 1))],
                dim=0,
            )
            collocation_data['temperatures'] = torch.cat(
                [collocation_data['temperatures'], _to_tensor(T_aug.reshape(-1, 1))],
                dim=0,
            )

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'collocation': collocation_data,
    }


def prepare_pinn_dataset_from_project_data(
    property_name: str,
    preproc_mode: str = 'sg+msc',
    sg_order: int = 3,
    sg_window: int = 15,
    fs_method: Optional[str] = None,
    fs_param: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    augment: bool = True,
    mixup_alpha: float = 0.2,
    mixup_samples: int = 100,
    random_state: Optional[int] = 1,
    require_times: bool = True,
    require_temperatures: bool = True,
    require_enose: bool = False,
    context_size: int = 2,
    horizon: int = 1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """使用仓库内标准的 NIR/属性文件生成 PINN 训练数据集。"""
    y, _ = build_property_vector_from_all_csv(
        'data/physical/all_csv_data.csv',
        property_name,
        'data/NIR',
        average_num=1,
    )
    nir_folder = PROJECT_ROOT / 'data' / 'NIR'
    file_names = sorted(path.name for path in nir_folder.glob('*.csv'))
    if not file_names:
        raise FileNotFoundError(f'No NIR CSV files found in {nir_folder}')

    X_nir = black_white_processing(
        nir_folder,
        black_dir=PROJECT_ROOT / 'data' / 'Black white',
    )[: y.shape[0], :].astype(np.float32)
    if X_nir.shape[0] != y.shape[0]:
        raise ValueError('NIR sample count does not match the property vector length for PINN training')

    metadata_rows = _extract_aligned_metadata_rows(file_names[: y.shape[0]])
    filename_metadata = _parse_filename_experiment_tokens(file_names[: y.shape[0]])
    metadata_rows = metadata_rows.merge(filename_metadata, on='csv_name', how='left')
    trajectory_ids = metadata_rows['filename_trajectory_id'].astype(str).to_numpy()
    times = _extract_numeric_series(
        metadata_rows,
        (
            'time',
            'times',
            'hour',
            'hours',
            'day',
            'days',
            'storage_time',
            'storage_hours',
            'storage_days',
            'filename_time',
        ),
        field_name='time',
        required=require_times,
    )
    temperatures = _extract_numeric_series(
        metadata_rows,
        ('temperature', 'temperatures', 'temp', 'temps', 't', 'filename_temperature'),
        field_name='temperature',
        required=require_temperatures,
    )
    if np.nanmax(temperatures) < 200:
        temperatures = temperatures + 273.15
    X_enose = _extract_enose_features(metadata_rows, y.shape[0], required=require_enose)

    resolved_fs_method = (fs_method or '').strip().lower()
    selected_feature_idx = None
    original_feature_count = int(X_nir.shape[1])
    if resolved_fs_method and resolved_fs_method != 'none':
        if resolved_fs_method == 'all':
            raise ValueError('PINN does not support fs_method=all; please choose one concrete method')
        fs_result = select_features_by_method(X_nir, y.astype(np.float32), resolved_fs_method, fs_param)
        selected_feature_idx = np.asarray(fs_result['selected_idx'], dtype=int)
        X_nir = X_nir[:, selected_feature_idx].astype(np.float32)

    X_nir, X_enose, y, times, temperatures, trajectory_ids = _build_sliding_forecast_windows(
        X_nir=X_nir,
        X_enose=X_enose,
        y=y.astype(np.float32),
        times=times.astype(np.float32),
        temperatures=temperatures.astype(np.float32),
        trajectory_ids=trajectory_ids,
        context_size=context_size,
        horizon=horizon,
    )

    mode_tokens = parse_preproc_mode(preproc_mode)
    dataset = prepare_pinn_dataset(
        X_nir=X_nir,
        X_enose=X_enose,
        y=y.astype(np.float32),
        times=times.astype(np.float32),
        temperatures=temperatures.astype(np.float32),
        group_ids=trajectory_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augment=augment,
        mixup_alpha=mixup_alpha,
        mixup_samples=mixup_samples,
        nir_snv='snv' in mode_tokens,
        nir_apply_sg='sg' in mode_tokens,
        nir_sg_window=sg_window,
        nir_sg_order=sg_order,
        nir_normalize=True,
        enose_normalize=True,
        random_state=random_state,
    )
    dataset['metadata'] = {
        'selected_feature_idx': selected_feature_idx.tolist() if selected_feature_idx is not None else None,
        'fs_method': resolved_fs_method or None,
        'fs_param': fs_param,
        'original_feature_count': original_feature_count,
        'selected_feature_count': int(X_nir.shape[1] // max(1, context_size)),
        'context_size': context_size,
        'horizon': horizon,
    }
    return dataset
