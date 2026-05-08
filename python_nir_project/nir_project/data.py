"""
数据处理模块 (data.py)

本模块负责NIR光谱数据的加载、预处理和数据集管理。

主要功能：
1. 光谱数据加载：从CSV文件读取NIR光谱数据
2. 黑白校正：去除仪器背景噪声
3. 属性向量构建：从CSV文件构建质量属性标签
4. 数据集保存/加载：压缩存储和快速加载

技术特点：
- 支持多种CSV格式的光谱数据
- 自动插值处理不同长度的光谱
- 压缩存储节省磁盘空间
- 元数据管理记录数据处理历史

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 项目根目录路径

def _resolve_project_root() -> Path:
    """自动定位包含 data 目录的项目根目录。"""
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / 'data').exists():
            return candidate
    return current.parents[1]


PROJECT_ROOT = _resolve_project_root()


def load_csv_spectrum(path: Path) -> np.ndarray:
    """
    从CSV文件加载NIR光谱数据

    支持多种CSV格式：
    - 单列：纯光谱数据
    - 多列：取最后一列作为光谱
    - 自动处理NaN值

    Args:
        path: CSV文件路径

    Returns:
        光谱数据数组 (一维)

    Raises:
        ValueError: 文件为空时抛出
    """
    # 读取CSV文件，无表头
    data = pd.read_csv(path, header=None).to_numpy(dtype=float)

    # 检查数据是否为空
    if data.size == 0:
        raise ValueError(f'Empty spectrum file: {path}')

    # 处理不同格式的CSV
    if data.ndim == 1:
        # 单列数据
        spectrum = data
    elif data.shape[1] >= 2:
        # 多列数据，取最后一列作为光谱
        spectrum = data[:, -1]
    else:
        # 其他情况，展平数据
        spectrum = data.ravel()

    # 移除NaN值并转换为float类型
    return spectrum[~np.isnan(spectrum)].astype(float)


def black_white_processing(
    csv_folder: Path,
    black_reference_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    black_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    黑白校正处理

    原理：从原始光谱中减去黑体参考（背景噪声），得到校正后的光谱。
    这是NIR光谱分析中的标准预处理步骤。

    Args:
        csv_folder: 包含NIR光谱CSV文件的文件夹
        black_reference_file: 黑体参考文件路径（可选）
        output_file: 输出文件路径（可选）
        black_dir: 黑体参考文件所在目录（可选）

    Returns:
        校正后的光谱矩阵 [n_samples, n_wavelengths]

    Raises:
        FileNotFoundError: 找不到黑体参考文件或NIR文件时抛出
    """
    csv_folder = Path(csv_folder)

    # 确定黑体参考文件
    if black_reference_file is None:
        if black_dir is None:
            # 默认黑体参考目录
            black_dir = PROJECT_ROOT / 'data' / 'Black white'
        else:
            black_dir = Path(black_dir)

        # 查找黑体参考文件（按文件名排序，取第一个）
        candidates = sorted(black_dir.glob('*.csv'), key=lambda p: p.name)
        if not candidates:
            raise FileNotFoundError(f'Black reference file not found in {black_dir}')
        black_reference_file = candidates[0]

    black_reference_file = Path(black_reference_file)

    # 查找所有NIR光谱文件
    nir_files = sorted(csv_folder.glob('*.csv'), key=lambda p: p.name)
    if not nir_files:
        raise FileNotFoundError(f'No NIR CSV files found in {csv_folder}')

    # 加载所有光谱
    spectra = []
    for csv_path in nir_files:
        spectrum = load_csv_spectrum(csv_path)
        spectra.append(spectrum)

    # 检查光谱长度是否一致
    lengths = [len(s) for s in spectra]
    if len(set(lengths)) != 1:
        # 如果长度不一致，进行插值对齐
        max_len = max(lengths)
        spectra = [
            np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(s)), s)
            for s in spectra
        ]

    # 堆叠成矩阵
    X = np.vstack(spectra)

    # 加载黑体参考
    black_data = load_csv_spectrum(black_reference_file)

    # 黑体参考长度对齐
    if black_data.shape[0] != X.shape[1]:
        black_data = np.interp(
            np.linspace(0, 1, X.shape[1]),
            np.linspace(0, 1, black_data.shape[0]),
            black_data,
        )

    # 黑白校正：减去背景噪声
    X = X - black_data.reshape(1, -1)

    # 保存结果（如果指定了输出文件）
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, X, delimiter=',')

    return X


def build_property_vector_from_all_csv(
    all_csv_relative_path: str,
    property_name: str,
    csv_subfolder: str,
    average_num: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从CSV文件构建属性向量

    从属性数据CSV文件中提取指定属性的标签值，并与NIR文件对应。
    支持多文件平均（减少测量噪声）。

    Args:
        all_csv_relative_path: 属性数据CSV文件相对路径
        property_name: 属性名称（如'sugar_content', 'moisture'）
        csv_subfolder: NIR光谱文件所在子文件夹
        average_num: 每个样本的平均文件数（默认1）

    Returns:
        y: 属性值数组 [n_samples]
        grouped_file_names: 分组后的文件名数组

    Raises:
        ValueError: CSV文件格式错误或属性不存在时抛出
    """
    # 构建完整路径
    all_csv_path = PROJECT_ROOT / all_csv_relative_path.replace('\\', '/').replace('\\', '/')
    csv_folder = PROJECT_ROOT / csv_subfolder.replace('\\', '/').replace('\\', '/')

    # 读取属性数据
    df = pd.read_csv(all_csv_path)

    # 检查必需列
    if 'csv_name' not in df.columns:
        raise ValueError('all_csv_data.csv must contain column csv_name')
    if property_name not in df.columns:
        raise ValueError(f'Property {property_name} not found in all_csv_data.csv')

    # 构建文件名到属性值的映射
    name_to_value = {
        str(name): float(value) if pd.notna(value) else np.nan
        for name, value in zip(df['csv_name'], df[property_name])
    }

    # 获取所有NIR文件
    nir_files = sorted(csv_folder.glob('*.csv'), key=lambda p: p.name)
    file_names = [p.name for p in nir_files]

    # 计算分组数量
    group_count = len(file_names) // average_num
    y = np.zeros(group_count, dtype=float)
    grouped_file_names = []

    # 按组处理
    for g in range(group_count):
        # 获取当前组的文件名
        group_names = file_names[g * average_num:(g + 1) * average_num]
        grouped_file_names.append(group_names)

        vals = []
        for name in group_names:
            if name not in name_to_value:
                raise ValueError(f'csv_name {name} exists in NIR folder but not in all_csv_data.csv')
            vals.append(name_to_value[name])

        # 计算平均值（自动处理NaN）
        y[g] = np.nanmean(vals)

    return y, np.array(grouped_file_names, dtype=object)


def save_dataset(
    dataset_tag: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    keep_exports: bool = False,
) -> Dict[str, Any]:
    """
    保存数据集到压缩文件

    使用numpy压缩格式(.npz)保存数据，包含特征矩阵X、标签y和元数据。
    可选择同时导出CSV格式用于检查。

    Args:
        dataset_tag: 数据集标签（用于文件夹命名）
        X: 特征矩阵 [n_samples, n_features]
        y: 标签数组 [n_samples]
        metadata: 元数据字典（可选）
        keep_exports: 是否保留CSV导出文件（默认False）

    Returns:
        完整的数据集字典，包含路径和元数据
    """
    if metadata is None:
        metadata = {}
    metadata = dict(metadata)
    metadata['dataset_tag'] = dataset_tag
    metadata['sample_count'] = X.shape[0]
    metadata['feature_count'] = X.shape[1]
    metadata['created_at'] = pd.Timestamp.now().isoformat()
    metadata['keep_exports'] = keep_exports

    if keep_exports:
        dataset_dir = PROJECT_ROOT / 'result' / 'dataset' / dataset_tag
    else:
        dataset_dir = PROJECT_ROOT / 'result' / 'temp' / 'dataset' / dataset_tag

    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / 'dataset.npz'

    # 压缩保存
    np.savez_compressed(dataset_path, X=X, y=y, metadata=metadata)

    # 可选：导出CSV用于检查
    if keep_exports:
        pd.DataFrame(X).to_csv(dataset_dir / 'X.csv', index=False, header=False)
        pd.DataFrame(y).to_csv(dataset_dir / 'y.csv', index=False, header=['y'])
        if 'selected_idx' in metadata:
            pd.DataFrame(np.array(metadata['selected_idx']).reshape(-1, 1)).to_csv(
                dataset_dir / 'selected_idx.csv', index=False, header=['selected_idx']
            )

    # 更新元数据
    dataset = {
        'X': X,
        'y': y.reshape(-1),
        'metadata': metadata,
        'paths': {
            'dir': str(dataset_dir),
            'npz': str(dataset_path),
        },
    }

    # 保存元数据JSON
    with open(dataset_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return dataset


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    """
    从压缩文件加载数据集

    Args:
        dataset_path: 数据集文件路径(.npz)

    Returns:
        完整的数据集字典

    Raises:
        FileNotFoundError: 文件不存在时抛出
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    # 加载压缩数据
    data = np.load(dataset_path, allow_pickle=True)

    # 提取元数据
    metadata = dict(data['metadata'].item()) if 'metadata' in data else {}

    return {
        'X': data['X'],
        'y': data['y'],
        'metadata': metadata,
        'paths': {'npz': str(dataset_path), 'dir': str(dataset_path.parent)},
    }
