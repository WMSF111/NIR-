# Python NIR Project

这是一个基于原 MATLAB `NIR+` 项目逻辑的 Python 实现。

## 特性

- 从 `data/physical/all_csv_data.csv` 中读取目标属性
- 从 `data/NIR` 中读取光谱数据，并进行黑白校正
- 支持 SG 滤波、MSC、SNV、基线归零、去尖刺等预处理
- 支持特征筛选：`corr_topk`、`pca`、`spa`、`cars`
- 支持回归器：`pls`、`pcr`、`svr`、`rf`、`gpr`、`knn`、`svr`、`rf`、`knn`
- 提供与 MATLAB 主流程一致的 `run_property_prediction` 和 `compare_property_prediction_pipeline`

## 安装

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python scripts/run_property_prediction.py --property_name "a*" --method_name "spa" --preproc_mode "sg+msc+snv" --sg_order 3 --sg_window 15
```

## 目录结构

- `nir_project/`：Python 包代码
- `scripts/`：运行入口脚本
- `requirements.txt`：依赖项
