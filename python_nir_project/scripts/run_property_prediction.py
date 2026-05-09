from __future__ import annotations

"""Single entry script with all result-affecting parameters gathered here."""

import subprocess
import sys
from pathlib import Path
from pprint import pprint

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
    # 'fs_param_grid': {
    #     'cars': [120, 150],
    #     'pca': [40, 80, 120, 160, 200],
    #     'corr_topk': [20, 40, 80, 120],
    #     'spa': [8, 12, 16, 20],
    # },
    'fs_param_grid': {
        'cars': [10, 20, 30, 40],
        'pca': [10, 20, 30, 40],
        'corr_topk': [10, 20, 30, 40],
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


def run_from_config():
    """Run directly from the configuration in this file."""
    result = run_prediction(
        property_name=RUN_CONFIG['property_name'],
        model_type=RUN_CONFIG['model_type'],
        mode=RUN_CONFIG['mode'],
        preproc_mode=RUN_CONFIG['preproc_mode'],
        sg_order=RUN_CONFIG['sg_order'],
        sg_window=RUN_CONFIG['sg_window'],
        fs_method=RUN_CONFIG['fs_method'],
        fs_param=RUN_CONFIG['fs_param'],
        include_preprocessed_group=RUN_CONFIG['include_preprocessed_group'],
        traditional_config=TRADITIONAL_CONFIG,
        pinn_config=PINN_CONFIG,
    )
    pprint(result)
    return result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main()
    else:
        run_from_config()
