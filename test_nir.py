#!/usr/bin/env python3
"""NIR 项目的定向回归测试集合。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'nir-mpl'))

import numpy as np
import torch

from nir_project.cli import run_prediction
from nir_project.data import load_dataset, save_dataset
from nir_project.feature_selection import quick_self_check_feature_selection, select_features_by_method
from nir_project.modeling import _preprocess_split, train_model_from_dataset
from nir_project.pinn.dataset import (
    _build_sliding_forecast_windows,
    prepare_pinn_dataset,
    prepare_pinn_dataset_from_project_data,
)
from nir_project.pinn.evaluate import evaluate_scalar_observations, plot_prediction_scatter, plot_sparse_observation_kinetics
from nir_project.pinn.loss import PINNLoss
from nir_project.pinn.train import train_pinn_two_stage
from nir_project.pipeline import (
    batch_compare_pinn_prediction,
    batch_compare_property_prediction,
    compare_property_prediction_pipeline,
    run_pinn_prediction,
    run_property_prediction,
)
from nir_project.preprocessing import parse_preproc_mode, preprocess_pair


class DatasetPersistenceTests(unittest.TestCase):
    """验证数据集保存与加载行为的一致性。"""

    def test_save_dataset_keeps_metadata_consistent_across_outputs(self) -> None:
        X = np.arange(12, dtype=float).reshape(3, 4)
        y = np.array([1.0, 2.0, 3.0])
        seed_metadata = {'property_name': 'a*', 'fs_method': 'corr_topk'}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('nir_project.data.PROJECT_ROOT', Path(tmp_dir)):
                dataset = save_dataset('demo/tag', X, y, seed_metadata, keep_exports=False)
                loaded = load_dataset(Path(dataset['paths']['npz']))
                metadata_json = Path(dataset['paths']['dir']) / 'metadata.json'
                metadata_from_json = json.loads(metadata_json.read_text(encoding='utf-8'))

        self.assertEqual(dataset['metadata'], loaded['metadata'])
        self.assertEqual(dataset['metadata'], metadata_from_json)
        self.assertEqual(loaded['metadata']['dataset_tag'], 'demo/tag')
        self.assertEqual(loaded['metadata']['sample_count'], 3)
        self.assertEqual(loaded['metadata']['feature_count'], 4)
        self.assertFalse(loaded['metadata']['keep_exports'])


class ImportSurfaceTests(unittest.TestCase):
    """验证公开入口的导入面保持可用。"""

    def test_public_entry_points_are_importable(self) -> None:
        self.assertTrue(callable(compare_property_prediction_pipeline))
        self.assertTrue(callable(run_prediction))


class CliRoutingTests(unittest.TestCase):
    """验证新的 mode 入口路由。"""

    def test_run_prediction_uses_mode_train_for_single_training(self) -> None:
        fake_result = {'method': 'pls', 'feature_selection': 'pca'}

        with patch('nir_project.cli.run_property_prediction', return_value=fake_result) as mocked_run:
            result = run_prediction(
                property_name='a*',
                model_type='traditional',
                mode='train',
                fs_method='pca',
                fs_param=40,
            )

        mocked_run.assert_called_once()
        self.assertEqual(mocked_run.call_args.kwargs['mode'], 'train')
        self.assertEqual(mocked_run.call_args.kwargs['fs_method'], 'pca')
        self.assertEqual(result, fake_result)


class PreprocessingLogicTests(unittest.TestCase):
    """验证预处理参数在边界场景下仍能生效。"""

    def test_preprocess_pair_applies_baseline_even_when_preproc_mode_is_none(self) -> None:
        X_train = np.array([[5.0, 7.0, 9.0], [2.0, 4.0, 6.0]])
        X_test = np.array([[3.0, 6.0, 9.0]])

        train_proc, test_proc = preprocess_pair(
            X_train,
            X_test,
            preproc_mode='none',
            baseline_zero_mode='first_point',
        )

        np.testing.assert_allclose(train_proc[:, 0], 0.0)
        np.testing.assert_allclose(test_proc[:, 0], 0.0)

    def test_modeling_preprocess_split_still_applies_baseline_when_mode_is_none(self) -> None:
        X = np.array([[5.0, 7.0, 9.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])

        train_proc, test_proc = _preprocess_split(
            X,
            preproc_mode='none',
            baseline_zero_mode='first_point',
        )

        np.testing.assert_allclose(train_proc[:, 0], 0.0)
        np.testing.assert_allclose(test_proc[:, 0], 0.0)


class PinnDatasetTests(unittest.TestCase):
    """验证 PINN 数据准备逻辑。"""

    def test_prepare_pinn_dataset_returns_test_split_and_matched_collocation_shapes(self) -> None:
        rng = np.random.default_rng(0)
        dataset = prepare_pinn_dataset(
            X_nir=rng.normal(size=(12, 8)).astype(np.float32),
            X_enose=rng.normal(size=(12, 3)).astype(np.float32),
            y=rng.normal(size=12).astype(np.float32),
            times=np.linspace(0, 11, 12, dtype=np.float32),
            temperatures=np.full(12, 298.15, dtype=np.float32),
            mixup_samples=4,
            random_state=0,
        )

        self.assertIn('test', dataset)
        self.assertGreater(dataset['test']['X_nir'].shape[0], 0)
        collocation = dataset['collocation']
        collocation_size = collocation['X_nir'].shape[0]
        self.assertEqual(collocation['X_enose'].shape[0], collocation_size)
        self.assertEqual(collocation['y'].shape[0], collocation_size)
        self.assertEqual(collocation['times'].shape[0], collocation_size)
        self.assertEqual(collocation['temperatures'].shape[0], collocation_size)

    def test_prepare_pinn_dataset_splits_by_trajectory_group(self) -> None:
        dataset = prepare_pinn_dataset(
            X_nir=np.ones((8, 6), dtype=np.float32),
            X_enose=np.ones((8, 2), dtype=np.float32),
            y=np.linspace(1.0, 8.0, 8, dtype=np.float32),
            times=np.array([1, 2, 101, 102, 201, 202, 301, 302], dtype=np.float32),
            temperatures=np.full(8, 298.15, dtype=np.float32),
            group_ids=np.array(['g1', 'g1', 'g2', 'g2', 'g3', 'g3', 'g4', 'g4']),
            augment=False,
            random_state=0,
        )

        def extract_group_markers(split_name: str) -> set[int]:
            times_array = dataset[split_name]['times'].numpy().reshape(-1)
            return {int(value // 100) for value in times_array}

        train_groups = extract_group_markers('train')
        val_groups = extract_group_markers('val')
        test_groups = extract_group_markers('test')

        self.assertTrue(train_groups.isdisjoint(val_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(val_groups.isdisjoint(test_groups))

    def test_build_sliding_forecast_windows_uses_previous_two_points_to_predict_next(self) -> None:
        X_nir = np.arange(30, dtype=np.float32).reshape(5, 6)
        X_enose = np.arange(10, dtype=np.float32).reshape(5, 2)
        y = np.array([10, 11, 12, 13, 14], dtype=np.float32)
        times = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        temperatures = np.full(5, 343.15, dtype=np.float32)
        groups = np.array(['70_0'] * 5)

        win_nir, win_enose, win_y, win_times, win_temps, win_groups = _build_sliding_forecast_windows(
            X_nir=X_nir,
            X_enose=X_enose,
            y=y,
            times=times,
            temperatures=temperatures,
            trajectory_ids=groups,
            context_size=2,
            horizon=1,
        )

        self.assertEqual(win_nir.shape[0], 3)
        np.testing.assert_allclose(win_times, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(win_y, [12.0, 13.0, 14.0])
        np.testing.assert_allclose(win_nir[0], np.concatenate([X_nir[0], X_nir[1]]))
        np.testing.assert_allclose(win_enose[1], np.concatenate([X_enose[1], X_enose[2]]))
        self.assertEqual(win_groups.tolist(), ['70_0', '70_0', '70_0'])

    def test_prepare_pinn_dataset_from_project_data_requires_real_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            physical_dir = root / 'data' / 'physical'
            nir_dir = root / 'data' / 'NIR'
            black_white_dir = root / 'data' / 'Black white'
            physical_dir.mkdir(parents=True)
            nir_dir.mkdir(parents=True)
            black_white_dir.mkdir(parents=True)

            for name in ('sample_1.csv', 'sample_2.csv', 'sample_3.csv'):
                (nir_dir / name).write_text('wavelength,intensity\n1000,1\n', encoding='utf-8')

            (physical_dir / 'all_csv_data.csv').write_text(
                'csv_name\nsample_1.csv\nsample_2.csv\nsample_3.csv\n',
                encoding='utf-8',
            )

            with patch('nir_project.pinn.dataset.PROJECT_ROOT', root), patch(
                'nir_project.pinn.dataset.build_property_vector_from_all_csv',
                return_value=(np.array([1.0, 2.0, 3.0], dtype=np.float32), None),
            ), patch(
                'nir_project.pinn.dataset.black_white_processing',
                return_value=np.ones((3, 4), dtype=np.float32),
            ):
                with self.assertRaisesRegex(ValueError, 'could not infer experiment metadata from file names'):
                    prepare_pinn_dataset_from_project_data('a*')

    def test_prepare_pinn_dataset_from_project_data_can_parse_time_and_temperature_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            physical_dir = root / 'data' / 'physical'
            nir_dir = root / 'data' / 'NIR'
            black_white_dir = root / 'data' / 'Black white'
            physical_dir.mkdir(parents=True)
            nir_dir.mkdir(parents=True)
            black_white_dir.mkdir(parents=True)

            for name in (
                '70_0_1.csv',
                '70_0_5.csv',
                '70_0_9.csv',
                '80_1_10.csv',
                '80_1_15.csv',
                '80_1_20.csv',
                '90_2_3.csv',
                '90_2_8.csv',
                '90_2_13.csv',
            ):
                (nir_dir / name).write_text('wavelength,intensity\n1000,1\n', encoding='utf-8')

            (physical_dir / 'all_csv_data.csv').write_text(
                (
                    'csv_name,a*\n'
                    '70_0_1.csv,1.0\n'
                    '70_0_5.csv,2.0\n'
                    '70_0_9.csv,3.0\n'
                    '80_1_10.csv,4.0\n'
                    '80_1_15.csv,5.0\n'
                    '80_1_20.csv,6.0\n'
                    '90_2_3.csv,7.0\n'
                    '90_2_8.csv,8.0\n'
                    '90_2_13.csv,9.0\n'
                ),
                encoding='utf-8',
            )

            with patch('nir_project.pinn.dataset.PROJECT_ROOT', root), patch(
                'nir_project.pinn.dataset.build_property_vector_from_all_csv',
                return_value=(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32), None),
            ), patch(
                'nir_project.pinn.dataset.black_white_processing',
                return_value=np.ones((9, 4), dtype=np.float32),
            ):
                dataset = prepare_pinn_dataset_from_project_data('a*')

        all_times = np.concatenate(
            [
                dataset['train']['times'].numpy().reshape(-1),
                dataset['val']['times'].numpy().reshape(-1),
                dataset['test']['times'].numpy().reshape(-1),
            ]
        )
        all_temperatures = np.concatenate(
            [
                dataset['train']['temperatures'].numpy().reshape(-1),
                dataset['val']['temperatures'].numpy().reshape(-1),
                dataset['test']['temperatures'].numpy().reshape(-1),
            ]
        )

        np.testing.assert_allclose(sorted(all_times.tolist()), [9.0, 13.0, 20.0])
        np.testing.assert_allclose(
            sorted(all_temperatures.tolist()),
            [343.15, 353.15, 363.15],
            rtol=0.0,
            atol=1e-3,
        )

    def test_prepare_pinn_dataset_from_project_data_applies_feature_selection_before_windowing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            physical_dir = root / 'data' / 'physical'
            nir_dir = root / 'data' / 'NIR'
            black_white_dir = root / 'data' / 'Black white'
            physical_dir.mkdir(parents=True)
            nir_dir.mkdir(parents=True)
            black_white_dir.mkdir(parents=True)

            for name in (
                '70_0_1.csv',
                '70_0_5.csv',
                '70_0_9.csv',
                '80_1_10.csv',
                '80_1_15.csv',
                '80_1_20.csv',
                '90_2_3.csv',
                '90_2_8.csv',
                '90_2_13.csv',
            ):
                (nir_dir / name).write_text('wavelength,intensity\n1000,1\n', encoding='utf-8')

            (physical_dir / 'all_csv_data.csv').write_text(
                (
                    'csv_name,a*\n'
                    '70_0_1.csv,1.0\n'
                    '70_0_5.csv,2.0\n'
                    '70_0_9.csv,3.0\n'
                    '80_1_10.csv,4.0\n'
                    '80_1_15.csv,5.0\n'
                    '80_1_20.csv,6.0\n'
                    '90_2_3.csv,7.0\n'
                    '90_2_8.csv,8.0\n'
                    '90_2_13.csv,9.0\n'
                ),
                encoding='utf-8',
            )

            fake_x = np.array(
                [
                    [9, 1, 0, 0],
                    [8, 1, 0, 0],
                    [7, 1, 0, 0],
                    [6, 1, 0, 0],
                    [5, 1, 0, 0],
                    [4, 1, 0, 0],
                    [3, 1, 0, 0],
                    [2, 1, 0, 0],
                    [1, 1, 0, 0],
                ],
                dtype=np.float32,
            )

            with patch('nir_project.pinn.dataset.PROJECT_ROOT', root), patch(
                'nir_project.pinn.dataset.build_property_vector_from_all_csv',
                return_value=(np.array([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32), None),
            ), patch(
                'nir_project.pinn.dataset.black_white_processing',
                return_value=fake_x,
            ):
                dataset = prepare_pinn_dataset_from_project_data(
                    'a*',
                    fs_method='corr_topk',
                    fs_param=2,
                )

        self.assertEqual(dataset['metadata']['fs_method'], 'corr_topk')
        self.assertEqual(dataset['metadata']['fs_param'], 2)
        self.assertEqual(len(dataset['metadata']['selected_feature_idx']), 2)
        self.assertEqual(dataset['metadata']['original_feature_count'], 4)
        self.assertEqual(dataset['metadata']['selected_feature_count'], 2)
        self.assertEqual(dataset['train']['X_nir'].shape[1], 4)


class PinnLossAndVizTests(unittest.TestCase):
    """验证 PINN 损失函数与可视化逻辑。"""

    def test_physics_loss_is_unlabeled(self) -> None:
        loss_fn = PINNLoss()
        c_pred = torch.tensor([[1.0], [0.8]], dtype=torch.float32)
        times = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        temps = torch.tensor([[298.15], [298.15]], dtype=torch.float32)
        dc_dt = torch.tensor([[-0.1], [-0.08]], dtype=torch.float32)

        loss = loss_fn.compute_physics_loss(c_pred, times, temps, dc_dt)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_evaluate_scalar_observations_validates_sparse_inputs(self) -> None:
        metrics = evaluate_scalar_observations(
            predictions=np.array([1.0, 2.0, 3.0]),
            observations=np.array([1.1, 1.9, 3.2]),
            time_points=np.array([0.0, 1.0, 2.0]),
            sample_ids=[0, 1, 2],
        )

        self.assertIn('rmse', metrics)

        with self.assertRaisesRegex(ValueError, '期望 1 维 predictions'):
            evaluate_scalar_observations(
                predictions=np.array([[1.0], [2.0]]),
                observations=np.array([1.0, 2.0]),
                time_points=np.array([0.0, 1.0]),
                sample_ids=[0, 1],
            )

    def test_plot_sparse_observation_kinetics_saves_sample_and_overview_figures(self) -> None:
        class DummyModel(torch.nn.Module):
            def forward(self, x_nir, x_enose, times, temperatures):
                return times + 0.01 * temperatures

        model = DummyModel()
        x_nir = torch.ones((3, 4), dtype=torch.float32)
        x_enose = torch.ones((3, 2), dtype=torch.float32)
        times = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        temperatures = torch.full((3, 1), 298.15, dtype=torch.float32)
        y_true = np.array([3.8, 4.1, 4.4], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = plot_sparse_observation_kinetics(
                model=model,
                X_nir=x_nir,
                X_enose=x_enose,
                times_seq=times,
                temperatures=temperatures,
                y_true=y_true,
                save_path=str(Path(tmp_dir) / 'kinetics.png'),
            )

            self.assertTrue(Path(paths['samples']).exists())
            self.assertTrue(Path(paths['overview']).exists())

    def test_plot_prediction_scatter_saves_chinese_explanation_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = plot_prediction_scatter(
                y_true=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                y_pred=np.array([1.1, 1.9, 3.2], dtype=np.float32),
                metrics={'r2': 0.95, 'rmse': 0.12, 'mae': 0.10},
                save_path=str(Path(tmp_dir) / 'prediction_scatter.png'),
            )

            self.assertEqual(save_path, str(Path(tmp_dir) / 'prediction_scatter.png'))
            self.assertTrue(Path(save_path).exists())

    def test_train_pinn_two_stage_uses_total_epoch_semantics(self) -> None:
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 1)

            def forward(self, x_nir, x_enose, times, temperatures):
                combined = torch.cat([x_nir[:, :1], x_enose[:, :1], times, temperatures], dim=1)
                return self.linear(combined)

        model = SimpleModel()
        loss_fn = PINNLoss()
        train_data = {
            'X_nir': torch.ones((4, 3), dtype=torch.float32),
            'X_enose': torch.ones((4, 1), dtype=torch.float32),
            'times': torch.arange(4, dtype=torch.float32).reshape(-1, 1),
            'temperatures': torch.full((4, 1), 298.15, dtype=torch.float32),
            'y': torch.ones((4, 1), dtype=torch.float32),
        }
        history = train_pinn_two_stage(
            model=model,
            loss_fn=loss_fn,
            train_data=train_data,
            collocation_data=train_data,
            total_epochs=6,
            stage1_ratio=0.5,
            verbose=False,
        )

        self.assertEqual(len(history['train_loss']), 6)


class PinnPipelineIntegrationTests(unittest.TestCase):
    """验证 PINN 主流程在最小输入下可端到端执行。"""

    def test_run_pinn_prediction_end_to_end_with_minimal_inputs(self) -> None:
        dataset = {
            'train': {
                'X_nir': torch.ones((4, 3), dtype=torch.float32),
                'X_enose': torch.ones((4, 1), dtype=torch.float32),
                'times': torch.arange(4, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((4, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((4, 1), dtype=torch.float32),
            },
            'val': {
                'X_nir': torch.ones((2, 3), dtype=torch.float32),
                'X_enose': torch.ones((2, 1), dtype=torch.float32),
                'times': torch.arange(2, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((2, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((2, 1), dtype=torch.float32),
            },
            'test': {
                'X_nir': torch.ones((3, 3), dtype=torch.float32),
                'X_enose': torch.ones((3, 1), dtype=torch.float32),
                'times': torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
                'temperatures': torch.full((3, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((3, 1), dtype=torch.float32),
            },
            'collocation': {
                'X_nir': torch.ones((4, 3), dtype=torch.float32),
                'X_enose': torch.ones((4, 1), dtype=torch.float32),
                'times': torch.arange(4, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((4, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((4, 1), dtype=torch.float32),
            },
        }

        with patch('nir_project.pinn.dataset.prepare_pinn_dataset_from_project_data', return_value=dataset), patch(
            'nir_project.pipeline.train.train_pinn_two_stage',
            return_value={'train_loss': [1.0], 'data_loss': [0.5], 'physics_loss': [0.5]},
        ), patch(
            'nir_project.pipeline.evaluate.plot_sparse_observation_kinetics',
            return_value={'samples': 'samples.png', 'overview': 'overview.png'},
        ), tempfile.TemporaryDirectory() as tmp_dir, patch(
            'nir_project.pipeline.PROJECT_ROOT',
            Path(tmp_dir),
        ):
            result = run_pinn_prediction('a*')
            self.assertTrue(Path(result['prediction_table_path']).exists())
            self.assertTrue(Path(result['metrics_path']).exists())
            self.assertTrue(Path(result['training_history_path']).exists())
            self.assertTrue(Path(result['run_config_path']).exists())
            self.assertTrue(Path(result['prediction_scatter_path']).exists())

        self.assertEqual(result['model_type'], 'PINN')
        self.assertIn('test_metrics', result)
        self.assertIn('kinetics_plots', result)
        self.assertIn('dataset_metadata', result)

    def test_run_pinn_prediction_sanitizes_windows_unsafe_plot_name(self) -> None:
        dataset = {
            'train': {
                'X_nir': torch.ones((4, 3), dtype=torch.float32),
                'X_enose': torch.ones((4, 1), dtype=torch.float32),
                'times': torch.arange(4, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((4, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((4, 1), dtype=torch.float32),
            },
            'val': {
                'X_nir': torch.ones((2, 3), dtype=torch.float32),
                'X_enose': torch.ones((2, 1), dtype=torch.float32),
                'times': torch.arange(2, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((2, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((2, 1), dtype=torch.float32),
            },
            'test': {
                'X_nir': torch.ones((3, 3), dtype=torch.float32),
                'X_enose': torch.ones((3, 1), dtype=torch.float32),
                'times': torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
                'temperatures': torch.full((3, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((3, 1), dtype=torch.float32),
            },
            'collocation': {
                'X_nir': torch.ones((4, 3), dtype=torch.float32),
                'X_enose': torch.ones((4, 1), dtype=torch.float32),
                'times': torch.arange(4, dtype=torch.float32).reshape(-1, 1),
                'temperatures': torch.full((4, 1), 298.15, dtype=torch.float32),
                'y': torch.ones((4, 1), dtype=torch.float32),
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('nir_project.pipeline.PROJECT_ROOT', Path(tmp_dir)), patch(
                'nir_project.pinn.dataset.prepare_pinn_dataset_from_project_data',
                return_value=dataset,
            ), patch(
                'nir_project.pipeline.train.train_pinn_two_stage',
                return_value={'train_loss': [1.0], 'data_loss': [0.5], 'physics_loss': [0.5]},
            ), patch(
                'nir_project.pipeline.evaluate.plot_sparse_observation_kinetics',
                return_value={'samples': 'samples.png', 'overview': 'overview.png'},
            ) as mocked_plot:
                run_pinn_prediction('a*')

        save_path = mocked_plot.call_args.kwargs['save_path']
        self.assertNotIn('*', save_path)
        self.assertTrue(save_path.endswith('pinn_kinetics_a_.png'))

    def test_run_prediction_routes_pinn_compare_all_to_batch_compare(self) -> None:
        with patch('nir_project.cli.batch_compare_pinn_prediction', return_value={'ok': True}) as mocked_compare:
            result = run_prediction(
                property_name='a*,b*',
                model_type='pinn',
                mode='compare',
                fs_method='all',
            )

        mocked_compare.assert_called_once()
        self.assertEqual(result, {'ok': True})


class BatchComparisonTests(unittest.TestCase):
    """验证批量算法对比输出。"""

    def test_batch_compare_generates_summary_tables(self) -> None:
        fake_result = {
            'method': 'pls',
            'group': 'selected',
            'feature_selection': 'pca',
            'fs_param': 40,
            'dataset_tag': 'demo/tag',
            'plot_path': 'demo_plot.png',
            'best_param': {'n_components': 3},
            'metrics': {
                'train': {'r2': 0.95, 'rmse': 0.1},
                'test': {'r2': 0.88, 'rmse': 0.2},
            },
            'ytrain': [1.0, 2.0],
            'ypred_train': [1.1, 1.9],
            'ytest': [3.0],
            'ypred': [2.8],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('nir_project.pipeline.PROJECT_ROOT', Path(tmp_dir)), patch(
                'nir_project.pipeline.compare_property_prediction_pipeline',
                return_value=[fake_result],
            ):
                result = batch_compare_property_prediction(
                    property_names=['a*', 'b*'],
                    preproc_mode='sg+msc',
                    feature_selection_method='pca',
                )
                self.assertTrue(Path(result['metrics_summary_path']).exists())
                self.assertTrue(Path(result['prediction_table_path']).exists())
                self.assertTrue(Path(result['parameter_table_path']).exists())
                self.assertEqual(set(result['property_plot_paths'].keys()), {'a*', 'b*'})

    def test_batch_compare_all_dispatches_all_feature_selection_runner(self) -> None:
        fake_result = {
            'method': 'pls',
            'group': 'selected',
            'feature_selection': 'spa',
            'fs_param': 8,
            'dataset_tag': 'demo/tag',
            'plot_path': 'demo_plot.png',
            'best_param': {'n_components': 3},
            'metrics': {
                'train': {'r2': 0.95, 'rmse': 0.1},
                'test': {'r2': 0.88, 'rmse': 0.2},
            },
            'ytrain': [1.0, 2.0],
            'ypred_train': [1.1, 1.9],
            'ytest': [3.0],
            'ypred': [2.8],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('nir_project.pipeline.PROJECT_ROOT', Path(tmp_dir)), patch(
                'nir_project.pipeline.compare_all_feature_selection_pipeline',
                return_value=[fake_result],
            ) as mocked_compare_all:
                result = batch_compare_property_prediction(
                    property_names=['a*'],
                    preproc_mode='sg+msc',
                    feature_selection_method='all',
                )

                mocked_compare_all.assert_called_once()
                self.assertTrue(Path(result['metrics_summary_path']).exists())

    def test_batch_compare_pinn_generates_summary_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prediction_path = root / 'single_prediction.csv'
            prediction_path.write_text(
                'sample_index,time,temperature_kelvin,y_true,y_pred\n1,1,298.15,1.0,1.1\n',
                encoding='utf-8-sig',
            )
            metrics_path = root / 'metrics.json'
            metrics_path.write_text('{}', encoding='utf-8')
            history_path = root / 'history.csv'
            history_path.write_text('train_loss\n1.0\n', encoding='utf-8-sig')
            config_path = root / 'config.json'
            config_path.write_text('{}', encoding='utf-8')

            fake_result = {
                'model_type': 'PINN',
                'test_metrics': {'r2': 0.8, 'rmse': 0.2, 'mae': 0.1, 'rpd': 1.8},
                'prediction_table_path': str(prediction_path),
                'metrics_path': str(metrics_path),
                'training_history_path': str(history_path),
                'run_config_path': str(config_path),
                'result_dir': str(root / 'single_run'),
                'dataset_metadata': {
                    'fs_method': 'corr_topk',
                    'fs_param': 20,
                    'original_feature_count': 2048,
                    'selected_feature_count': 20,
                    'selected_feature_idx': [1, 2, 3],
                },
            }

            with patch('nir_project.pipeline.PROJECT_ROOT', root), patch(
                'nir_project.pipeline.run_pinn_prediction',
                return_value=fake_result,
            ) as mocked_run:
                result = batch_compare_pinn_prediction(
                    property_names=['a*'],
                    preproc_mode='sg+msc',
                    feature_selection_method='corr_topk',
                    fs_param=20,
                )

            mocked_run.assert_called_once()
            self.assertTrue(Path(result['metrics_summary_path']).exists())
            self.assertTrue(Path(result['prediction_table_path']).exists())
            self.assertTrue(Path(result['parameter_table_path']).exists())


class ModelingGuardTests(unittest.TestCase):
    """验证传统建模流程的防御性保护。"""

    def test_train_model_from_dataset_rejects_too_few_samples(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('nir_project.data.PROJECT_ROOT', Path(tmp_dir)):
                dataset = save_dataset('tiny/demo', X, y, {'data_stage': 'raw'}, keep_exports=False)
                with self.assertRaisesRegex(ValueError, 'At least 4 samples'):
                    train_model_from_dataset(Path(dataset['paths']['npz']), 'pls', None)


class FeatureSelectionScalingTests(unittest.TestCase):
    """验证预处理模式解析等基础约束。"""

    def test_preproc_mode_parser_normalizes_shared_mode_semantics(self) -> None:
        self.assertEqual(parse_preproc_mode(' SG + snv + none '), {'sg', 'snv'})
        self.assertEqual(parse_preproc_mode('none'), set())

    def test_spa_feature_selection_runs_without_scalar_conversion_error(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(12, 6))
        y = rng.normal(size=12)

        result = select_features_by_method(X, y, 'spa', 3)

        self.assertEqual(len(result['selected_idx']), 3)

    def test_quick_self_check_feature_selection_supports_all_target_methods(self) -> None:
        result = quick_self_check_feature_selection(methods=['spa', 'pca', 'cars', 'corr_topk'])

        self.assertTrue(result['success'])
        self.assertEqual(set(result['methods'].keys()), {'spa', 'pca', 'cars', 'corr_topk'})
        for method_name, method_result in result['methods'].items():
            self.assertTrue(method_result['success'], msg=method_name)
            self.assertGreater(method_result['selected_count'], 0, msg=method_name)


if __name__ == '__main__':
    unittest.main()
