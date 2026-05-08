from __future__ import annotations

"""命令行兼容脚本，用于转调 `nir_project.cli` 入口。"""

import subprocess
import sys
from pathlib import Path

try:
    from nir_project.cli import main, run_prediction
except ModuleNotFoundError as exc:
    if exc.name == 'nir_project':
        project_package_root = Path(__file__).resolve().parents[1]
        repo_root = project_package_root.parent
        if str(project_package_root) not in sys.path:
            sys.path.insert(0, str(project_package_root))
        try:
            from nir_project.cli import main, run_prediction
        except ModuleNotFoundError:
            venv_python = repo_root / '.venv' / 'Scripts' / 'python.exe'
            current_python = Path(sys.executable).resolve() if sys.executable else None
            if venv_python.exists() and current_python != venv_python.resolve():
                raise SystemExit(
                    subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
                ) from exc
            raise SystemExit(
                '当前解释器无法导入 nir_project。'
                '请优先使用项目虚拟环境运行，或先在仓库根目录执行 '
                '`python -m pip install -e .`。'
            ) from exc
    raise


if __name__ == '__main__':
    main()
