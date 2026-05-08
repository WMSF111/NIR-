$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
$entryScript = Join-Path $scriptDir "python_nir_project\scripts\run_property_prediction.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "未找到虚拟环境解释器: $pythonExe"
    Write-Error "请先在仓库根目录创建 .venv，并安装项目依赖。"
    exit 1
}

& $pythonExe $entryScript @args
