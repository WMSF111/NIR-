@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "ENTRY_SCRIPT=%SCRIPT_DIR%python_nir_project\scripts\run_property_prediction.py"

if not exist "%PYTHON_EXE%" (
    echo Virtual environment interpreter not found: "%PYTHON_EXE%"
    echo Please create .venv in the repository root and install project dependencies first.
    exit /b 1
)

if not exist "%ENTRY_SCRIPT%" (
    echo Entry script not found: "%ENTRY_SCRIPT%"
    exit /b 1
)

"%PYTHON_EXE%" "%ENTRY_SCRIPT%" %*
