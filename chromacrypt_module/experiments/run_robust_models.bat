@echo off
echo ========================================================
echo Invoking Robust Models Evaluation...
echo ========================================================

call ..\venv_win\Scripts\activate.bat

python benchmark_robust_models.py

pause
