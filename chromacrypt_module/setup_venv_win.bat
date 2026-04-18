@echo off
echo ========================================================
echo Constructing Isolated ChromaCrypt Virtual Environment...
echo ========================================================

python -m venv venv_win
call venv_win\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing core mathematical logic and metrics...
pip install -r requirements.txt

echo ========================================================
echo Environment Synthesis Complete!
echo You can now execute experiments using the local runner scripts.
echo ========================================================
pause
