@echo off
REM AutoGateDetector launcher
REM Usage: drag a .npy file onto this bat, or run:
REM   AutoGateDetector.bat C:\path\to\cloud.npy [--axis X] [--step 1.0] [--zmin 0] [--zmax 7]

set PYTHON=C:\Users\zkrep\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT=%~dp0pipeline.py

if "%~1"=="" (
    echo Usage: AutoGateDetector.bat ^<cloud.npy^> [--axis X^|Y] [--step 1.0] [--zmin 0] [--zmax 7]
    echo.
    echo Drop a .npy file onto this bat file to use defaults:
    echo   axis=Y  step=1m  zmin=0  zmax=7
    pause
    exit /b 1
)

echo Running AutoGateDetector on: %~1
echo.
"%PYTHON%" "%SCRIPT%" %*
pause
