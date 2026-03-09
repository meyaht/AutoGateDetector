@echo off
REM AutoGateDetector launcher
REM Usage: drag a .npy or .e57 file onto this bat, or run:
REM   AutoGateDetector.bat C:\path\to\cloud.npy [--axis X] [--step 1.0] [--zmin 0] [--zmax 7]
REM   AutoGateDetector.bat C:\path\to\cloud.e57 [--axis X] [--step 1.0] [--zmin 0] [--zmax 7]

set PYTHON=C:\Users\zkrep\AppData\Local\Programs\Python\Python312\python.exe
set SCRIPT=%~dp0pipeline.py

if "%~1"=="" (
    echo Usage: AutoGateDetector.bat ^<cloud.npy^> [--axis X^|Y^|both] [--step 1.0] [--zmin 0] [--zmax 7]
    echo.
    echo Drop a .npy or .e57 file onto this bat file to use defaults:
    echo   axis=Y  step=1m  zmin=0  zmax=7
    echo.
    echo Use --axis both to scan XZ and YZ planes in a single run.
    pause
    exit /b 1
)

echo Running AutoGateDetector on: %~1
echo.
"%PYTHON%" "%SCRIPT%" %1 --axis both %2 %3 %4 %5 %6 %7 %8 %9
pause
