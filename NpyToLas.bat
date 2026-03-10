@echo off
if "%~1"=="" (
    echo Drag and drop a .npy file onto this batch file.
    pause
    exit /b 1
)
echo Converting %~nx1 to LAS...
"C:\Users\zkrep\AppData\Local\Programs\Python\Python312\python.exe" "C:\Users\zkrep\AutoGateDetector\npy_to_las.py" "%~1"
echo.
pause
