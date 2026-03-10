@echo off
set CLOUD_FILE=%~1
start "CloudViewer" /D "C:\Users\zkrep\AutoGateDetector" "C:\Users\zkrep\AppData\Local\Programs\Python\Python312\Scripts\streamlit.exe" run cloud_viewer.py --server.port 8054 --server.headless false
