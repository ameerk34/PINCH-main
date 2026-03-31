@echo off
setlocal

set "ROOT=C:\Users\ameer\OneDrive\Desktop\PINCH-main"
set "PYTHONW=%ROOT%\.venv\Scripts\pythonw.exe"
set "APP=%ROOT%\letsgo\pinchreader.py"

if not exist "%PYTHONW%" (
    echo pythonw.exe not found at:
    echo %PYTHONW%
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo App file not found at:
    echo %APP%
    pause
    exit /b 1
)

start "" "%PYTHONW%" "%APP%"
exit /b 0
