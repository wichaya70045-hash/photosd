@echo off
REM ============================================================
REM  AI Photo Selection System — Quick Start (Windows)
REM ============================================================
echo.
echo  ========================================
echo   AI Photo Selection System — Startup
echo  ========================================
echo.

REM Try to find a working Python (avoid Windows Store stub)
set PYTHON_EXE=
for %%P in (
    "%LOCALAPPDATA%\Python\bin\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%P (
        set PYTHON_EXE=%%P
        goto :found_python
    )
)

REM Fallback to PATH python
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_EXE=python
    goto :found_python
)

echo [ERROR] Python ไม่พบ กรุณาติดตั้ง Python 3.11+
pause
exit /b 1

:found_python
echo [INFO] ใช้ Python: %PYTHON_EXE%

REM Create venv if not exists
if not exist venv (
    echo [INFO] กำลังสร้าง virtual environment...
    %PYTHON_EXE% -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install dependencies
echo [INFO] กำลังติดตั้ง dependencies...
pip install -r requirements.txt

REM Start server
echo.
echo [INFO] เริ่มระบบ...
echo [INFO] เปิดเบราว์เซอร์ที่  http://localhost:8000
echo [INFO] กด Ctrl+C เพื่อหยุด
echo.
python main.py

pause
