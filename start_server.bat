@echo off
echo 🚀 GPU QuickCap Server Startup
echo ===============================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo ⚠️  Virtual environment not detected
    echo    Make sure to activate your venv first:
    echo    venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

echo ✅ Virtual environment active: %VIRTUAL_ENV%
echo.

REM Try different ports
echo 🔍 Looking for available port...

REM Try port 8080 first
echo Testing port 8080...
python -c "import socket; s=socket.socket(); s.bind(('localhost', 8080)); s.close(); print('✅ Port 8080 available')" 2>nul
if %errorlevel% equ 0 (
    echo 🌐 Starting server on http://localhost:8080
    echo.
    python app.py
    goto :end
)

REM Try port 3000
echo Testing port 3000...
python -c "import socket; s=socket.socket(); s.bind(('localhost', 3000)); s.close(); print('✅ Port 3000 available')" 2>nul
if %errorlevel% equ 0 (
    echo 🌐 Starting server on http://localhost:3000
    echo.
    uvicorn app:app --host 0.0.0.0 --port 3000 --reload
    goto :end
)

REM Try port 5000
echo Testing port 5000...
python -c "import socket; s=socket.socket(); s.bind(('localhost', 5000)); s.close(); print('✅ Port 5000 available')" 2>nul
if %errorlevel% equ 0 (
    echo 🌐 Starting server on http://localhost:5000
    echo.
    uvicorn app:app --host 0.0.0.0 --port 5000 --reload
    goto :end
)

echo ❌ No available ports found. Please close other applications.
echo    Or try manually: uvicorn app:app --host 0.0.0.0 --port 8080 --reload

:end
pause