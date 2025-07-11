@echo off
REM ------------------------------------------------------------
REM install_prerequisites.bat
REM Installs Python prerequisites for rom_duplicates.py,
REM sales_to_gamelist.py, all_rom_formats_list.py, and m3u_multidisc.py
REM ------------------------------------------------------------

REM 1. Check for Python
python --version >nul 2>&1
if ERRORLEVEL 1 (
    echo.
    echo ERROR: Python 3 is not found in your PATH.
    echo Please install Python 3 from https://www.python.org/downloads/ and try again.
    pause
    exit /B 1
)

REM 2. Upgrade pip
echo.
echo Upgrading pip to the latest version...
python -m pip install --upgrade pip

REM 3. Install required packages
echo.
echo Installing required Python packages...
python -m pip install rapidfuzz pandas pyreadline

if ERRORLEVEL 1 (
    echo.
    echo ERROR: Failed to install one or more packages.
    echo Please check your internet connection and that pip is configured correctly.
    pause
    exit /B 1
)

echo.
echo All prerequisites installed successfully!
pause
