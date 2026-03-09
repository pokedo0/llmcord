@echo off
chcp 65001 >nul
title llmcord - Local Debug

REM ─────────────────────────────────────────────
REM  llmcord 本地调试启动脚本
REM  用法: 双击此 bat 或在命令行运行
REM ─────────────────────────────────────────────

REM 切换到脚本所在目录 (确保相对路径生效)
cd /d "%~dp0"

REM ── 1. Python 虚拟环境 (如有) ──
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [*] No virtual environment found, using system Python.
)

REM ── 2. 设置环境变量 ──
REM Gemini cookies 路径: Docker 中挂载到 /tmp/gemini_webapi,
REM 本地直接指向项目下的 gemini_cookies 目录
set GEMINI_COOKIE_PATH=%~dp0gemini_cookies

REM 日志级别 (可选: DEBUG, INFO, WARNING, ERROR)
set LOGLEVEL=INFO

REM ── 3. 检查 gemini_cookies 目录是否存在 ──
if not exist "%GEMINI_COOKIE_PATH%" (
    echo [!] Warning: gemini_cookies directory not found at %GEMINI_COOKIE_PATH%
    echo [!] Creating directory...
    mkdir "%GEMINI_COOKIE_PATH%"
)

REM ── 4. 确认依赖已安装 ──
echo.
echo [*] Checking dependencies...
pip install -r requirements.txt -q 2>nul
if errorlevel 1 (
    echo [!] Warning: pip install failed. Make sure Python and pip are available.
    echo [!] Continuing anyway...
)

REM ── 5. 启动 Bot ──
echo.
echo =============================================
echo   llmcord - Local Debug Mode
echo   Config:    config\config.yaml
echo   Cookies:   %GEMINI_COOKIE_PATH%
echo   Log Level: %LOGLEVEL%
echo =============================================
echo.
echo [*] Starting llmcord.py ...
echo [*] Press Ctrl+C to stop.
echo.

python llmcord.py

REM ── 退出处理 ──
echo.
echo [*] Bot stopped.
pause
