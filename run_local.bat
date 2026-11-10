@echo off
chcp 65001 >nul
title llmcord - Local Debug

REM ---------------------------------------------
REM  llmcord 本地调试启动脚本
REM  用法: 双击此 bat 或在命令行运行
REM ---------------------------------------------

REM 切换到脚本所在目录
cd /d "%~dp0"

REM -- 1. Python 虚拟环境 (如有) --
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [*] No virtual environment found, using system Python.
)

REM -- 2. 设置环境变量 --
set LOGLEVEL=INFO

REM -- Gemini AIStudio2API VPS 远程直连地址 (若设置则优先覆盖 config.yaml) --
REM 如需本地直连 VPS 2048 端口，取消下一行的 rem 并填入 VPS 公网 IP 或域名与本地 API 密钥:
rem set GEMINI_BASE_URL=http://<VPS_IP>:2048/v1
rem set GEMINI_API_KEY=sk-aistudio-123456

REM -- 3. 确认依赖已安装 --
echo.
echo [*] Checking dependencies...
pip install -r requirements.txt -q 2>nul
if errorlevel 1 (
    echo [!] Warning: pip install failed. Make sure Python and pip are available.
    echo [!] Continuing anyway...
)

REM -- 4. 启动 Bot --
echo.
echo =============================================
echo   llmcord - Local Debug Mode
echo   Config:    config\config.yaml
if defined GEMINI_BASE_URL (
    echo   Gemini:    %GEMINI_BASE_URL% [ENV Override]
) else (
    echo   Gemini:    from config.yaml
)
echo   Log Level: %LOGLEVEL%
echo =============================================
echo.
echo [*] Starting llmcord.py ...
echo [*] Press Ctrl+C to stop.
echo.

python llmcord.py

REM -- 退出处理 --
echo.
echo [*] Bot stopped.
pause
