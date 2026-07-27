@echo off

if "%~1"=="" goto cli
if "%~1"=="cli" goto cli
if "%~1"=="web" goto web
if "%~1"=="discord" goto discord
if "%~1"=="discord-stop" goto discord_stop
if "%~1"=="discord-restart" goto discord_restart
if "%~1"=="status" goto status
if "%~1"=="setup" goto setup
if "%~1"=="help" goto help
goto help

:cli
cls
echo ========================================
echo   Hermes Agent -- Interactive CLI
echo ========================================
echo.
echo Model: deepseek/deepseek-v4-flash (high reasoning)
echo Provider: openrouter
echo.
echo Type your prompts below. Press Ctrl+C or type 'exit' to quit.
echo.
wsl hermes
exit /b 0

:web
cls
echo ========================================
echo       Hermes Web Dashboard
echo ========================================
echo.
echo Starting dashboard at http://localhost:8080 ...
echo.
start "" "http://localhost:8080"
wsl hermes dashboard
echo.
echo Dashboard stopped.
pause
exit /b 0

:discord
cls
echo ========================================
echo       Hermes Discord Gateway
echo ========================================
echo.
wsl sudo hermes gateway start --system 2>nul
timeout /t 2 /nobreak >nul
echo Checking status...
echo.
wsl hermes gateway status --system
echo.
echo Bot should now be online as ^'Hermes Agent#7835^'.
echo.
pause
goto status

:discord_stop
cls
echo ========================================
echo     Stopping Discord Gateway
echo ========================================
echo.
wsl sudo hermes gateway stop --system 2>nul
echo Stopped.
echo.
pause
goto status

:discord_restart
cls
echo ========================================
echo    Restarting Discord Gateway
echo ========================================
echo.
wsl sudo hermes gateway restart --system 2>nul
timeout /t 2 /nobreak >nul
echo Checking status...
echo.
wsl hermes gateway status --system
echo.
pause
goto status

:status
cls
echo ========================================
echo       Gateway Status
echo ========================================
echo.
wsl hermes gateway status --system
echo.
pause
exit /b 0

:setup
cls
echo ========================================
echo     Hermes Gateway Setup Wizard
echo ========================================
echo.
wsl hermes gateway setup
echo.
pause
exit /b 0

:help
echo ========================================
echo       Hermes Agent Launcher
echo ========================================
echo.
echo Usage: run-hermes.bat ^<command^>
echo.
echo   (no args)      - Start Interactive CLI (default)
echo   cli            - Start Interactive CLI
echo   web            - Start the Web Dashboard (port 8080)
echo   discord        - Start the Discord Gateway via systemd
echo   discord-stop   - Stop the Discord Gateway
echo   discord-restart- Restart the Discord Gateway
echo   status         - Check gateway status
echo   setup          - Run gateway setup wizard
echo   help           - Show this menu
echo.
exit /b 0
