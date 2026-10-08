@echo off
REM NOTE: Keep this file PURE ASCII (see start.bat header for why).
echo ============================================================
echo   stopping campus assistant services...
echo ============================================================

REM ===== read ports recorded by start.bat (fallback to defaults) =====
set "PORT_B=8000"
set "PORT_F=5173"
if not exist ".campus_ports" goto :nolist
for /f "usebackq tokens=1,2" %%a in (".campus_ports") do (
  set "PORT_B=%%a"
  set "PORT_F=%%b"
)
:nolist
echo ports: backend %PORT_B% / frontend %PORT_F%

REM ===== primary: kill listeners on the recorded ports =====
call :killport %PORT_B%
call :killport %PORT_F%

REM ===== fallback 1: close leftover cmd windows by title =====
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1

echo.
echo campus assistant services stopped.
echo press any key to close this window...
pause >nul
goto :eof

:killport
set "P=%1"
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r ":%P% " ^| findstr "LISTENING"') do taskkill /T /F /PID %%p >nul 2>&1
goto :eof
