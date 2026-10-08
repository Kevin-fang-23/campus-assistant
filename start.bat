@echo off
REM NOTE: Keep this file PURE ASCII. cmd's batch reader mis-parses batch files
REM containing multi-byte characters (UTF-8 or GBK) depending on the console
REM codepage - symptoms shift between runs and cannot be fully fixed.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title campus-main

REM ===== interpreter paths: prefer local venv / node, fallback to system =====
set "VENV_PY=C:/Users/86182/.workbuddy/binaries/python/envs/campus/Scripts/python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"
set "NPM=C:/Users/86182/.workbuddy/binaries/node/versions/22.22.2-2/npm.cmd"
if not exist "%NPM%" set "NPM=C:/Users/86182/.workbuddy/binaries/node/versions/22.22.2/npm.cmd"
if not exist "%NPM%" set "NPM=C:/Program Files/nodejs/npm.cmd"
if not exist "%NPM%" set "NPM=npm"

set "BACKEND=backend"
set "FRONTEND=frontend"

REM ===== 1) cleanup leftovers from last run (avoid port conflict) =====
echo [cleanup] stopping leftover campus services...
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1
ping -n 2 127.0.0.1 >nul

REM ===== 2) pick a free backend port (8000..8010) =====
set "PORT_B=8000"
:checkb
netstat -ano 2>nul | findstr /r ":%PORT_B% " | findstr "LISTENING" >nul
if !errorlevel! neq 0 goto :checkedb
set /a PORT_B+=1
if !PORT_B! gtr 8010 goto :portfail
goto :checkb
:checkedb
if not "%PORT_B%"=="8000" echo [note] port 8000 busy, backend moved to %PORT_B%

REM ===== 3) pick a free frontend port (5173..5175) =====
set "PORT_F=5173"
:checkf
netstat -ano 2>nul | findstr /r ":%PORT_F% " | findstr "LISTENING" >nul
if !errorlevel! neq 0 goto :checkedf
set /a PORT_F+=1
if !PORT_F! gtr 5175 goto :portfail
goto :checkf
:checkedf
if not "%PORT_F%"=="5173" echo [note] port 5173 busy, frontend moved to %PORT_F%
goto :record

:portfail
echo [error] no free port in range. Release ports and retry.
echo press any key to close...
pause >nul
goto :eof

REM ===== record ports so stop.bat can kill precisely =====
:record
echo %PORT_B% %PORT_F% > ".campus_ports"

REM ===== 4) start backend API =====
echo [1/2] starting backend API at http://127.0.0.1:%PORT_B% ...
start "campus-backend" cmd /k "cd /d %BACKEND% && "%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT_B%"

REM ===== 4.5) frontend toolchain self-check =====
REM vite 8 needs @rolldown native bindings; the old esbuild.exe check is obsolete.
REM Smoke test: node can import vite. On failure wipe @rolldown, npm install, retry.
REM If still broken, skip the dev server - backend serves the prebuilt site anyway.
set "FRONTEND_OK=1"
pushd "%FRONTEND%"
node -e "import('vite').then(function(){process.exit(0)},function(){process.exit(1)})" >nul 2>&1
if not errorlevel 1 goto :vite_ok
echo [self-check] frontend toolchain broken, repairing (about 1 min)...
if exist "node_modules\@rolldown" rmdir /s /q "node_modules\@rolldown"
call "%NPM%" install --no-save --no-audit --no-fund >nul 2>&1
node -e "import('vite').then(function(){process.exit(0)},function(){process.exit(1)})" >nul 2>&1
if not errorlevel 1 goto :vite_ok
set "FRONTEND_OK=0"
:vite_ok
popd

REM ===== 5) start frontend dev server (vite proxy points at backend port) =====
if "%FRONTEND_OK%"=="1" goto :startfe
echo [note] frontend repair failed, dev server skipped.
echo        prebuilt site still works at http://127.0.0.1:%PORT_B%
echo        manual fix: cd frontend, rmdir /s /q node_modules, npm install
goto :waitb
:startfe
echo [2/2] starting frontend at http://127.0.0.1:%PORT_F% ...
start "campus-frontend" cmd /k "cd /d %FRONTEND% && set BACKEND_PORT=%PORT_B% && call "%NPM%" run dev -- --port %PORT_F%"

REM ===== 6) wait for backend health (max 30s) then print banner =====
:waitb
echo waiting for backend to come up...
set /a tries=0
:wait
curl -s -o nul -m 2 http://127.0.0.1:%PORT_B%/health >nul 2>&1
if !errorlevel! equ 0 goto :banner
set /a tries+=1
if !tries! geq 30 goto :notready
ping -n 2 127.0.0.1 >nul
goto :wait
:notready
echo [warn] backend not ready after 30s, check campus-backend window log.
:banner
echo.
echo ============================================================
echo   campus assistant is up
echo     backend API : http://127.0.0.1:%PORT_B%   (docs at /docs)
echo     frontend    : http://127.0.0.1:%PORT_F%
echo ============================================================
echo services running. press any key to stop everything and close...
pause >nul

REM ===== stop: kill by port, then close child windows =====
echo stopping services...
call :killport %PORT_B%
call :killport %PORT_F%
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1
echo stopped.
endlocal
goto :eof

:killport
set "P=%1"
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r ":%P% " ^| findstr "LISTENING"') do taskkill /T /F /PID %%p >nul 2>&1
goto :eof
