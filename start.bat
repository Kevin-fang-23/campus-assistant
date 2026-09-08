@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title 校园助手 - 主控制台

REM ===== 解释器路径：优先用本机装好的 venv / node，缺失则回退系统命令 =====
set "VENV_PY=C:/Users/86182/.workbuddy/binaries/python/envs/campus/Scripts/python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"
set "NPM=C:/Users/86182/.workbuddy/binaries/node/versions/22.22.2-2/npm.cmd"
if not exist "%NPM%" set "NPM=C:/Users/86182/.workbuddy/binaries/node/versions/22.22.2/npm.cmd"
if not exist "%NPM%" set "NPM=C:/Program Files/nodejs/npm.cmd"
if not exist "%NPM%" set "NPM=npm"

set "BACKEND=backend"
set "FRONTEND=frontend"

REM ===== 1) 先清理上次可能残留的窗口（避免端口被旧进程占住导致打不开）=====
echo [清理] 停止上次可能残留的 campus 服务...
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1
timeout /t 1 >nul 2>&1

REM ===== 2) 选空闲后端端口（避免 8000 被占用导致启动失败）=====
set "PORT_B=8000"
:checkb
netstat -ano 2>nul | findstr /r ":%PORT_B% " | findstr "LISTENING" >nul
if !errorlevel!==0 (
  set /a PORT_B+=1
  if !PORT_B! gtr 8010 (
    echo [错误] 8000-8010 端口均被占用，请先释放端口后重试。
    echo 按任意键关闭本窗口...
    pause >nul
    goto :eof
  )
  goto :checkb
)
if not "%PORT_B%"=="8000" (
  echo [提示] 默认 8000 被占用，后端改用端口 %PORT_B%
)

REM ===== 3) 选空闲前端端口 =====
set "PORT_F=5173"
:checkf
netstat -ano 2>nul | findstr /r ":%PORT_F% " | findstr "LISTENING" >nul
if !errorlevel!==0 (
  set /a PORT_F+=1
  if !PORT_F! gtr 5175 (
    echo [错误] 前端端口 5173-5175 被占用，请先释放端口后重试。
    echo 按任意键关闭本窗口...
    pause >nul
    goto :eof
  )
  goto :checkf
)
if not "%PORT_F%"=="5173" (
  echo [提示] 默认 5173 被占用，前端改用端口 %PORT_F%
)

REM ===== 记录端口，供 stop.bat 精准停止（核心：不再只靠窗口标题）=====
echo %PORT_B% %PORT_F% > ".campus_ports"

REM ===== 4) 启动后端 API（固定绑 127.0.0.1，便于探活；去掉 --reload 减少子进程，更稳定）=====
echo [1/2] 启动后端 API (http://127.0.0.1:%PORT_B%) ...
start "campus-backend" cmd /k "cd /d %BACKEND% && %VENV_PY% -m uvicorn app.main:app --host 127.0.0.1 --port %PORT_B%"

REM ===== 4.5) 自检：esbuild 原生二进制缺失会导致 vite 起不来（可能被杀毒/清理误删），缺失则自动修复 =====
REM 注意：直接跑 npm install 修复不了——@esbuild/win32-x64 的 package.json 仍在，
REM npm 判定"已是最新"会跳过，必须先删掉整个包目录再重装。
if not exist "%FRONTEND%\node_modules\@esbuild\win32-x64\esbuild.exe" (
  echo [自检] 检测到前端 esbuild 二进制缺失，正在自动修复（约 1 分钟）...
  if exist "%FRONTEND%\node_modules\@esbuild\win32-x64" (
    rmdir /s /q "%FRONTEND%\node_modules\@esbuild\win32-x64"
  )
  pushd "%FRONTEND%"
  call "%NPM%" install --no-save --no-audit --no-fund
  popd
  if not exist "%FRONTEND%\node_modules\@esbuild\win32-x64\esbuild.exe" (
    echo.
    echo [错误] esbuild 自动修复失败。请手动执行：
    echo        cd frontend
    echo        rmdir /s /q node_modules\@esbuild\win32-x64
    echo        npm install
    echo.
    pause
    goto :eof
  )
  echo [自检] esbuild 已修复，继续启动。
)

REM ===== 5) 启动前端（把后端端口传入 vite 代理，保证联动）=====
echo [2/2] 启动前端 (http://127.0.0.1:%PORT_F%) ...
start "campus-frontend" cmd /k "cd /d %FRONTEND% && set BACKEND_PORT=%PORT_B% && %NPM% run dev -- --port %PORT_F%"

REM ===== 6) 探活后端，确认起来再打印地址 =====
echo 正在等待后端启动...
set /a tries=0
:wait
curl -s -o nul -m 2 http://127.0.0.1:%PORT_B%/health >nul 2>&1
if !errorlevel!==0 (
  set /a tries+=1
  if !tries! lss 30 (
    timeout /t 1 >nul
    goto :wait
  )
)
if !tries! geq 30 (
  echo [警告] 后端 30 秒内未就绪，请查看 campus-backend 窗口的日志。
)

echo.
echo ============================================================
echo   校园事务智能助手已启动
echo     后端 API : http://127.0.0.1:%PORT_B%   (接口文档 /docs)
echo     前端页面 : http://127.0.0.1:%PORT_F%
echo ============================================================
echo 服务正在运行。按任意键将停止全部服务并关闭本窗口...
pause >nul

REM ===== 停止：按端口精准终止，连带关闭 uvicorn / vite 子进程 =====
echo 正在停止服务...
call :killport %PORT_B%
call :killport %PORT_F%
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1
echo 已停止。
endlocal
goto :eof

REM ===== 按端口找到占用进程并终止其整个进程树 =====
:killport
set "P=%1"
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r ":%P% " ^| findstr "LISTENING"') do (
  echo   端口 %P% 占用进程 PID=%%p，正在终止（含子进程）...
  taskkill /T /F /PID %%p >nul 2>&1
)
goto :eof
