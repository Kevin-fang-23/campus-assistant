@echo off
chcp 65001 >nul
echo ============================================================
echo   正在停止校园助手服务...
echo ============================================================

REM ===== 读取启动时记录的端口（没有就回退默认 8000 / 5173）=====
set "PORT_B=8000"
set "PORT_F=5173"
if exist ".campus_ports" (
  for /f "usebackq tokens=1,2" %%a in (".campus_ports") do (
    set "PORT_B=%%a"
    set "PORT_F=%%b"
  )
  echo 已读取启动端口记录：后端 %PORT_B% / 前端 %PORT_F%
) else (
  echo 未找到端口记录，使用默认端口：后端 %PORT_B% / 前端 %PORT_F%
)

REM ===== 主手段：按端口终止占用进程（绕过不可靠的窗口标题匹配）=====
call :killport %PORT_B%
call :killport %PORT_F%

REM ===== 兜底 1：按窗口标题再清一次（关掉残留的 cmd 黑窗口）=====
echo 兜底清理：按窗口标题终止残留窗口...
taskkill /T /F /FI "WINDOWTITLE eq campus-backend" >nul 2>&1
taskkill /T /F /FI "WINDOWTITLE eq campus-frontend" >nul 2>&1

REM ===== 兜底 2：清理仍在跑的 vite / esbuild 子进程 =====
echo 兜底清理：终止残留的 node(vite) 进程...
for /f "skip=1 tokens=1" %%p in ('wmic process where "name='node.exe' and commandline like '%%vite%%'" get processid 2^>nul') do (
  if not "%%p"=="" (
    echo   终止 vite 进程 PID=%%p
    taskkill /T /F /PID %%p >nul 2>&1
  )
)

echo.
echo 校园助手服务已停止。
echo 按任意键关闭本窗口...
pause >nul
goto :eof

REM ===== 按端口找到占用进程并终止其整个进程树 =====
:killport
set "P=%1"
echo 检查端口 %P% 的占用进程...
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr /r ":%P% " ^| findstr "LISTENING"') do (
  echo   端口 %P% 占用进程 PID=%%p，正在终止（含子进程）...
  taskkill /T /F /PID %%p >nul 2>&1
)
goto :eof
