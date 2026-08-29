#!/usr/bin/env bash
# 一键启动校园事务智能助手（后端 + 前端）。
# 用法： bash start.sh   或   ./start.sh
set -u

cd "$(dirname "$0")"

# 解释器：优先用本机 venv，缺失则回退系统 python3 / python
VENV_PY="${VENV_PY:-python3}"
if [ ! -x "$VENV_PY" ] && command -v python3 >/dev/null 2>&1; then VENV_PY=python3; fi
if ! command -v "$VENV_PY" >/dev/null 2>&1; then VENV_PY=python; fi

NPM="${NPM:-npm}"
BACKEND="backend"
FRONTEND="frontend"
PORT_F=5173

# 自动挑选空闲的后端端口（避免 8000 被占用）
PORT_B=8000
port_free() {
  # 优先 lsof，回退 netstat
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1 && return 1 || return 0
  fi
  netstat -an 2>/dev/null | grep -E "[:.]${1}[[:space:]]" >/dev/null 2>&1 && return 1 || return 0
}
while ! port_free "$PORT_B"; do
  PORT_B=$((PORT_B + 1))
  if [ "$PORT_B" -gt 8010 ]; then
    echo "[错误] 8000-8010 端口均被占用，请先释放端口后重试。"
    exit 1
  fi
done
if [ "$PORT_B" -ne 8000 ]; then
  echo "[提示] 默认 8000 被占用，后端改用端口 $PORT_B"
fi

# 1) 后端
echo "[1/2] 启动后端 API (http://localhost:$PORT_B) ..."
BACKEND_PORT="$PORT_B" "$VENV_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT_B" --reload >/tmp/campus-backend.log 2>&1 &
PID_B=$!

# 2) 前端（把后端端口传入 vite 代理）
echo "[2/2] 启动前端 (http://localhost:$PORT_F) ..."
( cd "$FRONTEND" && BACKEND_PORT="$PORT_B" "$NPM" run dev -- --port "$PORT_F" ) >/tmp/campus-frontend.log 2>&1 &
PID_F=$!

echo
echo "============================================================"
echo "  校园事务智能助手已启动"
echo "    后端 API : http://localhost:$PORT_B   (接口文档 /docs)"
echo "    前端页面 : http://localhost:$PORT_F"
echo "============================================================"
echo "  按 Ctrl+C 停止全部服务..."

trap 'echo; echo 正在停止服务...; kill "$PID_B" "$PID_F" 2>/dev/null; wait 2>/dev/null; echo 已停止。' INT TERM
wait
