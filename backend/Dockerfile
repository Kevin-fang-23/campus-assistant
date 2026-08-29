FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（部分可选依赖如 paddleocr 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 默认 SQLite；用 docker-compose 时通过环境变量切到 PostgreSQL
EXPOSE 8000
# 云平台（Render/Railway 等）会把端口写入 $PORT；默认 8000 便于本地 docker 运行
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
