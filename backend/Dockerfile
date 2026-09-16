FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（部分可选依赖如 paddleocr 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 默认装精简集：排除 paddleocr/paddlepaddle（500MB+），否则 Render/Railway
# 免费层（512MB 内存 + 构建时限）大概率构建超时或启动时 OOM 反复重启。
# 本地用 docker compose 且确实需要 paddle 时再显式覆盖：
#   docker compose build --build-arg REQUIREMENTS=requirements.txt
#
# 装到 /tmp 而非 /app：下面的 `COPY . .` 会把仓库里的完整版 requirements.txt
# 一并拷进来，若在此处重命名成 /app/requirements.txt 就会被覆盖，
# 镜像里留下一个与实际已装依赖不符的文件，排障时会误导人。
ARG REQUIREMENTS=requirements-deploy.txt
COPY ${REQUIREMENTS} /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . .

# 默认 SQLite；用 docker-compose 时通过环境变量切到 PostgreSQL
EXPOSE 8000
# 云平台（Render/Railway 等）会把端口写入 $PORT；默认 8000 便于本地 docker 运行
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
