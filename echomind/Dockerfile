# EchoMind 2.0 - Dockerfile
# 多阶段构建：最小化生产镜像体积

FROM python:3.10-slim AS builder

WORKDIR /app

# 安装 Poetry
RUN pip install --no-cache-dir poetry==1.8.0

# 复制依赖文件
COPY pyproject.toml poetry.lock* ./

# 安装依赖（不包括开发依赖）
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# ---------- 生产阶段 ----------
FROM python:3.10-slim AS production

WORKDIR /app

# 复制已安装的依赖
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY app.py .
COPY .env.example .env

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash echomind \
    && chown -R echomind:echomind /app
USER echomind

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# 启动命令
CMD ["uvicorn", "app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]