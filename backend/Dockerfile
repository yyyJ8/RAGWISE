# 使用Python 3.12作为基础镜像
FROM python:3.12-slim

# 设置环境变量
ENV LANG=C.UTF-8 
ENV LC_ALL=C.UTF-8 
ENV TZ=Asia/Shanghai
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml poetry.lock /app/
COPY backend/ /app/backend/
COPY rag/ /app/rag/
COPY bge-reranker-large/ /app/bge-reranker-large/

# 安装Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# 配置Poetry
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

# 安装依赖
RUN poetry install --no-root

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install sentence-transformers
# 暴露端口
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# 启动命令
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]