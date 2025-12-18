FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py run.py ./

# 创建必要的目录
RUN mkdir -p logs models

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 运行应用
CMD ["python", "run.py"]

