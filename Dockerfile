FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. 安装字体和系统库
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    libgl1 \
    libglib2.0-0 \
 && fc-cache -fv \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 2. 【关键步骤】删除 Matplotlib 的缓存
# 这样确保下次 import matplotlib 时，它会被迫重新扫描 /usr/share/fonts
RUN rm -rf /root/.cache/matplotlib

CMD ["python", "llmcord.py"]