FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 1. 安装基础工具 (我们需要 wget 或 curl 来下载字体)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    fontconfig \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

 # 2. 【核心修改】手动下载 Noto Sans CJK SC (简体中文专用版)
# 修正了 URL 和文件名
RUN mkdir -p /usr/share/fonts/opentype/noto \
    && wget -qO /usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf \
       "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf" \
    && wget -qO /usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf \
       "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf" \
    && fc-cache -fv

# 2. 【关键步骤】删除 Matplotlib 的缓存
# 这样确保下次 import matplotlib 时，它会被迫重新扫描 /usr/share/fonts
RUN rm -rf /root/.cache/matplotlib

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码（.dockerignore 会排除本地敏感文件）
COPY . .

CMD ["python", "llmcord.py"]
