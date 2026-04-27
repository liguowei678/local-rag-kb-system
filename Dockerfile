FROM python:3.11

WORKDIR /app

# ==================== 环境变量配置 ====================
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_CACHE=/app/models/cache

# 统一模型根目录，禁止联网下载
ENV MODELS_ROOT=/app/models
ENV DOCLING_DISABLE_MODEL_DOWNLOADS=1
ENV RAPIDOCR_DISABLE_DOWNLOADS=1
ENV HF_DATASETS_CACHE=/app/models/docling
ENV HUGGINGFACE_HUB_CACHE=/app/models/docling
ENV HF_HUB_OFFLINE=1

# ==================== 系统依赖安装 ====================
# 安装Docling需要的系统依赖（OpenCV需要libGL.so.1）
# 在Debian 13中，安装完整的Mesa OpenGL库
RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ==================== 创建统一模型目录结构 ====================
RUN mkdir -p /app/models/{docling,rapidocr,reranker,cache}

# ==================== 复制预下载的Docling模型 ====================
COPY ./docling-models /app/models/docling/

# ==================== 复制预下载的OCR模型 ====================
# RapidOCR会在多个位置查找模型，我们复制到所有可能的位置
# 1. 系统Python包目录（RapidOCR默认查找）
RUN mkdir -p /usr/local/lib/python3.11/site-packages/rapidocr/models
COPY ./docling-models/EasyOcr/ /usr/local/lib/python3.11/site-packages/rapidocr/models/

# 2. 统一模型目录
COPY ./docling-models/EasyOcr/ /app/models/rapidocr/models/

# 3. 用户缓存目录
RUN mkdir -p /root/.cache/rapidocr/models
COPY ./docling-models/EasyOcr/ /root/.cache/rapidocr/models/

# ==================== 复制预下载的重排序模型 ====================
COPY ./models/reranker /app/models/reranker/

RUN chmod -R 755 /app/models

# ==================== Python依赖安装 ====================
COPY requirements.txt .

# 使用单个可靠的镜像源，避免冲突
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 120 \
    --retries 5

# ==================== 应用代码 ====================
COPY . .

RUN mkdir -p /app/data/uploads /app/logs

# 添加当前目录到Python路径，确保可以导入src模块
ENV PYTHONPATH=/app

# ==================== 简单验证 ====================
RUN python -c "import docling; print('Docling导入成功')" && \
    python -c "import langchain_text_splitters; print('LangChain分块器导入成功')" && \
    python -c "import rapidocr; print('RapidOCR导入成功')" && \
    python -c "import os; print('环境变量:'); print('MODELS_ROOT:', os.environ.get('MODELS_ROOT')); print('HF_HOME:', os.environ.get('HF_HOME'))"

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]