FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        python3-venv \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv

COPY requirements/base.txt /tmp/base.txt
COPY requirements/ml.txt /tmp/ml.txt
COPY requirements/research.txt /tmp/research.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        -r /tmp/base.txt \
        -r /tmp/ml.txt \
        -r /tmp/research.txt

RUN python -m ipykernel install \
        --sys-prefix \
        --name model1-ai \
        --display-name "Model 1 AI"

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root"]
