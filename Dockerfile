FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_task2.txt .
RUN pip install --upgrade pip && pip install -r requirements_task2.txt

COPY predict_task2.py .
COPY border_marker_cleanup.py .
COPY weights/ ./weights/

ENTRYPOINT ["python", "predict_task2.py", "--input", "/input", "--output", "/output"]
