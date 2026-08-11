FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.py ./
COPY service ./service

RUN mkdir -p logs models

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "run.py"]
