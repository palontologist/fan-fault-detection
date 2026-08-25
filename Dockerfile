FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model checkpoint (if exists locally)
COPY checkpoints/best_model.pth checkpoints/best_model.pth 2>/dev/null || true

# Copy application code
COPY . .

# Set library path
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

EXPOSE 8001

CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8001}