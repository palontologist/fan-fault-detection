FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including ffmpeg for pydub MP3/FLAC/OGG support)
RUN apt-get update && apt-get install -y \
    libstdc++6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model checkpoints from GitHub release (v2.2.0 - optimal thresholds)
RUN python -c "import urllib.request, os; os.makedirs('checkpoints', exist_ok=True); urllib.request.urlretrieve('https://github.com/palontologist/fan-fault-detection/releases/download/v2.2.0/best_model.pth', 'checkpoints/best_model.pth')"
# Download per-ID models
RUN for i in 00 01 02 03 04 05 06; do python -c "import urllib.request, os; os.makedirs('checkpoints', exist_ok=True); urllib.request.urlretrieve('https://github.com/palontologist/fan-fault-detection/releases/download/v2.2.0/best_model_id_${i}.pth', 'checkpoints/best_model_id_${i}.pth')"; done

# Copy application code
COPY . .

# Set library path
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

EXPOSE 8000

CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300 --timeout-graceful-shutdown 30