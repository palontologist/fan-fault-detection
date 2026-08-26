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

# Copy model checkpoints from build context
COPY checkpoints/ ./checkpoints/
# Verify copy; if empty, lazy loading at startup will use optimal thresholds from optimal_thresholds.json

# Copy application code
COPY . .

# Set library path
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

EXPOSE 8000

CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300 --timeout-graceful-shutdown 30