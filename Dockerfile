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

# Download model checkpoint from GitHub release
RUN python -c "import urllib.request, os; os.makedirs('checkpoints', exist_ok=True); urllib.request.urlretrieve('https://github.com/palontologist/fan-fault-detection/releases/download/v2.1.0/stgram_mfn_best.pth', 'checkpoints/stgram_mfn_best.pth')"

# Copy application code
COPY . .

# Set library path
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

EXPOSE 8001

CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8001}